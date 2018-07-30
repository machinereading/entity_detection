
from DetectorModel import LSTM_CRFModel
from WordBasedDetector import WordBasedLSTMCRF
import tensorflow as tf
import argparse
import configparser
import NERUtil
import json
import os
from NERUtil import test_metric
from datetime import datetime
# args = argparse.Namespace(char_dim=300,
# 						  cell_type="LSTM", 
# 						  save_path="saves/koreanner_300_lstm_without_word_vector_wikipedia/", 
# 						  hidden_layer_size=256, 
# 						  num_layers=2,
# 						  use_word_embedding=False,
# 						  learning_rate=0.005,
# 						  predict_mode=True
# 						  )
SAVE_ITER = 10
class NER():
	def __init__(self, args):
		self.model = LSTM_CRFModel(args)# if args.model == "char" else WordBasedLSTMCRF(args)
		self.args = args
	def _train(self, sess, train_file_name, dev_file_name, train_encode_func, eval_encode_func):
		if sess is None:
			sess = tf.Session()
			sess.__enter__()
		if not os.path.isdir(self.args.save_path):
			os.mkdir(self.args.save_path[:-1])
		with open(self.args.save_path+"args.json", "w") as f:
			json.dump(vars(self.args), f, indent="\t")
		self.args.mode = "train"
		saver = self.model.load_trained_session(sess, self.args)
		it = 0
		print()
		train_result = {}
		if os.path.isfile(self.args.save_path+"result.json"):
			with open(self.args.save_path+"result.json") as f:
				train_result = json.load(f)
		
		prefix = max(list(map(lambda x: int(x), train_result.keys()))) if len(train_result) > 0 else 0
		for it in range(1, self.args.epoch+1):
			with open(train_file_name, encoding="UTF8") as train:
				testset_gen = self.model.generate_trainset(train_encode_func, train, self.args)
				b = 0
				for sv, seqlen, label in testset_gen:
					try:
						_, loss = self.model.train(sess, sv, seqlen, label)
						b += len(seqlen)
						print("\r%s - Batch %d: %.2f" %(str(datetime.now()), b, loss), end="", flush=True)
					except Exception:
						pass
				print()
			
			if it % SAVE_ITER == 0:
				saver.save(sess, self.args.save_path+"%d.ckpt" % self.args.char_dim)
				print("Epoch %d: Model saved" % (it+prefix))
				efs, sfs = [0,0], [0,0]
				c = 0
				with open(dev_file_name, encoding="UTF8") as dev:
					for sentence, label in eval_encode_func(dev):
						pred, _ = self.model.predict_sent(sess, sentence)
						ef, sf = test_metric(sentence, label, pred)
						if ef == -1 and sf == -1: continue
						efs[0] += ef[0]
						efs[1] += ef[1]
						sfs[0] += sf[0]
						sfs[1] += sf[1]
						c += 1
						if c > 1000: break
					if c > 0:
						tef, tsf, wef, wsf = efs[0] / c, sfs[0] / c, efs[1] / c, sfs[1] / c
						train_result[it+prefix] = (tef, tsf, wef, wsf)
						print(train_result[it+prefix])
			with open(self.args.save_path+"result.json", "w") as f:
				json.dump(train_result, f, indent="\t")

	def _evaluate(self, input_data, generate_func, sess=None):
		flag = sess is None
		if flag:
			sess = tf.Session()
			sess.__enter__()
		print()
		self.model.load_trained_session(sess, self.args)
		efs = [0,0]
		sfs = [0,0]
		c = 0
		for sent, answer in generate_func(input_data):
			predict_label, _ = self.model.predict_sent(sess, sent)
			ef, sf = test_metric(sent, answer, predict_label)
			if ef == -1 and sf == -1: continue
			efs[0] += ef[0]
			efs[1] += ef[1]
			sfs[0] += sf[0]
			sfs[1] += sf[1]
			c += 1
			if c % 100 == 0:
				tef, tsf, wef, wsf = efs[0] / c, sfs[0] / c, efs[1] / c, sfs[1] / c
				print("\r%d: %.2f %.2f %.2f %.2f" % (c, tef, tsf, wef, wsf), end="", flush=True)
		tef, tsf, wef, wsf = efs[0] / c, sfs[0] / c, efs[1] / c, sfs[1] / c
		print("\r%d: %.2f %.2f %.2f %.2f" % (c, tef, tsf, wef, wsf), end="", flush=True)
		if flag: sess.__exit__()
	
	def predict(self, input_json, sess=None):
		flag = sess is None
		if flag:
			print("no session")
			sess = tf.Session()
			sess.__enter__()
		self.args.mode = "predict"
		self.model.load_trained_session(sess, self.args)

		sentence = input_json if type(input_json) is str else input_json["sentence"]
		output = {}
		output["sentence"] = sentence
		entities = []
		predict_label, score = self.model.predict_sent(sess, sentence)
		# for c, l in zip(sentence, predict_label):
		# 	print(c, l)
		predicted_entities = NERUtil.decode_label(sentence, predict_label, score)
		for name, ty, sin, ein, score in predicted_entities:
			x = {}
			x["name"] = name
			x["type"] = ty 
			x["start_index"] = sin
			x["end_index"] = ein
			x["score"] = float(score)
			entities.append(x)
		output["entities"] = entities
		if flag: sess.__exit__()
		return output




if __name__ == "__main__":
	from DataPrepareModule import DataPreparer as dp
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, default="char", help="model type, must be one of char, word, word_char")
	parser.add_argument("--morph_type", type=str, default="ko", help="if using word type, specify what morph model. one of ko, kk")
	parser.add_argument('--char_dim', type=int, help='dimension of char vector', default=300)
	parser.add_argument('--hidden_layer_size', type=int, default=256, help='hidden dimension of rnn')
	parser.add_argument('--num_layers', type=int, default=2, help='number of layers in rnn')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
	parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
	parser.add_argument('--save_path', type=str, default="saves/koreanner_300_lstm_without_word_vector_wikipedia/", help="path of saved model")
	parser.add_argument("--learning_rate", type=float, default=0.005, help="learning rate of model")
	parser.add_argument("--batch_maximum", type=int, default=-1, help="maximum batch iteration, -1 for infinite")
	parser.add_argument("--batch_repeat", type=bool, default=False, help="")
	parser.add_argument("--cell_type", type=str, default="LSTM", help="Cell type of RNN cell. must be one of RNN, LSTM, GRU")
	parser.add_argument("--use_word_embedding", type=str, default="False")
	parser.add_argument("--load_from_file", default=False, action="store_true")

	args = parser.parse_args()
	if not args.save_path.endswith("/"):
		args.save_path += "/"
	module = NER(args)
	with tf.Session() as sess:
		# with open("corpus/test.txt", encoding="UTF8") as rf:
		# 	result = []
		# 	result_raw = []
		# 	for sentence in rf.readlines():
		# 		if len(sentence.strip()) == 0: continue
		# 		x = module.predict(sentence.strip(), sess)
		# 		result.append(x)
		# 		result_raw.append(NERUtil.mark(sentence.strip(), x["entities"]))
		# 	with open("testresult.json", "w", encoding="UTF8") as wf:
		# 		json.dump(result, wf, ensure_ascii=False, indent="\t")
		# 	with open("testresult.txt", "w", encoding="UTF8") as wf:
		# 		for item in result_raw:
		# 			wf.write(item+"\n")
		module._train(sess, "corpus/train2.txt", "corpus/golden_all.txt", dp.corpus, dp.wikipedia_golden)
		# path = "corpus/golden/"
		# for item in os.listdir(path):
		# 	if os.path.isfile(item+".json"): continue
		# 	results = []
		# 	with open(path+item, encoding="UTF8") as f:
		# 		for sentence, _ in dp.wikipedia_golden(f):
		# 			results.append(module.predict(sentence, sess))
		# 	with open(item+".json", "w", encoding="UTF8") as f:
		# 		json.dump(results, f, ensure_ascii=False, indent="\t")
		# 	with open(path+item, encoding="UTF8") as f:
		# 		print()
		# 		print(item)
		# # f = open("etri_corpus/corpus/EXOBRAIN_NE_CORPUS_10000.txt", encoding="UTF8")
		# 		module._evaluate(f, dp.wikipedia_golden, sess)
			
				# f.close()