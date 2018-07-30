import tensorflow as tf
import KoreanUtil
import numpy as np
import os
from NERUtil import encode_label
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score

WORD_EMBEDDING_SIZE = 300

SAVE_ITER = 10
np.random.seed(10)

BILOU_TRANSITION_MATRIX = np.genfromtxt("crf_prob.csv", delimiter=",")
wv = KeyedVectors.load_word2vec_format("vector/ko.bin", binary=True)
class LSTM_CRFModel():
	def __init__(self, args):
		if args.load_from_file:
			self.load_arguments(args)
		self.initialized_variables = False
		self.char_embedding_size = args.char_dim
		
		# prepare dataset
		with tf.name_scope("LSTM_CRF"):
			label_size = 17
			self.use_word_embedding = args.use_word_embedding
			self.save_path = args.save_path if args.save_path is not None else "saves/no_word_vector_%d" % args.char_dim
			# self.batch_sentence = tf.placeholder(tf.string, [BATCH_SIZE, ]) # ndarray of sentence
			# self.char_embedding = tf.placeholder(tf.float32, [BATCH_SIZE, None, CHAR_EMBEDDING_SIZE]) # 하나의 char마다 3*30 embedding을 수행한 값들 200, ?, 30
			# self.word_embedding = tf.placeholder(tf.float32, [BATCH_SIZE, None, WORD_EMBEDDING_SIZE]) # 하나의 word마다 fasttext vector를 가져와
			if args.use_word_embedding == "True":
				self.sentence_embedding = tf.placeholder(tf.float32, [None, None, args.char_dim+WORD_EMBEDDING_SIZE])
			else:
				self.sentence_embedding = tf.placeholder(tf.float32, [None, None, args.char_dim])
			self.sentence_length = tf.placeholder(tf.int32, [None, ]) # length of sentence -> CHAR LENGTH!!!
			self.labels = tf.placeholder(tf.int32, [None, None])
			# self.char_embedding = tf.py_func(self.generate_char_embedding, [self.batch_sentence, self.sentence_length], tf.float32)
			# self.word_embedding = tf.py_func(self.generate_word_embedding, [self.batch_sentence], tf.float32)
			
			
			cell_caller = {"RNN": tf.nn.rnn_cell.BasicRNNCell, "LSTM": tf.nn.rnn_cell.BasicLSTMCell, "GRU": tf.nn.rnn_cell.GRUCell}
			

			# BiLSTM Layer
			cell_fw = tf.contrib.rnn.MultiRNNCell([cell_caller[args.cell_type](args.hidden_layer_size) for _ in range(args.num_layers)])
			cell_bw = tf.contrib.rnn.MultiRNNCell([cell_caller[args.cell_type](args.hidden_layer_size) for _ in range(args.num_layers)])
			(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.sentence_embedding, sequence_length=self.sentence_length, dtype=tf.float32)

			# CRF Layer TODO
			context_rep = tf.concat([output_fw, output_bw], axis=-1)
			# W = tf.get_variable("W", shape=[2*HIDDEN_LAYER_SIZE, label_size], dtype=tf.float32)

			# b = tf.get_variable("b", shape=[label_size], dtype=tf.float32, initializer=tf.zeros_initializer())

			ntime_steps = tf.shape(context_rep)[1]
			context_rep_flat = tf.reshape(context_rep, [-1, 2*args.hidden_layer_size])
			w = tf.get_variable("W", shape=[2*args.hidden_layer_size, label_size], dtype=tf.float32)
			b = tf.get_variable("b", shape=[label_size], dtype=tf.float32)
			# self.pred = tf.layers.dense(inputs=context_rep, units=label_size, use_bias=True)
			pred = tf.matmul(context_rep_flat, w) + b

			# print(self.pred.shape)
			self.scores = tf.reshape(pred, [-1, ntime_steps, label_size])# BATCH, ?, 5
			# print(scores.shape)
			# mask = tf.sequence_mask(self.sentence_length)
			# self.log_likelihood, self.transition_matrix = tf.contrib.crf.crf_log_likelihood(scores, self.labels, self.sentence_length)
			self.log_likelihood, self.transition_matrix = tf.contrib.crf.crf_log_likelihood(self.scores, self.labels, self.sentence_length, tf.convert_to_tensor(BILOU_TRANSITION_MATRIX, dtype=tf.float32))
			
			# train
			self.loss = tf.reduce_mean(-self.log_likelihood)
			# self.softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pred, labels=tf.one_hot(self.labels, label_size)))
			self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(self.loss)
			self.viterbi_seq, self.seq_score = tf.contrib.crf.crf_decode(self.scores, self.transition_matrix, self.sentence_length)

			# self.accuracy = tf.reduce_mean(tf.cast(tf.boolean_mask(tf.equal(self.viterbi_seq, self.labels), mask), tf.float32))
			# self.optimizer = tf.train.AdamOptimizer(learning_rate=0.05).minimize(self.softmax_loss)

	def cost(self, output, target):
		# Compute cross entropy for each frame.
		cross_entropy = target * tf.log(output)
		cross_entropy = -tf.reduce_sum(cross_entropy, 2)
		mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
		cross_entropy *= mask
		# Average over actual sequence lengths.
		cross_entropy = tf.reduce_sum(cross_entropy, 1)
		cross_entropy /= tf.reduce_sum(mask, 1)
		return tf.reduce_mean(cross_entropy)

	def predict_sent(self, sess, sentence):
		if len(sentence) == 0: return None
		seqlen = [len(sentence)]
		encode_func = self.generate_char_embedding if self.use_word_embedding != "True" else self.generate_concatenated_embedding_from_sentence
		sv = np.array([encode_func(sentence)])
		a, b = self.predict_label(sess, sv, seqlen)
		return a[0], b[0]
	
	# for validation
	def predict_label(self, sess, sv, seqlen):
		feed = {self.sentence_embedding: sv, self.sentence_length: seqlen}
		return sess.run([self.viterbi_seq, self.scores], feed_dict=feed)
		

	def train(self, sess, svs, seqlen, list_of_labels, saver=None, wf=None):
		feed = {self.sentence_embedding: svs, self.sentence_length: seqlen, self.labels: list_of_labels}
		seq, loss, _= sess.run([self.viterbi_seq, self.loss, self.optimizer], feed_dict=feed)
		return seq, loss

	
	def load_trained_session(self, sess, args):
		if self.initialized_variables:
			return
		saver = tf.train.Saver(tf.global_variables())
		try:
			saver.restore(sess, args.save_path+"%d.ckpt" % args.char_dim)
			print("saved model restored")
		except Exception:
			if args.mode == "predict":
				raise Exception("No model exists! (%s)" % (args.save_path))
			sess.run(tf.global_variables_initializer())
		self.initialized_variables = True
		return saver


	def generate_concatenated_embedding_from_sentence(self, sentence, maxlen=None):
		sv = []
		for word in sentence.split(" "):
			wvs = wv[word] if word in wv else np.zeros([WORD_EMBEDDING_SIZE,])
			for char in word:
				base = KoreanUtil.char_embedding(char, self.char_embedding_size)
				c = np.concatenate((base, wvs), axis=0)
				sv.append(c)
			sv.append(np.concatenate((KoreanUtil.char_embedding(" ", self.char_embedding_size), np.zeros([WORD_EMBEDDING_SIZE,])), axis=0)) # 공백은 그냥 0벡터
		del sv[-1]
		if maxlen is not None:
			while len(sv) < maxlen:
				sv.append(np.zeros([self.char_embedding_size+WORD_EMBEDDING_SIZE,]))
		# print(len(sv))
		return np.array(sv)

	def generate_char_embedding(self, sentence, maxlen=None):
		sv = []
		for char in sentence:
			base = KoreanUtil.char_embedding(char, self.char_embedding_size)
			sv.append(base)
		if maxlen is not None:
			while len(sv) < maxlen:
				sv.append(np.zeros([self.char_embedding_size,]))
		# print(len(sv))
		return np.array(sv)
	# Batch size * max length of sentence in batch
	def generate_trainset(self, generator_fn, generator_file, args):
		c = 0
		generator = generator_fn(generator_file)
		while True:
			if args.batch_maximum > -1 and c > args.batch_maximum:
				return
			svbuf = []
			lbuf = []
			lens = []
			t = 0
			while t < args.batch_size:
				try:
					sentence, label = next(generator)
					if len(sentence) > 250: continue # 너무 긴 문장 제외
					if all(map(lambda x: x == "O", label)): continue # 최소 1개의 entity가 있는 것만
					c += 1
					t += 1
				except StopIteration:
					if not args.batch_repeat:
						return
					generator = generator_fn(generator_file)
					continue
				svbuf.append(sentence)
				lens.append(len(sentence))
				lbuf.append(label) # padding
			maxlen = max(list(map(len, svbuf)))
			if args.use_word_embedding == "True":
				embedding_func = self.generate_concatenated_embedding_from_sentence
			else:
				embedding_func = self.generate_char_embedding
			x = np.array(list(map(lambda x: embedding_func(x, maxlen), svbuf)))
			# for item in lbuf:
			# 	item += "O" * (maxlen - len(item))
			# 	print(len(item))
			lbuf = list(map(lambda x: x + ["O"] * (maxlen - len(x)), lbuf)) # 0으로 tagging하기 위해
			# one_hot = np.array(list(map("BILOU".index, lbuf)))
			label_num = np.array([list(map(encode_label, item)) for item in lbuf])
			lens = np.array(lens)
			yield x, lens, label_num
	def load_arguments(self, args):
		if os.path.isdir(args.save_path) and os.path.isfile(args.save_path+"args.json"):
			import json
			with open(args.save_path+"args.json") as f:
				j = json.load(f)
				for k, v in j.items():
					setattr(args, k, v)




# if __name__ == '__main__':
# 	model = LSTM_CRFModel(args)
# 	import os
# 	from DataPrepareModule import DataPreparer as dp
# 	from NERUtil import test_metric
# 	encode_func = dp.corpus
# 	with tf.Session() as sess:
# 		saver = model.load_trained_session(sess, args)
# 		it = 0
# 		print()
# 		for it in range(1, args.epoch+1):
# 			# with open("corpus/train.txt", encoding="UTF8") as train:
# 			# 	testset_gen = model.generate_trainset(encode_func, train, args)
# 			# 	b = 0
# 			# 	for sv, seqlen, label in testset_gen:
# 			# 		b += len(seqlen)
# 			# 		seq, loss = model.train(sess, sv, seqlen, label)
# 			# 		print("\rBatch %d: %.2f" %(b, loss), end="", flush=True)
# 			# 	print()
			
# 			if it % SAVE_ITER == 0:
# 				saver.save(sess, args.save_path+"%d.ckpt" % args.char_dim)
# 				print("Epoch %d: Model saved" % it)
# 				efs, sfs = 0, 0
# 				c = 0
# 				with open("corpus/dev.txt", encoding="UTF8") as dev:
# 					for sentence, label in encode_func(dev):
# 						pred, score = model.predict_sent(sess, sentence)
# 						ef, sf = test_metric(sentence, label, pred)
# 						if ef == -1 and sf == -1: continue
# 						efs += ef
# 						sfs += sf
# 						c += 1
# 						if c > 1000: break
# 					if c > 0:
# 						print(efs / c, sfs / c)