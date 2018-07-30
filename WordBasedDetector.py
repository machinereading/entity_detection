import tensorflow as tf
import KoreanUtil
import numpy as np
from NERUtil import encode_label
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score
from konlpy.tag import Kkma, Komoran, Hannanum
import argparse
WORD_EMBEDDING_SIZE = 300

SAVE_ITER = 10
np.random.seed(10)

BILOU_TRANSITION_MATRIX = np.genfromtxt("crf_prob.csv", delimiter=",")
WV_PREFIX = "vector/word2vec_%s.bin"

class WordBasedLSTMCRF():
	def __init__(self, args):
		# specify items with arguments
		self.wv = KeyedVectors.load_word2vec_format(WV_PREFIX % (args.morph_type), binary=True)
		
		morphemizers = {"kk": Kkma, "ko": Komoran, "h": Hannanum}
		self.morphemizer = morphemizers[args.morph_type]()
		with tf.name_scope("WORD_LSTM_CRF"):
			label_size = 17
			
			self.sentence_embedding = tf.placeholder(tf.float32, [None, None, WORD_EMBEDDING_SIZE])
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

	def predict_sent(self, sess, sentence):
		try:
			s = self.morphemizer.morphs(sentence)
		except Exception:
			return [], [] # 못찾은거지 뭐

		seqlen = [len(s)]
		sv = np.array([self.generate_embedding(s)])
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
		saver = tf.train.Saver(tf.global_variables())
		try:
			
			save_path = "saves/word_based/%s_%s" % (args.morph_type, args.cell_type) if args.save_path is None else args.save_path
			
			saver.restore(sess, save_path+"%s.ckpt" % args.morph_type)
			print("saved model restored")
		except Exception:
			sess.run(tf.global_variables_initializer())
		return saver

	# sentence: list of morphs by self.morphemizers
	# maxlen: maximum morph length of batch
	def generate_embedding(self, sentence, maxlen=None):
		sv = []
		for word in sentence:
			sv.append(np.zeros([WORD_EMBEDDING_SIZE,]) if word not in self.wv else self.wv[word])
		if maxlen is not None:
			while len(sv) < maxlen:
				sv.append(np.zeros([WORD_EMBEDDING_SIZE,]))
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
			x = np.array(list(map(lambda x: self.generate_embedding(x, maxlen), svbuf)))
			# for item in lbuf:
			# 	item += "O" * (maxlen - len(item))
			# 	print(len(item))
			lbuf = list(map(lambda x: x + ["O"] * (maxlen - len(x)), lbuf)) # 0으로 tagging하기 위해
			# one_hot = np.array(list(map("BILOU".index, lbuf)))
			label_num = np.array([list(map(encode_label, item)) for item in lbuf])
			lens = np.array(lens)
			yield x, lens, label_num
