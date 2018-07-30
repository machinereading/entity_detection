import NER
import argparse
import sys
import tensorflow as tf
import json
import socket
import threading
module_lock = False
def open_socket(module):
	with tf.Session() as sess:
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.bind((socket.gethostname(), 6666))
		s.listen(1)
		print("server ready")
		while True:
			c, _ = s.accept()
			sock_func(c, sess, module)
			# t = threading.Thread(target=sock_func, args=[c, sess, module])
			# t.start()

def sock_func(sock, sess, module):
	try:
		while True:
			data = sock.recv(4096).decode("utf-8")
			print(data, len(data))
			result = json.dumps(module.predict(data, sess), ensure_ascii=False).encode("utf-8")
			sock.send(result)
	except Exception:
		print("EXCEPTION")
		sock.close()

if __name__ == '__main__':
	args = argparse.Namespace()
	args.save_path = "saves/wikipedia_150/"
	args.load_from_file = True
	ner_module = NER.NER(args)
	open_socket(ner_module)