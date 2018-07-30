import configparser
import sys
import json
import socket

def recvall(sock):
	result = ""
	while True:
		dat = sock.recv(4096)
		result += dat.decode("UTF8")
		if len(dat) < 4096: break
	return result

if __name__ == '__main__':
	config = configparser.ConfigParser()
	config.read(sys.argv[1])
	input_file = config["preprocess"]["prepro_in_path"]
	output_file = config["preprocess"]["prepro_ner_out_path"]
	sock = socket.socket()
	sock.connect(("143.248.136.2", 6666))
	with open(input_file, encoding="UTF8") as rf, open(output_file, "w", encoding="UTF8") as wf:
		result = []
		for line in rf.readlines():
			if len(line.strip()) == 0:
				continue
			sock.send(line.strip().encode("UTF8"))
			response = recvall(sock)
			result.append(json.loads(response))
		json.dump(result, wf, ensure_ascii=False, indent="\t")
	sock.close()