import re
from konlpy.tag import Twitter, Hannanum, Kkma, Komoran
import traceback
class DataPreparer():
	ko = Komoran()
	h = Hannanum()
	kk = Kkma()
	p_dict = {"ko": ko, "h": h, "kk": kk}
	
	@classmethod
	def korean_ner(cls, hclt_format_file, include_tag=True):
		sentences = []
		tagdict = {"LC": "L", "PS": "P", "OG": "O", "DT": "M", "TI": "M"}
		for line in hclt_format_file.readlines():
			if line.startswith(";"):
				sentences.append(line.strip()[2:])
			if line.startswith("$"):
				temp = []
				labeltemp = ""
				pm = 0
				for char in line.strip()[1:]:
					# print(char, end="")
					if char == '<' and pm == 0:
						pm = 1
						continue
					if char == ">" and pm == 1:
						pm = 0
						x = labeltemp.split(":")
						labeltemp = ":".join(x[:-1])
						ty = tagdict[x[-1]]
						if len(labeltemp) == 1:
							temp.append("U/%s" % ty)
							labeltemp = ""
							continue
						for item in (("B/%s " % ty) +(("I/%s " % ty)*(len(labeltemp)-2))+("L/%s" % ty)).split(" "):
							temp.append(item)
						labeltemp = ""
						continue
					
					if pm == 0:
						temp.append("O")
					elif pm == 1:
						labeltemp += char
				yield sentences[-1], temp
				sentences = []

	@classmethod
	def corpus(cls, corpus_file, include_tag=True):
		for line in corpus_file.readlines():
			ind = []
			# while True:
			# 	try:
			# 		si = line.index("<<")
			# 		ei = line.index(">>", si)
			# 	except Exception:
			# 		break
			# 	line = line[:si]+line[ei+3:] # >> 뒤에는 항상 공백이 하나 있더라?
			if len(line.strip()) == 0: continue
			skip = False
			while True:
				try:
					si = line.index("[[[")
					ei = line.index(">>", si)
				except Exception:
					break
				
				txt = re.sub(r"\(.*\)", "", line[si+3:ei].replace("_", " "))
				try:
					en, ty = txt.split("]]]<<")
					en = en.split("|")[-1]
					ty = ty.split("|")[-1]
				except Exception:
					skip = True
					break
				if en.endswith(" "): en = en[:-1]
				ind.append((si, len(en), ty))
				line = line[:si]+en+line[ei+3:]
			label = []
			if skip: continue
			for s, l, t in ind:
				for _ in range(s-len(label)):
					label.append("O")
				if l == 1:
					label.append("U/%s" % t[0])
				else:
					for c in (("B/%s " % t[0])+(("I/%s " % t[0]) * (l-2))+("L/%s" % t[0])).split(" "):
						label.append(c)
			l = line.strip()
			for _ in range(len(l) - len(label)):
				label.append("O")
			yield l, label

	@classmethod
	def corpus_morph(cls, corpus_file, args=None):
		if args is None:
			parser = cls.kk
		else:
			parser = cls.p_dict[args.morph_type]
		for line in corpus_file.readlines():
			try:
				line = line.strip()
				# while True:
				# 	try:
				# 		si = line.index("<<")
				# 		ei = line.index(">>", si)
				# 	except Exception:
				# 		break
				# 	line = line[:si]+line[ei+3:] # >> 뒤에는 항상 공백이 하나 있더라?
				
				# normalize entity text
				l = 0
				while True:
					try:
						si = line.index("[[[", l)
						ei = line.index("]]]", si)
						l = ei
					except Exception:
						break
					txt = re.sub(r"\(.*\)", "", line[si+3:ei].replace("_", " ")).split("|")[-1]
					if txt.endswith(" "): txt = txt[:-1]
					line = line[:si+3]+txt+line[ei:]
				try:
					x = parser.morphs(line)
				except Exception:
					continue
				sent = []
				label = []
				pm = 0
				st = []
				labeltype = ""
			
				ind = 0
				
				print(x)
				while ind <= len(x) - 3:
					a,b,c = (x[ind+i] for i in range(3))
					if a==b==c=="[":
						del x[ind]
						del x[ind+1]
						x[ind] = "[[["
						continue
					if a==b==c=="]":
						del x[ind]
						del x[ind+1]
						x[ind] = "]]]"
						continue
					if b==c=="<":
						del x[ind+2]
						x[ind+1] = "<<"
						continue
					if b==c==">":
						del x[ind+2]
						x[ind+1] = ">>"
						continue
					ind += 1
				for word in x:
					if len(word) < 1: continue
					if "[[[" in word:
						s = word.index("[[[")
						if s != 0:
							sent.append(word[:s])
							label.append("O")
						pm += 1
						continue
					if pm == 1 and "|" in word:
						st = []
						continue
					if "]]]" in word or "<<" in word:
						pm = 2
						continue
					if ">>" in word:
						# add label and type
						for item in st:
							sent.append(item)
						if len(st) == 1:
							label.append("U/%s" % labeltype)
						else:
							for item in (("B/%s " % labeltype)+(("I/%s " % labeltype) * (len(st)-2)) + ("L/%s" % labeltype)).split(" "):
								label.append(item)
						st = []
						labeltype = ""
						pm = 0
						continue
					if pm == 2 and "|" in word:
						labeltype = ""
						continue
					if pm == 0:
						sent.append(word)
						label.append("O")
					if pm == 1:
						st.append(word)
					if pm == 2:
						labeltype = word[0]

				# for word in x:
				# 	if len(word) < 1: continue
				# 	if "[[[" in word:
				# 		s = word.index("[[[")
				# 		if s != 0:
				# 			sent.append(word[:s])
				# 			label += "O"
				# 		pm += 1
				# 		continue
				# 	if "]]]" in word:
				# 		pm = 0
				# 		if label[-1] == "B":
				# 			label = label[:-1]+"U"
				# 		else: label = label[:-1]+"L"
				# 		continue
				# 	sent.append(word)
				# 	if pm == 0:
				# 		label += "O"
				# 	else:
				# 		label += "B" if len(label) == 0 or label[-1] in "OLU" else "I"

				yield sent, label
			except Exception : 
				traceback.print_exc(2)
				print(line.strip())
	@classmethod
	def premade(cls, corpus_file):
		i = 0
		sent = []
		label = []
		for line in corpus_file.readlines():
			if len(line.strip()) == 0: continue
			if i % 2 == 0:
				sent = line.strip().split("/sep/")
			else:
				label = line.strip().split("/sep/")
				if len(sent) == 0 or len(label) == 0: continue
				yield sent, label
			i += 1
			
	@classmethod
	def etri_golden(cls, corpus_file):
		label_func = lambda x: x[0] if x[0] not in ["T", "D"] else "M"
		for sentence in corpus_file.readlines():
			s = sentence.strip()
			pm = 0
			senttemp = ""
			labeltemp = ""
			labels = []
			for c in s:
				if c == "<":
					pm = 1
					continue
				if pm == 1 and c == ">":
					pm = 0
					ll = labeltemp.split(":")
					surface = ":".join(ll[:-1])
					ty = ll[-1]
					senttemp += surface
					labeltype = label_func(ty)
					if len(surface) == 1:
						labels.append("U/%s" % labeltype)
					else:
						for item in (("B/%s " % labeltype)+(("I/%s " % labeltype) * (len(surface)-2)) + ("L/%s" % labeltype)).split(" "):
							labels.append(item)
					labeltemp = ""
					continue
				if pm == 1:
					labeltemp += c
				else:
					senttemp += c
					labels.append("O")
			yield senttemp, labels
	@classmethod
	def wikipedia_golden(cls, corpus_file):
		for sentence in corpus_file.readlines():
			if len(sentence.strip()) == 0: continue
			pm = 0
			sent = ""
			lt = ""
			labels = []
			for c in sentence.strip():
				if c == "[":
					pm += 1
					continue
				if c == "]":
					pm = 0
					try:
						surface, ty = lt.split(";")
					except Exception:
						print(sentence)
						import sys
						sys.exit(1)
					if ty == "T": ty = "M"
					if len(surface) == 1:
						labels.append("U/%s" % ty)
					else:
						for item in (("B/%s " % ty)+(("I/%s " % ty) * (len(surface)-2)) + ("L/%s" % ty)).split(" "):
							labels.append(item)
					lt = ""
					sent += surface
					continue
				if pm == 0:
					sent += c
					labels.append("O")
				else:
					lt += c
			yield sent, labels

if __name__ == '__main__':
	with open("corpus/koreanner/original/dev.txt", encoding="UTF8") as f:
		i=0
		for s, l in DataPreparer.korean_ner(f):
			i+=1
		print(i)