import re
import time
fname = "corpus/entityTypeTaggedText_add.txt"

def geti(sentence):
	eb = []
	si = 0
	while True:
		try:
			si = sentence.index("[[[", si)
			ei = sentence.index(">>", si)
		except Exception:
			break
		eb.append((si, ei))
		si = ei
	return eb
sc = 0
with open(fname, encoding="UTF8") as f, open(fname.split(".")[0]+"3.txt", "w", encoding="UTF8") as wf:
	entities = set([])
	sentences = []
	nec = 0
	for line in f.readlines():
		si = 0
		sentences.append(line.strip())
		while True:			
			try:
				si = line.index("[[[", si)
				ei = line.index("]]]", si)
			except Exception:
				break
			en = line[si+3:ei].replace("_", " ")
			si = line.index("<<", si)
			ei = line.index(">>", si)
			tag = line[si+2:ei].split("|")[1]
			entities.add((en, tag))
		
		if line.strip() == "<&doc&>": # delimiter
			sc += 1
			
			entities = sorted(list(entities), key=lambda x: -len(x[0]))
			for sentence in sentences[:-1]:
				sentence = sentence.strip()
				eb = geti(sentence)
				ac = 0
				skip = False
				for entity, tag in entities:
					added = True
					not_entity_index = []
					while added:
						entity_index = []
						nsi = 0
						added = False
						while True:
							try:
								nsi = sentence.index(entity, nsi)
								entity_index.append(nsi)
								nsi += len(entity)
							except Exception:
								break
						for nsi in entity_index:
							new_entity = True
							for si, ei in eb:
								if si <= nsi < ei:
									new_entity = False
									break
							if new_entity:
								# print(sentence[:nsi])
								# print(entity)
								# print(sentence[nsi+len(entity):])
								# print()
								# time.sleep(1)
								if nsi != 0:
									lw = sentence[nsi-1]
									if 0xAC00 <= ord(lw) <= 0xD7A3:
										continue
								sentence = sentence[:nsi]+"[[["+entity+"]]]<<"+tag+">> "+sentence[nsi+len(entity):]
								eb = geti(sentence)
								added = True
								nec += 1
								ac += 1
								break
						if ac > 100: # may be errorous sentence. cut it out!
							# print(sentence)
							ac = 0
							nec -= ac
							added = False
							skip = True
				if not skip:
					wf.write(sentence+"\n")
				print("\r%d --> %d" % (sc, nec), end="", flush=True)
			wf.write("\n")
			sentences = []
			entities = set([])
			
		
		

# print(sc)
# entities = sorted(list(entities), key=lambda x: -len(x[0]))
# print(len(entities))
# nec = 0
# lc = 0
# with open(fname, encoding="UTF8") as rf, open(fname.split(".")[0]+"2.txt", "w", encoding="UTF8") as wf:
# 	for line in rf.readlines():
# 		# print(line.strip())
# 		si = 0
# 		eb = []
# 		while True:
# 			try:
# 				si = line.index("[[[", si)
# 				ei = line.index("]]]", si)
# 			except Exception:
# 				break
# 			eb.append((si+3, ei))
# 		for entity, tag in entities:
# 			try:
# 				nsi = line.index(entity)
# 			except Exception:
# 				continue
# 			new_entity = True
# 			for si, ei in eb:
# 				if si <= nsi < ei:
# 					new_entity = False
# 					break
# 			if new_entity:
# 				print(entity, end=", ")
# 				line = line[:si]+"[[["+entity+"]]]<<"+tag+">> "+line[si+len(entity):]
# 				eb.append(si, si+len(entity))
# 				nec += 1
# 		wf.write(line)
# 		lc += 1
# 		print("\r%d/%d" %(lc, sc), end="", flush=True)
# 		# print()
# print(nec)