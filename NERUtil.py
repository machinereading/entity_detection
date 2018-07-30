#################################################
# Scoring functions and label-related functions #
#################################################

from DataPrepareModule import DataPreparer as dp
import numpy as np

def label_no(label):
	bilu = "BILU"
	plom = "PLOM"
	if label == "O": return 0
	x = label.split("/")
	return bilu.index(x[0])+plom.index(x[1])*4+1

def decode_label(sentence, label, score=None):
	entities = []
	temp = ""
	pm = 0
	if score is None:
		score = [None] * len(label)
	if all(map(lambda x: type(x) is np.int32, list(label))):
		t = []
		for item in label:
			if item == 0:
				t.append("O")
				continue
			plom = "PLOM"
			bilu = "BILU"
			item -= 1
			t.append(bilu[item % 4]+"/"+plom[item // 4])
		label = t
	ind = 0
	sin = 0
	sc = 0
	for c, l, s in zip(sentence, label, score):
		if l[0] == "U":
			entities.append([c, l[-1], ind, ind+1, s[label_no(l)] if s is not None else 0])
			ind += 1
			continue
		if l[0] == "B":
			pm += 1
			sin = ind
			temp += c
			if s is not None:
				sc += s[label_no(l)]
		if pm == 1 and l[0] == "I":
			temp += c
			if s is not None:
				sc += s[label_no(l)]
		if pm == 1 and l[0] == "L":
			temp += c
			if s is not None:
				sc += s[label_no(l)]
			entities.append([temp, l[-1], sin, ind+1, sc / (ind-sin+1)])
			temp = ""
			pm = 0
		ind += 1
	return entities

def encode_label(tag):
	# BILU*4 + O
	try:
		bilu = "BILU".index(tag[0])
		seq = "PLOM"
		ty = tag[-1]
		return seq.index(ty)*4+bilu+1
	except ValueError:
		return 0	

def is_correct(a, l):
	al = list(map("BILOU".index, a))
	return al == l

def test_metric(s, a, l):
	tp_exact = tp_sub = 0
	fp_exact = fp_sub = 0
	fn_exact = fn_sub = 0
	
	
	a_word = list(map(lambda x: x[:-2], decode_label(s, a)))
	l_word = list(map(lambda x: x[:-2], decode_label(s, l)))
	correct_word = []
	sub_correct_word = []
	if len(a_word) == 0:
		return -1, -1
	elif len(a_word) > 0 and len(l_word) == 0:
		return [0,0], [0,0]
	
	for item in l_word:
		if item in a_word:
			tp_exact += 1
			correct_word.append(item)
		else:
			fp_exact += 1
			flag = False
			for aw in a_word:
				if (item[0] in aw[0] or aw[0] in item[0]) and item[1] == aw[1]:
					tp_sub += 1
					flag = True
					sub_correct_word.append(aw)
					break
			if not flag:
				fp_sub += 1
	for item in a_word:
		if item in correct_word: continue
		if item not in l_word:
			fn_exact += 1
			if item not in sub_correct_word:
				fn_sub += 1
	exact_precision = tp_exact / (tp_exact+fp_exact)
	exact_recall = tp_exact / (tp_exact+fn_exact)
	ef = 2*exact_precision*exact_recall/(exact_precision+exact_recall) if (exact_precision+exact_recall) > 0 else 0
	tp_sub += tp_exact
	sub_precision = tp_sub / (tp_sub+fp_sub)
	sub_recall = tp_sub / (tp_sub+fn_sub)
	sf = 2*sub_precision*sub_recall/(sub_precision+sub_recall) if (sub_precision+sub_recall) > 0 else 0
	typed = (ef, sf)

	# without type matching
	tp_exact = tp_sub = 0
	fp_exact = fp_sub = 0
	fn_exact = fn_sub = 0
	correct_word = []
	sub_correct_word = []
	l_word = list(map(lambda x: x[0], l_word))
	a_word = list(map(lambda x: x[0], a_word))
	for item in l_word:
		if item in a_word:
			tp_exact += 1
			correct_word.append(item)
		else:
			fp_exact += 1
			flag = False
			for aw in a_word:
				if (item in aw or aw in item) :
					tp_sub += 1
					flag = True
					sub_correct_word.append(aw)
					break
			if not flag:
				fp_sub += 1
	for item in a_word:
		if item in correct_word: continue
		if item not in l_word:
			fn_exact += 1
			if item not in sub_correct_word:
				fn_sub += 1
	exact_precision = tp_exact / (tp_exact+fp_exact)
	exact_recall = tp_exact / (tp_exact+fn_exact)
	ef = 2*exact_precision*exact_recall/(exact_precision+exact_recall) if (exact_precision+exact_recall) > 0 else 0
	tp_sub += tp_exact
	sub_precision = tp_sub / (tp_sub+fp_sub)
	sub_recall = tp_sub / (tp_sub+fn_sub)
	sf = 2*sub_precision*sub_recall/(sub_precision+sub_recall) if (sub_precision+sub_recall) > 0 else 0
	untyped = (ef, sf)

	return typed, untyped

def mark(sent, entities):
	s = "start_index"
	e = "end_index"
	ent = sorted(entities, key=lambda x: x[s])
	for item in ent:
		sent = sent[:item[s]]+"["+sent[item[s]:]
		for item1 in ent:
			item1[s] += 1
			item1[e] += 1
		sent = sent[:item[e]]+("/%s]" % item["type"])+sent[item[e]:]
		for item1 in ent:
			item1[s] += 3
			item1[e] += 3
	return sent