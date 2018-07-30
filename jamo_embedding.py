import KoreanUtil

with open("corpus/wikiraw/entityTypeTaggedText_add3.txt", encoding="UTF8") as rf, open("corpus/wikiraw/jamo_decomposed.txt", "w", encoding="UTF8") as wf:
	for line in rf.readlines():
		for char in line:
			for jamo in KoreanUtil.char_to_elem(char, False, True):
				wf.write(jamo)