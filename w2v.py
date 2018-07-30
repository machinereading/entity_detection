from gensim.models import Word2Vec
# from konlpy.tag import Twitter, Hannanum, Kkma, Komoran
from DataPrepareModule import DataPreparer as dp
class MakeIter(object):
    def __init__(self, generator_func, **kwargs):
        self.generator_func = generator_func
        self.kwargs = kwargs
    def __iter__(self):
        return self.generator_func(**self.kwargs)

def stream(fn, mod):
	it = 0
	with open(fn, encoding="UTF8") as f:
		for line in f.readlines():
			if len(line.strip()) == 0: continue
			yield mod.morphs(line.strip())
			print("\r%d" % it, end="", flush=True)
			it += 1
it = 0
a = ["kk", "ko"]
# for item in [Kkma(), Komoran(), Hannanum()]:
	# with open("raw_%s" % a[it], "w", encoding="UTF8") as wf:
	# 	for morphs in stream("raw", item):
	# 		wf.write(" ".join(morphs)+"\n")

def gen(f):
	for item, _ in dp.corpus(f):
		yield item

with open("corpus/wikiraw/jamo_decomposed.txt", encoding="UTF8") as f:
	model = Word2Vec(MakeIter(gen, f=f), size=300, window=5)
	model.wv.save_word2vec_format("vector/word2vec_jamo_decoposed.bin", binary=True)
