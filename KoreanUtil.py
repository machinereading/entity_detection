# NLG module의 static 함수들이라고 생각해
import numpy as np
# import konlpy
# redirection = {"은": eun, "을": eul, "이": ee, "과": gwa, "어미_이": eomi_ee}
# josa = {}
# tagger = konlpy.tag.Twitter()
# with open("vocab/josa.json", encoding="UTF8") as f:
#     for line in f.readlines():
#         s = line.strip().split(":")
#         josa[s[0]] = s[1].split(",")
np.random.seed(10)
def build_jamo_data(dim=10):
	np.save("models/pieced_jamo_vector%d" % dim, np.random.rand(len(cho)+len(jung)+len(jong), dim))

def build_punc_data(dim=30):
	np.save("models/ascii_vector%d" % dim, np.random.rand(128, dim))

cho = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
jung = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
jong = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
decomposer = {"ㄳ": ["ㄱ","ㅅ"], "ㄵ": ["ㄴ","ㅈ"], "ㄶ": ["ㄴ", "ㅎ"], "ㄺ": ["ㄹ", "ㄱ"], "ㄻ": ["ㄹ", "ㅁ"], "ㄼ": ["ㄹ", "ㅂ"], "ㅀ": ["ㄹ","ㅎ"], "ㅄ": ["ㅂ","ㅅ"]}
jamo_len = len(cho) + len(jung) + len(jong)
class Embedding():
	def __init__(self):
		self.j = None
		self.a = None
		self.dim = 0
	
	def load_vector(self, dim):
		if self.dim == dim: return
		self.dim = dim
		try:
			self.j = np.load("models/pieced_jamo_vector%d.npy" % (dim // 3))
			self.a = np.load("models/ascii_vector%d.npy" % dim)
		except IOError:
			build_jamo_data(dim // 3)
			build_punc_data(dim)
			self.j = np.load("models/pieced_jamo_vector%d.npy" % (dim // 3))
			self.a = np.load("models/ascii_vector%d.npy" % dim)
	
	def build_jamo_data(self, dim=10):
		np.save("models/pieced_jamo_vector%d" % dim, np.random.rand(len(cho)+len(jung)+len(jong), dim))

	def build_punc_data(self, dim=30):
		np.save("models/ascii_vector%d" % dim, np.random.rand(128, dim))

def has_jongsung(character):
	x = ord(character)
	if(x < 0xAC00 or x > 0xD7A3): return False
	return (x - 0xAC00) % 28 != 0


def char_to_elem(character, to_num=False, decompose=False):
	x = ord(character)
	if(x < 0xAC00 or x > 0xD7A3): return character
	x -= 0xAC00
	result = []
	result.append(x%len(jong) if to_num else jong[x % len(jong)])
	x //= len(jong)
	result.append(x%len(jung) if to_num else jung[x % len(jung)])
	x //= len(jung)
	result.append(x%len(cho) if to_num else cho[x % len(cho)])
	result.reverse()
	if decompose:
		if result[-1] in decomposer:
			result = result[:-1]+decomposer[result[-1]]
	return result

def elem_to_char(elems):
	if all(map(lambda x: type(x) is int, elems)) or all(map(lambda x: type(x) is str, elems)):
		x = elems[:]
		c = [cho, jung, jong]
		if type(x[0]) is str:
			try:
				
				for i in range(len(x)):
					x[i] = c[i].index(x[i])
			except IndexError:
				print("Index error")
				return None
		if len(x) < 3:
			x.append(0)
		print(x)
		mod = 0
		for i in range(2):
			mod += x[i]
			mod *= len(c[i+1])
		mod += x[2]
		return chr(mod + 0xAC00)
	return None

e = Embedding()
def char_embedding(char, dim):
	while dim % 3 != 0: dim += 1
	e.load_vector(dim)
	index = char_to_elem(char, True)
	if index is None: # None-korean character
		if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
			return e.a[ord(char), :]
		elif ord(char) < 128: # punctuation
			return e.a[0, :]
		else:
			return np.zeros([dim,])
	else: # 초성, 중성, 종성에 서로 다른 vector 부여
		index[1] += len(cho)
		index[2] += len(cho) + len(jung)
	x = [e.j[index[x], :] for x in range(3)]
	return np.concatenate(x, axis=0)

