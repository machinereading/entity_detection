import re
class RuleBasedNERModel():
	def __init__(self):
		self.josa_list = ["은", "는", "이", "가", "을", "를", "에서", "에서는", "께서", "의", "께", "에서부터", "보다", "으로서", "으로", "로", "으로써", "고", "라고", "와", "과", "랑", "이랑", "에", "같이", "처럼", "부터", "로부터", "으로부터", "도", "까지", "마저", "조차", "대로", "뿐", "만", "뿐만"].sort(key=lambda x: -len(x))
	def predict_sent(self, sentence):
		# predict time label
		
		time_pattern = r'(\d+(년|월|일} ?)+|(\d+시 \d+분)'
		bracket_pattern = r'<.*>|〈,*〉|《.*》'
		results = []
		start_index = 0
		for item in sentence.split(" "):
			for josa in self.josa_list:
				if item.endswith(josa):
					results.append((item[:-len(josa)], start_index, start_index+len(item)-len(josa), "M"))
		
		
		
		for pattern in [time_pattern, bracket_pattern]:
			for item in self.extract_pattern_in_sentence(sentence, pattern):
				results.append(item)
		
		return results
	
	def extract_pattern_in_sentence(self, sentence, pattern):
		return [(item[0], item.start, item.end, "M") for item in re.finditer(pattern, sentence)]
			