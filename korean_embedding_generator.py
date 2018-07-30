import numpy as np
from KoreanUtil import cho, jung, jong
np.random.seed(10)

# 한글 자모 (종성 분리/미분리) + 영어 알파벳 + 공백 + 기타 문자, 미존재시 0
o_with_separated_jongsung = np.random.rand(len(cho)+len(jung)+len(jong) + 26 + 2, 30)
o_with_merged_jongsung = np.random.rand(len(jung)+len(jong) + 26 + 2, 30)

# np.save("models/korean_char_embedding_separated_jongsung", o_with_separated_jongsung)
# np.save("models/korean_char_embedding_merged_jongsung", o_with_merged_jongsung)
np.save("vector/dummy_word_vector", np.random.rand(1, 300))