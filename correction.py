import pickle
from difflib import get_close_matches
import re

def load_dictionary_utf8(path):
  with open(path, 'rb') as handle:
      dictionary = pickle.load(handle)
  return dictionary
def load_data_telex(path):
  with open(path, 'r') as f:
    data = f.read()
  data = data.split('\n')
  return data

def convert_utf2telex(w, dict_utf2telex):
  tmp = []
  w = w.lower()
  for c in w:
    if c in dict_utf2telex:
      tmp.append(utf2telex[c])
    else:
      tmp.append(c)
  return ''.join(tmp)

def convert_telex2utf(w, dict_telex2utf):
  ## Tìm telex với trigram
  tmp = []
  i = 0
  while i < (len(w)- 2):
    if w[i:i+3] in dict_telex2utf:
      tmp.append(dict_telex2utf[w[i:i+3]])
      i = i + 2
    else:
      tmp.append(w[i])
    i += 1
  tmp.append(w[i:])
  w = ''.join(tmp)

  ## Tìm telex với bigram
  tmp = []
  i = 0
  while i < (len(w) - 1):
    if w[i:i+2] in dict_telex2utf:
      tmp.append(dict_telex2utf[w[i:i+2]])
      i = i + 1
    else:
      tmp.append(w[i])
    i += 1
  tmp.append(w[i:])
  return ''.join(tmp)

def map_telex_utf8(data):
  telex2utf = {}
  utf2telex = {}
  for line in data:
    telex, utf8 = line.split('\t')
    telex2utf[telex] = utf8
    utf2telex[utf8] = telex
  return telex2utf, utf2telex

def get_list_telex_utf(dict, dic_utf2telex):
  words_telex = []
  words_utf = []
  for w in dict:
    words_telex.append(convert_utf2telex(w, dic_utf2telex))
    words_utf.append(w)
  return words_telex, words_utf

def dict_teencode(path):
  with open(path, 'r') as f:
    data = f.read()
  teencode = {}
  for line in data.split('\n'):
    a, b = line.split('\t')
    teencode[a] = b
  return teencode

def correction(word): 
    cand = candidates(word)
    return cand
def candidates(word): 
    "Generate possible spelling corrections for word."

    # return [word]
    return known([word]) or known(edits1(word)) or known(edits2(word))
def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in words_telex)
def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)
def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def detect(s):
  list_miss = []
  s = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()]', ' daucau', s)
  s = re.sub(r'[0-9]', ' so', s)
  s = s.lower()
  s = s.split()
  for i, v in enumerate(s):
    if v in teencode:
      list_miss.append(i)
      continue
    if v not in dictionary:
      list_miss.append(i)
  return s, list_miss

def process_words(w):
  letters    = 'abcdefghijklmnopqrstuvwxyz'
  for i in letters:
    s = i + i
    w = re.sub(i+'('+i +')+', s , w)
  return w
def model_correction(res, list_miss):
  result = {}
  for i in list_miss:
    tmp = []
    if res[i] in teencode:
      result[res[i]] = [teencode[res[i]]]
      continue
    p = process_words(res[i])
    w = convert_utf2telex(p, utf2telex)
    for txt in correction(w):
      tmp.append(convert_telex2utf(txt, telex2utf))
    
    if res[i] not in result:
      result[res[i]] = tmp

  return result
def model_dictionary(res, list_miss):
  result = {}
  for i in list_miss:
    tmp = []
    p = process_words(res[i])
    w = convert_utf2telex(p, utf2telex)
    res_w = get_close_matches(w, words_telex, 20, cutoff=0.5)
    for a in res_w:
      tmp.append(convert_telex2utf(a, telex2utf))
    
    if res[i] not in result:
      result[res[i]] = tmp

  return result
def model(s):
  res, list_miss = detect(s)
  print(res)
  print(list_miss)
  result = {}
  result_correct = model_correction(res, list_miss)
  result_dictionary = model_dictionary(res, list_miss)

  for i in list_miss:
    if res[i] not in result:
      result[res[i]] = []
    else:
      continue
    if len(result_correct[res[i]]) < 10:
      result[res[i]].extend(result_correct[res[i]])
    else:
      result[res[i]].extend(result_correct[res[i]][:10])
    count = len(result[res[i]])
    for w in result_dictionary[res[i]]:
      if w not in result[res[i]]:
        result[res[i]].append(w)
        count += 1
      if count >= 15:
        break
  check_doc = []
  for s in res:
    if s in result:
      check_doc.append([s, '1', result[s]])
    else:
      check_doc.append([s, '0', []])
  return check_doc

def words(text): return re.findall(r'\w+', text.lower())

teencode_path = 'teencode.txt'
teencode = dict_teencode(teencode_path)

from collections import Counter

WORDS = Counter(words(open('all_sentences.txt').read()))

dictionary_path = 'dict.pickle'
dictionary = load_dictionary_utf8(dictionary_path)
dictionary['daucau'] = 1
dictionary['numbers'] = 1 

# Append dictionary
for k, v in WORDS.most_common():
  if v > 100:
    if (len(k) == 1) or (k[0] in '0123456789'):
      continue
    if k not in dictionary:
      dictionary[k] = v

telex_path = 'telex.txt'
data_telex = load_data_telex(telex_path)

telex2utf, utf2telex = map_telex_utf8(data_telex)
words_telex, words_utf = get_list_telex_utf(dictionary, utf2telex)

# len(dictionary)

# s1 = 'Các nhà đầu tư Trung Quốc được nghà ngước hậu thuẫn đả thể hiện sự quan tâm đến nhữn cơ sở hạ tần cảng ở ba quốc gia mà ông Dương dự kiến đến thăm'
# s2 = 'VnExpress hay Tin nhanh Việt Nam là một chang báo điện tử tại Việt Nam đượcc thành lập bởi tập đoàn FPT, ra mắt dào ngày 26 tháng 02 năm 2001 và được Bộ Văn hóa - Thông tin cấp giấy phép'
# s = 'Ảnh: Singapore sáng nay cũng thông báo chất lượng không chí ngui hại, chong bối cảnh quốc đảo này đan chuẩn bỵ cho giải đua mô tô Công thức 1 vào 22/9'

# s = 'Ngay cả trong cuộc xống vợ chồng trúng tôi cũng tôn trọng cá tính của nghau'
# s = 'Giá cà chua tại chợu dân sinh, nông phamm ở hafaa noiii'
# s = 'Bám riết nhau xát nút, Hoa hậu Bà Biển Ngà, Yéo Jennifer, gyành chiến thắng ở thứ thátr đầu tiên, theo xau đó là đại diện Phần Lan'

# s = "Phairi lamf sao khiiii tooi ddang ddi tren dduowngf"
# s = "Tooi yeeu emmmm raatttt nhiêufff"
# s = 'Tôi yeu emmmmm bằng hết tấm thân mk'
# s = 'xe ddapj lachs cachs tooi vaanx chuwa quen dduowng thif tooi choi vowi conf toi vann cuws dduwngs ddowij'
# s = 'Tooiii yeeuuu êm nhiều lamwsmm'

# result = model(s)
# print("Correction")
# for k in result:
#   print(k, result[k])

