import re
from collections import Counter

import os
import json
import logging
import re

import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn 

import sentencepiece as spm
from seqeval.metrics import precision_score as seq_precision, recall_score as seq_recall, f1_score as seq_f1
from transformers import AutoTokenizer, XLMRobertaModel, XLMRobertaForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup


from tqdm.notebook import tqdm
from easydict import EasyDict
import gc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pickle

from difflib import get_close_matches

logger = logging.getLogger(__name__)

# import zipfile
# with zipfile.ZipFile('/content/drive-download-20211030T133316Z-001.zip') as zf:
#     zf.extractall()

telex_char = {}
with open("vocab/telex.txt", 'r', encoding='utf-8') as f:
  for line in f.readlines():
    key, value = line.strip().split('\t')
    # print(line.strip().split(' '))
    telex_char[key] = value

from string import punctuation

def num_parameters(parameters):
    num = 0
    for i in parameters:
        num += len(i)
    return num

class Detector(nn.Module):
    def __init__(self, input_dim, output_dim,  embedding_dim, num_layers, hidden_size):

        super(Detector, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim  = embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings = self.input_dim, embedding_dim = self.embedding_dim, )
        self.LSTM = nn.LSTM(input_size = self.embedding_dim, hidden_size= self.hidden_size, num_layers = self.num_layers, 
                            batch_first = True, dropout = 0.1, bidirectional = True)
        self.linear = nn.Linear(self.hidden_size*2, self.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)
        outputs, (h_n, h_c) = self.LSTM(emb)
        logits = self.linear(outputs)

        p = self.sigmoid(logits)
        return p


class HardMasked(nn.Module):
    def __init__(self, detector, MaskedLM, detector_tokenizer, maskedlm_tokenzier,device ):
        super(HardMasked, self).__init__()

        self.detector = detector.to(device)
        self.MaskedLM = MaskedLM.to(device)
        self.detector_tokenizer = detector_tokenizer
        self.maskedlm_tokenizer = maskedlm_tokenzier
        self.use_device = device

    def forward(self, text):
        prepro_text = ''.join([c for c in text if c not in punctuation])

        maskedlm_features, mask_before_tok, mask_after_tok = self.prepare_input(prepro_text)
        outputs = self.MaskedLM(input_ids = torch.tensor([maskedlm_features['input_ids']], dtype = torch.long, device = self.use_device), 
                            attention_mask = torch.tensor([maskedlm_features['attention_mask']], dtype = torch.long, device = self.use_device) )
        logits = outputs['logits'][0]

        output_ids = torch.argsort(logits, dim= -1, descending=True)
        
        wrong_words = prepro_text.split()
        dic_out = {}
        dic_out_best = {}

        for wrong_idx, pred_idx in zip(mask_before_tok, mask_after_tok):
          poten_word = []
          for i in range(30):
              poten_word.append(self.maskedlm_tokenizer.decode(output_ids[pred_idx, i]))
        #   print(poten_word)
          poten_words = ' '.join(poten_word)
          poten_word = poten_words
          wrong_word = wrong_words[wrong_idx]
          
          poten_telex_word = poten_word
          wrong_telex_word = wrong_word
          for  telex_c, vietnam_c in telex_char.items():
            poten_telex_word = poten_telex_word.replace(vietnam_c, telex_c)
            wrong_telex_word = wrong_telex_word.replace(vietnam_c, telex_c)
          
          dict_telex = list(set(poten_telex_word.split()))
          dict_telex2utf = dict(zip(poten_telex_word.split(), poten_words.split()))
          
          result = get_close_matches(wrong_telex_word, dict_telex, 5, cutoff=0.1)
          result = [dict_telex2utf[w] for w in result if w[0] != '<']
          
          dic_out[wrong_idx] = result
        return dic_out
        
    def prepare_input(self, prepro_text):
        # print("Preprocessing text: ", prepro_text)
        detector_input_ids = self.detector_tokenizer.encode(prepro_text, out_type = int)
        # print("detector_input_ids: ", detector_input_ids)
        detector_input_pieces = self.detector_tokenizer.id_to_piece(detector_input_ids)
        # print("detector_input_pieces: ", detector_input_pieces)
        detector_outputs = (self.detector(torch.tensor([detector_input_ids], dtype = torch.long, device = self.use_device))[0].reshape(1,-1) > 0.5).int()[0] 
        # print("detector_outputs: ",detector_outputs)

        for i in range(1, len(detector_input_pieces)):
            if detector_outputs[i] == 1:
                if detector_input_pieces[i][0] == '▁':
                  detector_input_pieces[i] = ' <mask>'
                else:
                  detector_input_pieces[i] = '<del-mask>'
        # print(detector_input_pieces)
        masked_s = self.detector_tokenizer.decode(detector_input_pieces)
        # print(masked_s)
        
        masked_s = re.sub(r'<del-mask>', '', masked_s)      
        # print("masked_s: ", masked_s)
        
        mask_before_tok = []
        msk = masked_s.split()
        for i in range(len(msk)):
          if msk[i] == '<mask>':
            mask_before_tok.append(i)

        
        maskedlm_features = self.maskedlm_tokenizer(masked_s)
        # print("maskedlm_features: ", maskedlm_features)

        mask_after_tok =[]
        input_ids = maskedlm_features["input_ids"]
        mask_after_tok = np.where(np.array(input_ids)==250001)[0]
        # print('mask_after_tok:', mask_after_tok)
        return maskedlm_features, mask_before_tok, mask_after_tok


def load_model():
    print('Load model')
    detector_path = 'model/Detector.pkl'

    # Load detector and XLM-R masked language model to create Hard-Masked XLM-R
    # Change the directories to Detector967.pkl and spm_tokenizer.model 
    # MaskedLM = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-large')
    MaskedLM = torch.load('model/MaskedLM.t0')
    maskedlm_tokenizer = torch.load('model/maskedlm_tokenizer.t0')
    # maskedlm_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')

    detector_tokenizer_path = 'model/spm_tokenizer.model'
    detector_tokenizer = spm.SentencePieceProcessor(detector_tokenizer_path, )
    detector = torch.load(detector_path, map_location=torch.device('cpu'))

    model = HardMasked(detector, MaskedLM, detector_tokenizer, maskedlm_tokenizer, 'cpu')

    return model


def predict(doc, model):
    if '.' not in doc:
        doc += '.'
    sents = doc.split('.')[:-1]
    output = []
    for s in sents:
        text = ''.join([c for c in s if c not in punctuation])
        text = text.split()
        res = model(s)
        check_doc = []
        for i, t in enumerate(text):
            if i == len(text)-1:
                t = t + '.'
                print('end sent:', t)
            if i in res:
                check_doc.append([t, 1, res[i]])
            else:
                check_doc.append([t, 0, []])

        output.extend(check_doc)
        print(output)
    
    return output


