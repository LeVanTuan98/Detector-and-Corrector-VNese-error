
# import os
# import json
import logging
import os.path
import re

# import numpy as np
import torch
# from torch.nn import functional as F
import torch.nn as nn

import sentencepiece as spm
# from seqeval.metrics import precision_score as seq_precision, recall_score as seq_recall, f1_score as seq_f1
from transformers import AutoTokenizer, XLMRobertaModel, XLMRobertaForMaskedLM
# from transformers import AdamW, get_linear_schedule_with_warmup


# from tqdm.notebook import tqdm
# from easydict import EasyDict
# import gc
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# import pickle
from pyunpack import Archive

logger = logging.getLogger(__name__)


# MODEL #

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
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=self.input_dim, embedding_dim=self.embedding_dim, )
        self.LSTM = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True, dropout=0.1, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size*2, self.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)
        outputs, (h_n, h_c) = self.LSTM(emb)
        logits = self.linear(outputs)

        p = self.sigmoid(logits)
        return p


class HardMasked(nn.Module):
    def __init__(self, detector, MaskedLM, detector_tokenizer, maskedlm_tokenizer,device):
        super(HardMasked, self).__init__()

        self.detector = detector.to(device)
        self.MaskedLM = MaskedLM.to(device)
        self.detector_tokenizer = detector_tokenizer
        self.maskedlm_tokenizer = maskedlm_tokenizer
        self.use_device = device

    def forward(self, s):
        maskedlm_features, list_miss = self.prepare_input(s)
        outputs = self.MaskedLM(input_ids=torch.tensor([maskedlm_features['input_ids']], dtype=torch.long, device=self.use_device),
                            attention_mask=torch.tensor([maskedlm_features['attention_mask']], dtype=torch.long, device=self.use_device))
        logits = outputs['logits'][0]
        # print(outputs['logits'])
        # output_ids = torch.argmax(logits, dim= -1)
        output_ids = torch.argsort(logits, dim=-1, descending=True)
        sent = s.split()
        final_output = []
        for i in range(len(sent)):
            if i in list_miss:
                out = self.maskedlm_tokenizer.decode(output_ids[i+1, :10])
                final_output.append([sent[i], 1, out.split()])
            else:
                final_output.append([sent[i], 0, []])
        return final_output
        
    def prepare_input(self, s):
        detector_input_ids = self.detector_tokenizer.encode(s, out_type = int)
        detector_input_pieces = self.detector_tokenizer.id_to_piece(detector_input_ids)
        # print(detector_input_pieces)
        detector_outputs = (self.detector(torch.tensor([detector_input_ids], dtype = torch.long, device = self.use_device))[0].reshape(1,-1) > 0.1).int()[0]

        # print(detector_outputs)
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

        msk = masked_s.split()
        # print(masked_s)
        list_miss = []
        for i in range(len(msk)):
          if msk[i] == '<mask>':
            list_miss.append(i)
        
        maskedlm_features = self.maskedlm_tokenizer(masked_s)

        return maskedlm_features, list_miss


def load_model():
    print('Load Model')

    detector_path = 'model/Detector_final.pkl'

    detector_tokenizer_path = 'Text_correction/spm_tokenizer.model'

    if not os.path.isfile(detector_path):
        print('Extract model')
        extract_model()

    MaskedLM = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-large')

    maskedlm_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')

    detector_tokenizer = spm.SentencePieceProcessor(detector_tokenizer_path, )

    detector = torch.load(detector_path, map_location=torch.device('cpu'))

    model = HardMasked(detector, MaskedLM, detector_tokenizer, maskedlm_tokenizer, 'cpu')

    return model


def extract_model():
    Archive("model/Detector_final.part1.rar").extractall("model/")


def predict(model=None, input=''):
    # s = 'xe đạp lách cách tôi vẫn chưa quen, đường thì tối chơi vơi còn tôi vẫn cứ đứng đợi'
    model(input)

    # result = [('aa', 1, ('a', 'b', 'c')), ('bb', 0, ()), ('cc', 1, ('f', 'b', 'e')), ('dd', 0, ()), ('ee', 0, ()),
    #           ('aa', 1, ('a', 'v', 'c')), ('bb', 0, ()), ('cc', 1, ('a', 'v', 'c')), ('dd', 0, ()), ('ee', 0, ())]
    return model(input)

