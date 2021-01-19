"""
Data loader for TACRED json files.
"""

import os
import pickle
import json
import random
import torch
import numpy as np

from transformers import BertTokenizer

from utils import constant, helper

def get_tokenizer():
    """
    Load or download BertTokenizer
    """
    model_name = "BERT"
    tokenizer_path = f"saved_models/{model_name}_tokenizer.pkl"
    if os.path.isfile(tokenizer_path):
        # load from existing file
        tokenizer = pickle.load(open(tokenizer_path, 'rb'))
        print("Loaded tokenizer from saved path.")
    else:
        # download from pretrained
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True)
        
        # save tokenizer
        pickle.dump(tokenizer, open(tokenizer_path, 'wb'))
        print(f"Saved {model_name} tokenizer at {tokenizer_path}")
    tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]'])

    return tokenizer

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, tokenizer=None, evaluation=False, input_method=None):
        self.batch_size = batch_size
        self.opt = opt
        if tokenizer is None:
            self.tokenizer = get_tokenizer()
        else:
            self.tokenizer = tokenizer
        self.eval = evaluation
        self.input_method = input_method

        # Method 1, standard input
        # Method 2, positional embedding
        # Method 3, entity markers
        assert input_method in [1, 2, 3]  # sanity check

        with open(filename) as infile:
            data = json.load(infile)
        # data = data[:(len(data) // 10)]   # only load a part of dataset, for debug
        data = self.preprocess(data)
        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
        self.labels = [id2label[d[-2]] for d in data] 
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            tokens = d['token']
            if self.opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']

            max_len = 96

            subj_pos = None
            obj_pos = None
            e1_pos_seq = None
            e2_pos_seq = None
            if self.input_method == 2:
                e1_pos_seq = get_pos_seq(ss, se, max_len)
                e2_pos_seq = get_pos_seq(os, oe, max_len)
            elif self.input_method == 3:
                # insert entity markers
                tokens = insert_entity_markers(tokens, (ss, se), (os, oe))
                if ss < os:
                    os += 2
                    oe += 2
                else:
                    ss += 2
                    se += 2
            e1_pos = (ss, se+1)
            e2_pos = (os, oe+1)

            # convert to id
            sent = ''.join(tokens)
            encoded = self.tokenizer(
                sent,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,)
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            relation = constant.LABEL_TO_ID[d['relation']]
            
            processed += [(input_ids, e1_pos_seq, e2_pos_seq, deprel, e1_pos, e2_pos, relation, attention_mask)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        #return 50
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 8

        # sort all fields by lens for easy RNN operations
        # lens = [len(x) for x in batch[0]]
        # batch, orig_idx = sort_all(batch, lens)
        
        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        # pos = get_long_tensor(batch[1], batch_size)
        # ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)

        e1_pos_seq = None
        e2_pos_seq = None
        
        if self.input_method == 2:
            e1_pos_seq = get_long_tensor(batch[1], batch_size)
            e2_pos_seq = get_long_tensor(batch[2], batch_size)

        subj_pos = get_long_tensor(batch[4], batch_size)
        obj_pos = get_long_tensor(batch[5], batch_size)

        rels = torch.LongTensor(batch[6])
        attention_masks = get_long_tensor(batch[7], batch_size)

        return (words, e1_pos_seq, e2_pos_seq, deprel, subj_pos, obj_pos, rels, attention_masks)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_token_type_ids(ss, se, os, oe, length):
    """ Return [0, ..., 0, 1, ..., 1, 0, ..., 0, 2, ..., 2, 0, ..., 0]
                           ^       ^             ^       ^
                           ss      se            os      oe
    """
    res = []
    for i in range(length):
        if ss <= i <= se:
            res.append(1)
        elif os <= i <= oe:
            res.append(2)
        else:
            res.append(0)
    return res

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. 
    [-start, ..., -1, 0, ..., 0, 1, ..., len - end]
                      ^       ^
                 start_idx  end_idx
    """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_pos_seq(start, end, length):
    seq = get_positions(start, end, length)
    return [e + length for e in seq]

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

def insert_entity_markers(tokens: list, e1_pos, e2_pos):
    if e1_pos[0] < e2_pos[0]:
        tokens = insert_elem("[/E2]", tokens, e2_pos[1] + 1)
        tokens = insert_elem("[E2]", tokens, e2_pos[0])
        tokens = insert_elem("[/E1]", tokens, e1_pos[1] + 1)
        tokens = insert_elem("[E1]", tokens, e1_pos[0])
    else:
        tokens = insert_elem("[/E1]", tokens, e1_pos[1] + 1)
        tokens = insert_elem("[E1]", tokens, e1_pos[0])
        tokens = insert_elem("[/E2]", tokens, e2_pos[1] + 1)
        tokens = insert_elem("[E2]", tokens, e2_pos[0])
    return tokens

def insert_elem(elem, lis: list, idx: int):
    return lis[:idx] + [elem] + lis[idx:]