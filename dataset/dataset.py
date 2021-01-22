import json
import torch
from torch.utils.data import Dataset, DataLoader



class REDataset(Dataset):
    """
    Sentence-level relation extraction dataset
    """
    def __init__(self, path, rel2id, tokenizer, kwargs):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
        """
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.kwargs = kwargs
    
        # load the file
        self.data = []
        jsondata = json.load(open(path))
        self.data = self.preprocess(jsondata)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        seq = list(self.tokenizer(item, **self.kwargs))
        res = [self.rel2id[item['relation']]] + seq
        return res # label, seq1, seq2, ...
    
    def collate_fn(data):
        data = list(zip(*data))
        labels = data[0]
        seqs = data[1:]
        batch_labels = torch.tensor(labels).long() # (B)
        batch_seqs = []
        for seq in seqs:
            batch_seqs.append(torch.cat(seq, 0)) # (B, L)
        return [batch_labels] + batch_seqs
    
    def eval(self, preds, use_name=False):
        """
        Args:
            preds: a list of predicted label (id)
                Make sure that the `shuffle` param is set to `False` when getting the loader.
            use_name: if True, `preds` contains predicted relation names instead of ids
        Return:
            F1-score
        """
        correct = 0
        total = len(self.data)
        correct_positive = 0
        pred_positive = 0
        gold_positive = 0
        neg = -1
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others']:
            if name in self.rel2id:
                if use_name:
                    neg = name
                else:
                    neg = self.rel2id[name]
                break
        for i in range(total):
            if use_name:
                golden = self.data[i]['relation']
            else:
                golden = self.rel2id[self.data[i]['relation']]
            if golden == preds[i]:
                correct += 1
                if golden != neg:
                    correct_positive += 1
            if golden != neg:
                gold_positive +=1
            if preds[i] != neg:
                pred_positive += 1
        acc = float(correct) / float(total)
        try:
            micro_p = float(correct_positive) / float(pred_positive)
        except:
            micro_p = 0
        try:
            micro_r = float(correct_positive) / float(gold_positive)
        except:
            micro_r = 0
        try:
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
        except:
            micro_f1 = 0
        print( "Precision (micro): {:.3%}".format(micro_p) )
        print( "   Recall (micro): {:.3%}".format(micro_r) )
        print( "       F1 (micro): {:.3%}".format(micro_f1) )
        result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
        # logging.info('Evaluation result: {}.'.format(result))
        return result

    def preprocess(self, data):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            tokens = d['token']
            ss = int(d['subj_start'])
            se = int(d['subj_end']) + 1
            os = int(d['obj_start'])
            oe = int(d['obj_end']) + 1
            dic = {
                "token": tokens, 
                "h": {"name": " ".join(tokens[ss:se]), "pos": [ss, se]},
                "t": {"name": " ".join(tokens[os:oe]), "pos": [os, oe]},
                "relation": d["relation"]}
            processed.append(dic)
        return processed

def REDataLoader(
        path,
        rel2id,
        tokenizer,
        batch_size,
        shuffle,
        num_workers=8,
        collate_fn=REDataset.collate_fn,
        **kwargs):
    dataset = REDataset(path=path, rel2id=rel2id, tokenizer=tokenizer, kwargs=kwargs)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn)
    return data_loader