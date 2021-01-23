import torch
from torch import nn

from .bert import BertModel
from transformers import BertPreTrainedModel
# from transformers import BertModel, BertPreTrainedModel

class BertClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier_mention_pooling = nn.Linear(config.hidden_size * 2, config.num_labels)
        self.classifier_entity_start = nn.Linear(config.hidden_size * 2, config.num_labels)

        self.linear = nn.Linear(2 * config.hidden_size, 2 * config.hidden_size)

        self.init_weights()

    def set_tokenizer(self, tokenizer, max_length, input_method, output_method):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_method = input_method
        self.output_method = output_method

    def forward(
        self,
        input_ids=None,
        att_mask=None,
        e1_pos=None,
        e2_pos=None,
    ):
        # print("e1_pos:", e1_pos)
        # print(e1_pos.shape)
        # print("e2_pos:", e2_pos)
        # print(e2_pos.shape)

        if self.input_method == 2:
            # create index: [0, 1, ..., k, k, ..., ]
            dev = input_ids.device
            p1 = []
            p2 = []
            for p in e1_pos:
                i0 = self.max_length - p[0]
                i1 = self.max_length
                i2 = 2 * self.max_length - p[1]
                a = torch.arange(i0, i1).to(dev)
                b = torch.full((p[1] - p[0] + 1,), i1).long().to(dev)
                try:
                    c = torch.arange(i1 + 1, i2).to(dev)
                except:
                    print(e1_pos)
                    print(e2_pos)
                    print(p)
                    print(i1, i2)
                    exit(0)
                seq = torch.cat((a, b, c), dim=0)      # (H,)
                p1.append(seq)

            for p in e2_pos:
                i0 = self.max_length - p[0]
                i1 = self.max_length
                i2 = 2 * self.max_length - p[1] + 1
                a = torch.arange(i0, i1).to(dev)
                b = torch.full((p[1] - p[0],), i1).long().to(dev)
                c = torch.arange(i1 + 1, i2).to(dev)
                seq = torch.cat((a, b, c), dim=0)      # (H,)
                p2.append(seq)

            e1_pos_seq = torch.stack(p1)    # (B, H)
            e2_pos_seq = torch.stack(p2)    # (B, H)
            # print("e1_pos_seq:", e1_pos_seq)
            # print(e1_pos_seq.shape)
            # print("e2_pos_seq:", e2_pos_seq)
            # print(e2_pos_seq.shape)

            outputs = self.bert(
                input_ids,
                attention_mask=att_mask,
                e1_pos_seq=e1_pos_seq,
                e2_pos_seq=e2_pos_seq,
            )
        else:
            outputs = self.bert(input_ids, attention_mask=att_mask)

        # Method 1, [CLS] token (default)
        # Method 2, Entity mention pooling
        # Method 3, Entity start
        if self.output_method == 1:
            x = outputs[1]              # (H)
            x = self.dropout(x)         # (H)
            logits = self.classifier(x) # (C)
        elif self.output_method == 2:
            # Assume that e1_pos, e2_pos are (B, 2), where e1_pos[i] is (start, end)
            last_hidden_states = outputs[0] # (B, L, H)
            batch_size = input_ids.shape[0]

            # Because entity length differs from batch to batch, so have to handle individually
            embeds = []
            for i in range(batch_size):
                # First handle entity 1
                e1_embed = last_hidden_states[i, e1_pos[i][0]:e1_pos[i][1]+1, :]  # (K, H), k is token count in entity
                e1_embed = e1_embed.permute(1, 0)                               # (H, K)
                if e1_embed.shape[1] > 1:
                    # pool
                    pooling = nn.MaxPool1d(kernel_size=e1_embed.shape[1], stride=1)
                    e1_embed = e1_embed.unsqueeze(0)
                    e1_embed = pooling(e1_embed)    # (H, 1)
                e1_embed = e1_embed.squeeze()  # (H)
                
                # Repeat for entity 2
                e2_embed = last_hidden_states[i, e2_pos[i][0]:e2_pos[i][1]+1, :]
                e2_embed = e2_embed.permute(1, 0)
                if e2_embed.shape[1] > 1:
                    # pool
                    pooling = nn.MaxPool1d(e2_embed.shape[1], stride=1)
                    e2_embed = e2_embed.unsqueeze(0)
                    e2_embed = pooling(e2_embed)
                e2_embed = e2_embed.squeeze()   # (H)

                embed = torch.cat((e1_embed, e2_embed), dim=0)  # concatenate output for two entities (2*H)
                embeds.append(embed)
            x = torch.stack(embeds, dim=0)      # (B, H), final output representation
            # classify
            x = self.linear(x)                           # (B, 2H)
            x = self.dropout(x)                          # (B, 2H)
            logits = self.classifier_mention_pooling(x)  # (B, C)
        elif self.output_method == 3:
            # e1_pos: (B, 1)
            hidden_states = outputs[0]          # (B, L, H)
            batch_size = e1_pos.shape[0]

            # Get embedding of start markers
            onehot1 = torch.zeros(hidden_states.size()[:2]).float().to(input_ids.device)  # (B, L)
            onehot2 = torch.zeros(hidden_states.size()[:2]).float().to(input_ids.device)  # (B, L)
            onehot1 = onehot1.scatter_(1, e1_pos, 1)                    # (B, L)
            onehot2 = onehot2.scatter_(1, e2_pos, 1)                    # (B, L)
            hidden1 = (onehot1.unsqueeze(2) * hidden_states).sum(1)     # (B, H)
            hidden2 = (onehot2.unsqueeze(2) * hidden_states).sum(1)     # (B, H)
            x = torch.cat([hidden1, hidden2], 1)                        # (B, 2H)
            
            x = self.linear(x)                          # (B, 2H)
            x = self.dropout(x)                         # (B, 2H)
            logits = self.classifier_entity_start(x)    # (B, C)
        else:
            raise ValueError("Method must be one of [1, 2, 3]")
        return logits
    
    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
        else:
            sentence = item['token']
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False
        
        sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
        ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
        sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
        ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
        sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

        if self.input_method == 3: 
            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']

        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        
        # make sure pos index are not > max_length
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Position
        if self.output_method == 2:     # Mention pooling
            # compute end pos
            if not rev:
                end1 = pos1 + len(ent0) - 1
                end2 = pos2 + len(ent1) - 1
            else:
                end1 = pos1 + len(ent1) - 1
                end2 = pos2 + len(ent0) - 1
            end1 = min(self.max_length - 1, end1)
            end2 = min(self.max_length - 1, end2)

            pos1 = torch.tensor([[pos1, end1]]).long()
            pos2 = torch.tensor([[pos2, end2]]).long()
        else:
            pos1 = torch.tensor([[pos1]]).long()
            pos2 = torch.tensor([[pos2]]).long()
            

        # Padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)  # 0 is id for [PAD]
        indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2


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
