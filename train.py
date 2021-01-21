"""
Train a model on TACRED.
"""

import os
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AdamW
from model.bert import BertForSequenceClassification
from matplotlib import pyplot as plt

from dataset.loader import DataLoader, get_tokenizer
from utils import scorer, constant, helper, torch_utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/tacred')
    parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
    parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
    parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
    parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
    parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')
    parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
    parser.add_argument('--no-lower', dest='lower', action='store_false')
    parser.set_defaults(lower=False)

    parser.add_argument('--attn', dest='attn', action='store_true', help='Use attention layer.')
    parser.add_argument('--no-attn', dest='attn', action='store_false')
    parser.set_defaults(attn=True)
    parser.add_argument('--attn_dim', type=int, default=200, help='Attention size.')
    parser.add_argument('--pe_dim', type=int, default=30, help='Position encoding dimension.')

    parser.add_argument('--lr', type=float, default=1.0, help='Applies to SGD and Adagrad.')
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--optim', type=str, default='sgd', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--num_epoch', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=50, help='Print log every k steps.')
    parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
    parser.add_argument('--save_epoch', type=int, default=1, help='Save model checkpoints every k epochs.')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
    parser.add_argument('--id', type=str, default='test', help='Model ID under which to save models.')
    parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    return parser.parse_args()

args = parse_args()

# method
input_method_name = ["", "standard", "positional_embedding", "entity_markers"]
output_method_name = ["", "cls_token", "mention_pooling", "entity_start"]
input_method = 3
output_method = 3
print(f"Input method: {input_method_name[input_method]}")
print(f"Output method: {output_method_name[output_method]}")
model_name = 'bert-base-uncased'

grad_acc_steps = 64 // args.batch_size

# seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# make opt
opt = vars(args)
opt['num_class'] = len(constant.LABEL_TO_ID)

# load data
tokenizer = get_tokenizer(model_name)
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, tokenizer, evaluation=False, input_method=input_method)
dev_batch = DataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, tokenizer, evaluation=True, input_method=input_method)

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)
print("model_save_dir:", model_save_dir)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
# vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_f1")

# print model info
helper.print_config(opt)

# model
# model = RelationModel(opt, emb_matrix=emb_matrix)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(constant.LABEL_TO_ID)
)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# optimizer
optim = AdamW(model.parameters(), lr=3e-5)

id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
dev_f1_history = []
current_lr = opt['lr']

loss_func = nn.CrossEntropyLoss()
global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']

list_train_loss = []
list_dev_loss = []
list_dev_f1 = []

# start training
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    model.train()
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1

        input_ids, e1_pos_seq, e2_pos_seq, e1_pos, e2_pos, labels, att_mask = batch

        # change device
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        att_mask = att_mask.to(device)
        e1_pos = e1_pos.to(device)
        e2_pos = e2_pos.to(device)
        if input_method == 2:
            e1_pos_seq = e1_pos_seq.to(device)
            e2_pos_seq = e2_pos_seq.to(device)

        # pass to model
        outputs = model(
            input_ids,
            attention_mask=att_mask,
            e1_pos=e1_pos,
            e2_pos=e2_pos,
            e1_pos_seq=e1_pos_seq,
            e2_pos_seq=e2_pos_seq,
            labels=labels,
            output_method=output_method)

        loss = outputs.loss
        loss.backward()
        if (i + 1) % grad_acc_steps == 0:
            optim.step()
            optim.zero_grad()

        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))

    # eval on dev
    print("Evaluating on dev set...")
    predictions = []
    dev_loss = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dev_batch):
            input_ids, e1_pos_seq, e2_pos_seq, e1_pos, e2_pos, labels, att_mask = batch

            # change device
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            att_mask = att_mask.to(device)
            e1_pos = e1_pos.to(device)
            e2_pos = e2_pos.to(device)
            if input_method == 2:
                e1_pos_seq = e1_pos_seq.to(device)
                e2_pos_seq = e2_pos_seq.to(device)
            
            # pass to model
            outputs = model(
                input_ids,
                attention_mask=att_mask,
                e1_pos=e1_pos,
                e2_pos=e2_pos,
                e1_pos_seq=e1_pos_seq,
                e2_pos_seq=e2_pos_seq,
                labels=labels,
                output_method=output_method)

            logits = outputs.logits
            loss = loss_func(logits, labels)
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            predictions += preds
            dev_loss += loss
        predictions = [id2label[p] for p in predictions]
        dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions)
        
        train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
        dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']
        print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,\
                train_loss, dev_loss, dev_f1))
        file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_f1))

        # save
        model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
        # model.save(model_file, epoch)
        torch_utils.save(model, optim, opt, filename=model_file)
        if epoch == 1 or dev_f1 > max(dev_f1_history):
            copyfile(model_file, model_save_dir + '/best_model.pt')
            print("new best model saved.")
        if epoch % opt['save_epoch'] != 0:
            os.remove(model_file)
        
        # lr schedule
        if len(dev_f1_history) > 10 and dev_f1 <= dev_f1_history[-1] and \
                opt['optim'] in ['sgd', 'adagrad']:
            current_lr *= opt['lr_decay']
            # model.update_lr(current_lr)
            torch_utils.change_lr(optim, current_lr)

        dev_f1_history += [dev_f1]
        print("")

        # plot and save figure
        list_train_loss.append(train_loss)
        list_dev_loss.append(dev_loss)
        list_dev_f1.append(dev_f1)
        plt.xlabel("epoch")
        plt.plot(list_train_loss, label="train loss")
        plt.plot(list_dev_loss, label="dev loss")
        plt.plot(list_dev_f1, label="dev f1")
        plt.legend()
        plt.savefig(model_save_dir + '/loss_f1_vs_epoch.png')
        plt.clf()
        plt.close()

print("Training ended with {} epochs.".format(epoch))

