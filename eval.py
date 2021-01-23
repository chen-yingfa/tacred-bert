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
from torch import nn, optim
from transformers import AdamW
from model.classifier_bert import BertClassifier
# from model.bert import BertForSequenceClassification
from matplotlib import pyplot as plt

from dataset.loader import DataLoader, get_tokenizer
from dataset.dataset import get_data_loaders
from utils import scorer, constant, helper, torch_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/tacred')
    parser.add_argument('--log_step', type=int, default=50, help='Print log every k steps.')
    parser.add_argument('--output', type=str, default='logs.txt', help='Write training log to file.')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
    parser.add_argument('--id', type=str, default='test', help='Model ID under which to save models.')
    return parser.parse_args()


def test(args):
    # method
    input_method_name = ["", "standard", "positional_embedding", "entity_markers"]
    output_method_name = ["", "cls_token", "mention_pooling", "entity_start"]
    input_method = 3
    output_method = 3
    print(f"Input method: {input_method_name[input_method]}")
    print(f"Output method: {output_method_name[output_method]}")

    # constants
    pretrain_path = 'bert-base-uncased'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
    lr = args.lr
    weight_decay = 1e-5
    warmup_step = 300
    max_length = 128

    torch_utils.set_seed(12345)

    # make opt
    opt = vars(args)
    opt['num_labels'] = len(constant.LABEL_TO_ID)

    grad_acc_steps = 64 // args.batch_size

    # model
    tokenizer = get_tokenizer(pretrain_path)
    model = BertForSequenceClassification.from_pretrained(
        pretrain_path,
        num_labels=len(id2label))
    model = BertClassifier.from_pretrained(
        pretrain_path,
        num_labels=len(id2label))
    )
    
    model.set_tokenizer(tokenizer, max_length)
    model.to(device)
    train_loader, dev_loader, test_loader = get_data_loaders(
        opt['data_dir'],
        model.tokenize,
        opt['batch_size'])

    # model dir
    model_save_dir = opt['save_dir'] + '/' + args.id
    opt['model_save_dir'] = model_save_dir
    helper.ensure_dir(model_save_dir, verbose=True)
    print("model_save_dir:", model_save_dir)

    # save config
    helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
    file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_f1")

    # print model info
    helper.print_config(opt)

    # train_steps = len(train_loader) // opt['batch_size'] * opt['num_epoch']
    train_steps = len(train_loader) * opt['num_epoch']

    optimizer = torch_utils.get_optimizer(opt['optim'], model, lr, weight_decay)
    scheduler = torch_utils.get_scheduler(optimizer, train_steps, warmup_step)
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    global_start_time = time.time()
    format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.4f} ({:.3f} sec/batch)'

    list_train_loss = []
    list_dev_loss = []
    list_dev_f1 = []

    # start training
    for epoch in range(1, opt['num_epoch']+1):
        train_loss = 0
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            start_time = time.time()

            # labels, input_ids, att_mask, e1_pos, e2_pos, e1_pos_seq, e2_pos_seq = batch
            labels, input_ids, att_mask, e1_pos, e2_pos = batch

            for i in range(len(batch)):
                if batch[i] is not None:
                    batch[i] = batch[i].to(device)

            # pass to model
            logits = model(
                input_ids,
                att_mask=att_mask,
                e1_pos=e1_pos,
                e2_pos=e2_pos,
                # e1_pos_seq=e1_pos_seq,
                # e2_pos_seq=e2_pos_seq,
                output_method=output_method)

            # print(logits)

            loss = criterion(logits, labels)

            # log
            if global_step % opt['log_step'] == 0:
                duration = time.time() - start_time
                timestr = '{:%m-%d %H:%M:%S}'.format(datetime.now())
                print(format_str.format(timestr, global_step, train_steps, epoch,\
                        opt['num_epoch'], loss, duration))



            # optimize
            loss.backward()
            # if (i + 1) % grad_acc_steps == 0: # gradient accumulation

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            train_loss += loss
            global_step += 1
        

        # eval on dev
        print("Evaluating on dev set...")
        predictions = []
        dev_loss = 0
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dev_loader):
                # labels, input_ids, att_mask, e1_pos, e2_pos, e1_pos_seq, e2_pos_seq = batch
                labels, input_ids, att_mask, e1_pos, e2_pos = batch

                # change device
                labels = labels.to(device)
                input_ids = input_ids.to(device)
                att_mask = att_mask.to(device)
                e1_pos = e1_pos.to(device)
                e2_pos = e2_pos.to(device)
                if input_method == 2:
                    e1_pos_seq = e1_pos_seq.to(device)
                    e2_pos_seq = e2_pos_seq.to(device)
                
                # pass to model
                logits = model(
                    input_ids,
                    att_mask=att_mask,
                    e1_pos=e1_pos,
                    e2_pos=e2_pos,
                    # e1_pos_seq=e1_pos_seq,
                    # e2_pos_seq=e2_pos_seq,
                    output_method=output_method)

                # logits = outputs.logits
                loss = criterion(logits, labels)
                preds = torch.argmax(logits, dim=1).cpu().tolist()

                predictions += preds
                dev_loss += loss
            predictions = [id2label[p] for p in predictions]
            # dev_p, dev_r, dev_f1 = scorer.score(dev_loader.gold(), predictions)
            result = dev_loader.dataset.eval(predictions, True)
            dev_p = result['micro_p']
            dev_r = result['micro_r']
            dev_f1 = result['micro_f1']

            # log avg loss per batch
            train_loss = train_loss / len(train_loader.dataset) * opt['batch_size']
            dev_loss = dev_loss / len(dev_loader.dataset) * opt['batch_size']
            # train_loss = train_loss / train_loader.num_examples * opt['batch_size']
            # dev_loss = dev_loss / dev_loader.num_examples * opt['batch_size']
            print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,\
                    train_loss, dev_loss, dev_f1))
            file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_f1))

            # save
            model_file = model_save_dir + '/ckpt_epoch_{}.pt'.format(epoch)
            # model.save(model_file, epoch)
            # torch_utils.save(model, optim, opt, filename=model_file)
            torch.save({'state_dict': model.state_dict()}, model_file)
            if len(list_dev_f1) == 0 or dev_f1 > max(list_dev_f1):      # best model
                copyfile(model_file, model_save_dir + '/best_model.pt')
                print("new best model saved.")
            if epoch % opt['save_epoch'] != 0:
                os.remove(model_file)

            print("")

            # plot and save figure
            list_train_loss.append(train_loss)
            list_dev_loss.append(dev_loss)
            list_dev_f1.append(dev_f1)
            plot_and_save(list_list_loss, list_dev_loss, list_dev_f1, model_save_dir)

    print("Training ended with {} epochs.".format(epoch))


def main():
    args = parse_args()
    test(args)


if __name__ == '__main__':
    main()