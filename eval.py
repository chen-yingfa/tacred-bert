"""
Train a model on TACRED.
"""

import os
from datetime import datetime
import time
import numpy as np
import argparse
import torch
from torch import nn, optim
from transformers import AdamW
# from model.classifier_bert import BertClassifier
from model.bert import BertForSequenceClassification

from dataset.loader import DataLoader
from dataset.dataset import get_data_loaders, REDataLoader
from utils import scorer, constant, helper, torch_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/tacred-example')
    parser.add_argument('--log_step', type=int, default=50, help='Print log every k steps.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output', type=str, default='logs.txt', help='Write training log to file.')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
    parser.add_argument('--id', type=str, default='best', help='Model ID under which to save models.')
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

    torch_utils.set_seed(12345)

    # make opt
    opt = vars(args)
    opt['num_labels'] = len(constant.LABEL_TO_ID)

    # model dir
    # model
    model_path = f"{opt['save_dir']}/{args.id}/best_model.pt"
    print("model_path:", model_path)
    model = torch.load(model_save_dir)
    model.to(device)
    loader = REDataLoader(
        data_dir + '/test.json',
        label2id,
        tokenize,
        batch_size,
        False)

    # print model info
    helper.print_config(opt)
    criterion = nn.CrossEntropyLoss()

    num_examples = len(train_loader)
    start_time = time.time()
    format_str = '{}: examples {}/{}, loss = {:.4f} ({:.3f} sec/batch)'

    print("Testing on test set...")
    total_loss = 0
    preds = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            for i in range(len(batch)):
                if batch[i] is not None:
                    batch[i] = batch[i].to(device)

            labels, input_ids, att_mask, e1_pos, e2_pos = batch
            
            # pass to model
            logits = model(
                input_ids,
                att_mask=att_mask,
                e1_pos=e1_pos,
                e2_pos=e2_pos,
                output_method=output_method)

            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            list_preds += preds
            total_loss += loss

            # log
            if (i+1) % opt['log_step'] == 0:
                duration = time.time() - start_time
                timestr = '{:%m-%d %H:%M:%S}'.format(datetime.now())
                print(format_str.format(
                    timestr,
                    i+1,
                    num_examples,
                    opt['num_epoch'],
                    loss,
                    duration))
        predictions = [id2label[p] for p in list_preds]
        result = loader.dataset.eval(predictions, True)
        # dev_p = result['micro_p']
        # dev_r = result['micro_r']
        # dev_f1 = result['micro_f1']
    print("Testing finished")


def main():
    args = parse_args()
    test(args)


if __name__ == '__main__':
    main()