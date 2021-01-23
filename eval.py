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
from model.classifier_bert import BertClassifier
# from model.bert import BertForSequenceClassification

from dataset.data_loader import get_data_loaders, REDataLoader, get_tokenizer
from utils import scorer, constant, helper, torch_utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/tacred-example')
    parser.add_argument('--model_file', type=str, default='best_model.pt')
    parser.add_argument('--log_step', type=int, default=50, help='Print log every k steps.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output', type=str, default='test_result.txt', help='Write training log to file.')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
    parser.add_argument('--name', type=str, default='best', help='Model ID under which to save models.')
    return parser.parse_args()


def test(args):
    # Input and output method
    input_method_name = ["", "standard", "positional_embedding", "entity_markers"]
    output_method_name = ["", "cls_token", "mention_pooling", "entity_start"]
    input_method = 3
    output_method = 3
    print(f"Input method: {input_method_name[input_method]}")
    print(f"Output method: {output_method_name[output_method]}")

    # Constants
    pretrain_path = 'bert-base-uncased'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
    max_length = 128

    torch_utils.set_seed(12345)

    # make opt
    opt = vars(args)
    opt['num_labels'] = len(constant.LABEL_TO_ID)

    # Load Model
    tokenizer = get_tokenizer(pretrain_path)
    model_path = f"{opt['save_dir']}/{args.name}/{args.model_file}"
    print("model_path:", model_path)
    print('')
    model = BertClassifier.from_pretrained(
        pretrain_path, num_labels=42)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.set_tokenizer(tokenizer, max_length, input_method, output_method)

    # model = torch.load(model_save_dir)
    model.to(device)

    # Data loader
    data_path = args.data_dir + '/test.json'
    num_workers = 0 if os.name == 'nt' else 8
    data_loader = REDataLoader(
        data_path,
        constant.LABEL_TO_ID,
        model.tokenize,
        args.batch_size,
        False,
        num_workers=num_workers)
    print(f"\nLoaded {len(data_loader.dataset)} examples from {data_path}")

    # Print model info
    helper.print_config(opt)
    criterion = nn.CrossEntropyLoss()

    format_str = '[{}] batch {}/{}, loss = {:.4f} ({:.3f} sec/batch)'

    print("Testing on test set...")
    total_loss = 0
    list_preds = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            start_time = time.time()

            for j in range(len(batch)):
                if batch[j] is not None:
                    batch[j] = batch[j].to(device)

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

            # predictions = [id2label[p] for p in preds]
            # truth = [id2label[x] for x in labels.cpu().tolist()]
            # prec, recall, f1 = scorer.score(truth, predictions)
            # print(f"prec. = {prec}, recall = {recall}, f1 = {f1}")

            # log
            if (i+1) % args.log_step == 0:
                duration = time.time() - start_time
                timestr = '{:%m-%d %H:%M:%S}'.format(datetime.now())
                print(format_str.format(
                    timestr,
                    i+1,
                    len(data_loader),
                    loss,
                    duration))
        predictions = [id2label[p] for p in list_preds]

        print("Total:")
        result = data_loader.dataset.eval(predictions, True)
        # dev_p = result['micro_p']
        # dev_r = result['micro_r']
        # dev_f1 = result['micro_f1']
    print("Testing finished")


def main():
    args = parse_args()
    test(args)


if __name__ == '__main__':
    main()