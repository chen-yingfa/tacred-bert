tacred-bert
=========================

This repo is an implementation of the model described in the paper [Matching the Blanks: Distributional Similarity for Relation Learning](https://arxiv.org/abs/1906.03158) (only the first half, did not implement pre-training with Matching the Blanks task).

**The TACRED dataset**: Details on the TAC Relation Extraction Dataset can be found on [this dataset website](https://nlp.stanford.edu/projects/tacred/).

## Requirements

- Python 3 (tested on 3.6.11)
  - PyTorch (tested on 1.7.0)
  - Transformers (tested on 4.2.1)

## Usage

Train
```
python train.py --id "model_folder_name" --data_dir dataset/tacred
```

Model checkpoints and logs will be saved to `saved_models/model_folder_name`.

Choose input and output representation by setting value of `input_method` and `output_method` in `train.py` . 

Input methods:

1. Standard
2. Positional embedding
3. Entity markers

Output methods:

1. [CLS] token
2. Mention pooling
3. Entity start

## Evaluation

Not implemented yet.

## Results

**Note:** some of the following results might not be correct, because I am messy and don't have time to double check.

Parameters:

- Weight initialization: bert-base-uncased
- Transformers architecture: 12 layers, 768 hidden size, 12 heads
- Batch size: 64
- Learning rate: 2e-5 with AdamW,
  - Warmup steps: 300
  - Weight decay: 1e-5

| Input type      | Output type     | Dev F1 | Test F1 |
| --------------- | --------------- | ------ | ------- |
| Standard        | [CLS]           | 23.2   |         |
| Standard        | Mention pooling | 65.7   |         |
| Positional emb. | Mention pooling |        |         |
| Entity markers  | [CLS]           | 65.8   |         |
| Entity markers  | Mention pooling | 67.5   |         |
| Entity markers  | Entity start    | 69.3   | 68.5    |

## Developer's Note

This repo was created by copying the repo [tacred-relation](https://github.com/yuhaozhang/tacred-relation) and adding and deleting files, but there are still much leftover from that repo. The implementation of BERT model in `bert.py` was copied from Huggingface's Transformers and changing it.