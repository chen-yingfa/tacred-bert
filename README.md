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

- Weight Initialization: bert-base-uncased
- Transformers Architecture: 12 layers, 768 hidden size, 12 heads
- Batch size: 8
- Learning Rate: 3e-5 with Adam
- 1/10 of train set (because I don't have enough computing power)

| Input type      | Output type     | Dev F1 |
| --------------- | --------------- | ------ |
| Standard        | [CLS]           | 15.5   |
| Standard        | Mention pooling | 21.3   |
| Positional emb. | Mention pooling |        |
| Entity markers  | [CLS]           | 54.0   |
| Entity markers  | Mention pooling | 48.1   |
| Entity markers  | Entity start    | 69.3   |

## Developer's Note

This repo was created by copying the repo [tacred-relation](https://github.com/yuhaozhang/tacred-relation) and adding and deleting files, but there are still much leftover from that repo. The implementation of BERT model in `bert.py` was copied from Huggingface's Transformers and changing it.