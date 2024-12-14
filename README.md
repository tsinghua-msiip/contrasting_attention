# Enhancing elusive clues in knowledge learning by contrasting attention of language models

This repository provides code for fine-tuning (continual pretraining) language models on a biography dataset and evaluating models's memorization of the biography information on a question-answering task. 

The code implements the proposed method for data augmentation using attention differences between two language models. 

The following commands are used to produce results in the paper:

Extract attention from a language model
```bash
python save_attention.py \
  +model=meta-llama/Meta-Llama-3-8B \
  +dataset=data/biography_real.csv +save_name=attn_dump_llama_3_8b
```

Contrast attention between two language models
```bash
python contrast_attention.py \
  +attn_dump_large=attn_dump_llama_3_70b +attn_dump_small=attn_dump_llama_3_8b \
  +attn_diff=attn_diff_llama_70b_8b
```

Augment the dataset using attention differences
```bash
# augment the dataset (10 times) with random token-dropout
python data_augmentation.py \
  +augment_type=random +alpha=0.7 +multiply=10 \
  +model=meta-llama/Meta-Llama-3-8B \
  +dataset=data/biography_real.csv +save_name=biography_real_aug

# augment the dataset with token-dropout weighted by attention (or attention differences)
python data_augmentation.py \
  +augment_type=attention +alpha=0.7 +beta=0.05 +multiply=10 \
  +attn_weights=attn_diff_llama_70b_7b \
  +model=meta-llama/Meta-Llama-3-8B \
  +dataset=data/biography_real.csv +save_name=biography_real_aug
```

Training

```bash
python train.py --config-name train.yaml +model_dir=model_out
```

Evaluation

```bash
python evaluate.py --config-name evaluate.yaml +model_dir=model_out/checkpoint-130
```

## Citation
```
pending
```
