
import hydra
from omegaconf import DictConfig
import pandas as pd
from transformers import AutoTokenizer
from utils import tokenize_with_mask
import random
import csv
import math
import pickle
import numpy as np
import os


def threshold_function(x, alpha, beta):
    return alpha*math.exp(-beta*x)+1-alpha


@hydra.main(version_base=None, config_path=".")
def main(config: DictConfig) -> None:

    # load dataset
    dataset = pd.read_csv(config.dataset)
    tokenizer = AutoTokenizer.from_pretrained(config.model)

    # load attention weights
    if config.augment_type == "attention":
        with open(f'{config.attn_weights}.pkl', 'rb') as f:
            attn_weights = pickle.load(f)

    # augment dataset
    new_dataset = []
    for i, row in dataset.iterrows():
        # tokenize text
        text = row["text"]
        tokenized_text, _, text = tokenize_with_mask(text, tokenizer)

        # calculate dropout proability
        if config.augment_type == "random":
            threshold = [config.alpha] * len(tokenized_text.input_ids)
        elif config.augment_type == "attention":
            attn_sorted = np.argsort(attn_weights[i])[::-1]
            attn_ranking = {pos: rank for rank, pos in enumerate(attn_sorted.tolist())}
            threshold = [threshold_function(attn_ranking[pos], config.alpha, config.beta) for pos in range(len(tokenized_text.input_ids))]
        else:
            raise ValueError("Invalid augment_type")

        # dropout tokens to create new text
        for _ in range(config.multiply):
            new_text = text
            for pos, token_id in enumerate(tokenized_text.input_ids):
                if tokenizer.convert_ids_to_tokens(token_id) in [",", ".", "<bos>", "<s>"]:
                    continue
                else:
                    if random.random() > threshold[pos]:
                        token_start, token_end = tokenized_text.offset_mapping[pos]
                        new_text = new_text[:token_start] + ' ' * (token_end-token_start) + new_text[token_end:]
            new_dataset.append(new_text)
    

    # write augmented dataset
    os.makedirs("data_augmented", exist_ok=True)
    with open(f'data_augmented/{config.save_name}.csv', 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["text"])
        for text in new_dataset:
            writer.writerow([text])

if __name__ == "__main__":
    main()
        