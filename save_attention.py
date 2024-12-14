"""
Dump averaged attention weights from a model.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from utils import tokenize_with_mask
from tqdm import tqdm


@hydra.main(version_base=None, config_path="config")
def main(config: DictConfig) -> None:

    # Prepare model
    model_kwargs = {
        'torch_dtype': torch.bfloat16,
        'device_map': 'auto',
        'attn_implementation': 'eager',
        'output_attentions':True
    }   
    model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)


    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    def tokenize_func(eg):
        result, _, _ = tokenize_with_mask(eg["text"], tokenizer)
        return result

    dataset = load_dataset('csv', data_files=config.dataset)["train"]
    dataset = dataset.map(tokenize_func)


    # dump attention weights
    all_attention = []
    with torch.no_grad():
        for eg in tqdm(dataset):
            input_ids = torch.tensor([eg["input_ids"]]).to('cuda')
            attention = model(input_ids)[-1]

            # calculate average attention across all layers and heads
            avg_attention = torch.stack(attention).mean(dim=(0,1,2))

            # calculate average attention to a token from all following tokens
            avg_attention = torch.tril(avg_attention)
            avg_attention = avg_attention.sum(dim=-1) / torch.arange(1, avg_attention.size(-1)+1).to(avg_attention)

            avg_attention = avg_attention.float().cpu().numpy()
            all_attention.append(avg_attention)
    
    with open(f'{config.save_name}.pkl', 'wb') as f:
        pickle.dump(all_attention, f)


if __name__ == "__main__":
    main()