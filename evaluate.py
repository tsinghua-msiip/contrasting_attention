import os
import hydra
import torch
import json
import random
import pandas as pd
from peft import PeftModel
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase
import re
from datasets import load_metric
from tqdm import tqdm


def generate_in_batches(model, tokenizer, prompts, batch_size, max_new_tokens):
    results = []

    current_idx = 0
    with torch.no_grad():
        while current_idx < len(prompts):
            batch_prompts = prompts[current_idx:current_idx + batch_size]
            input_ids = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=False).input_ids.cuda()

            output_ids = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False)

            output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for input_text, output_text in zip(batch_prompts, output_texts):
                output_text = output_text[len(input_text):]
                # truncate answer at the first \n
                output_text = output_text.split('\n')[0]
                results.append(output_text)

            current_idx += batch_size

    return results


def find_field(field, text):
    """Find the field in the text enclosed in <field> </field> tags."""
    return re.search(rf'<{field}>(.*?)</{field}>', text).group(1)


squad_metric = load_metric("squad")

def em_and_f1(predictions, references):
    """f1 score adapted from the SQuAD evaluation script"""
    assert len(predictions) == len(references)
    predictions = [{'prediction_text': t, 'id': i} for i, t in enumerate(predictions)]
    references = [{'answers': {'answer_start': [0], 'text': [t]}, 'id': i} for i, t in enumerate(references)]
    return squad_metric.compute(predictions=predictions, references=references)



@hydra.main(version_base=None, config_path="config")
def main(config: DictConfig) -> None:
    
    # Prepare model
    model_kwargs = {
        'torch_dtype': torch.bfloat16,
        'device_map': 'auto',
        'quantization_config': BitsAndBytesConfig(**OmegaConf.to_object(config.quant))
    }
    model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
    model = PeftModel.from_pretrained(model, config.model_dir)
    model.eval()

    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id            # patch llama tokenizer
        
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = 'left'

    random.seed(config.seed)

    fewshot_egs = pd.read_csv(config.fewshot_data)   # load few-shot examples
    egs = pd.read_csv(config.eval_data)              # load evaluation examples
    prompts = pd.read_csv(config.prompt_data)            # load prompts

    for field, prompt in prompts.values:
        # generate inputs
        inputs, answers = [], []
        for eg, in tqdm(egs.values):
            name, answer = find_field('name', eg), find_field(field, eg)
            # select few-shot examples
            fewshot = random.sample(list(fewshot_egs.values), 5)
            fewshot = [[find_field('name', eg), find_field(field, eg)] for eg, in fewshot]
            # generate prompt
            text = '\n\n'.join(prompt.format(name=name) + ' ' + answer for name, answer in fewshot)
            text += '\n\n' + prompt.format(name=name)
            inputs.append(text)
            answers.append(answer)
        
        # run generation
        results = generate_in_batches(model, tokenizer, inputs, config.batch_size, config.max_new_tokens)

        # evaluate
        accuracy = em_and_f1(results, answers)
        print(f"Field: {field}, Exact match: {accuracy['exact_match']:.2f}%, F1: {accuracy['f1']:.2f}%")


if __name__ == "__main__":
    main()