import hydra
import torch
import transformers
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import tokenize_with_mask


@hydra.main(version_base=None, config_path="config")
def main(config: DictConfig) -> None:
    
    # Prepare model
    model_kwargs = {
        'torch_dtype': torch.bfloat16,
        'device_map': 'auto',
        'quantization_config': BitsAndBytesConfig(**OmegaConf.to_object(config.quant))
    }   
    model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
            model, 
            LoraConfig(**OmegaConf.to_object(config.lora))
        )
    model.enable_input_require_grads()       


    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id            # patch llama tokenizer
        
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_func(eg):
        result, _, _ = tokenize_with_mask(eg["text"], tokenizer)
        return result

    dataset = load_dataset('csv', data_files=config.dataset)["train"]
    dataset = dataset.map(tokenize_func)


    # Prepare trainer
    trainer = Trainer(
        train_dataset=dataset,
        model=model,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.accumulate_grad_batches,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.max_epochs,
            gradient_checkpointing=True,
            optim="adamw_torch",
            learning_rate=config.lr,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type=config.scheduler,
            max_grad_norm=config.gradient_clip_val,
            bf16=True,                
            logging_steps=10,
            evaluation_strategy="no",
            save_strategy="epoch",
            output_dir=config.model_dir,
            overwrite_output_dir=True,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            group_by_length=False,
            report_to="tensorboard",
            seed=config.seed,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False, pad_to_multiple_of=8,
        ),
    )

    # Perform training
    trainer.train()

if __name__ == "__main__":
    main()