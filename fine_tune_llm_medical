import os
import sys
from typing import List

import fire
import wandb
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.prompter import Prompter
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

def print_params(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

def initialize_model(base_model, device_map):
    model = AutoModelForCausalLM.from_pretrained(base_model, load_in_8bit=True, torch_dtype=torch.float16, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    return model, tokenizer

def tokenize(prompt, tokenizer, cutoff_len, add_eos_token=True):
    result = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
    if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len and add_eos_token:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point, prompter, tokenizer, train_on_inputs):
    full_prompt = prompter.generate_prompt(data_point["instruction"], data_point["input"], data_point["output"])
    tokenized_full_prompt = tokenize(full_prompt, tokenizer)
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(data_point["instruction"], data_point["input"])
        tokenized_user_prompt = tokenize(user_prompt, tokenizer, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
    return tokenized_full_prompt

def setup_wandb(wandb_project, wandb_watch, wandb_log_model):
    use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    return use_wandb

def train_model(model, train_data, val_data, micro_batch_size, gradient_accumulation_steps, world_size, ddp, use_wandb, wandb_run_name):
    gradient_accumulation_steps = micro_batch_size // gradient_accumulation_steps

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    class SavePeftModelCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_folder)
            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.1,
            fp16=True,
            logging_steps=8,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_data is not None else "no",
            save_strategy="steps",
            eval_steps=32 if val_data is not None else None,
            save_steps=32,
            output_dir=output_dir,
            save
