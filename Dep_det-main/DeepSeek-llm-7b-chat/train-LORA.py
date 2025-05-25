import os
import torch
import numpy as np
from dataclasses import dataclass, field
from transformers import (
    Trainer, TrainingArguments, AutoModelForSequenceClassification,
    AutoTokenizer, DataCollatorWithPadding
)
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from sklearn.model_selection import train_test_split

MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"
CACHE_DIR = "/workspace/deepseek_env"
DATA_PATH = "/workspace/features/deepseek_balanced_dataset.pt"
OUTPUT_DIR = "/workspace/models/deepseek_classification"
CHECKPOINT_DIR = "/workspace/models/deepseek_classification/checkpoint-22000"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=MODEL_NAME)
    use_lora: bool = True
    trainable: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
    lora_rank: int = 8
    lora_dropout: float = 0.1
    lora_alpha: float = 32.0
    modules_to_save: str = "embed_tokens,lm_head"

@dataclass
class TrainingConfig(TrainingArguments):
    output_dir: str = field(default=OUTPUT_DIR)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=250)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=250)
    save_total_limit: int = field(default=5)
    learning_rate: float = field(default=1e-5)
    optim: str = field(default="adamw_torch")
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    num_train_epochs: int = field(default=10)
    weight_decay: float = field(default=0.01)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    # tf32: bool = field(default=True)
    max_grad_norm: float = field(default=1)
    logging_steps: int = field(default=50)
    warmup_ratio: float = field(default=0.1)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = field(default=False)
    resume_from_checkpoint: str = field(default=CHECKPOINT_DIR)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=1)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def build_model(model_args):
    """Load DeepSeek model and configure for classification with local caching."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=4,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.config.id2label = {0: "non", 1: "mild", 2: "moderate", 3: "severe"}
    model.config.label2id = {"non": 0, "mild": 1, "moderate": 2, "severe": 3}

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=CACHE_DIR
    )

    #Fix missing pad token (Set it explicitly)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    #Ensure proper padding
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt", padding=True)

    #Apply LoRA
    if model_args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules=model_args.trainable.split(','),
            inference_mode=False,
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            modules_to_save=model_args.modules_to_save.split(','),
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer, data_collator

def train():
    """Train model with preprocessed dataset."""
    model_args = ModelArguments()
    training_args = TrainingConfig()

    #Load pre-tokenized dataset
    print("Loading preprocessed dataset...")
    data = torch.load(DATA_PATH)
    input_ids = data["input_ids"]
    attention_masks = data["attention_masks"]
    labels = data["labels"]

    #Convert to Hugging Face Dataset format
    dataset = Dataset.from_dict({
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_masks.tolist(),
        "labels": labels.tolist()
    })

    dataset = dataset.shuffle(seed=42)

    #Train-Test Split 80-20
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    #Load model, tokenizer, and collator
    model, tokenizer, data_collator = build_model(model_args)

    print("Training started!")

    #Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if os.path.exists(CHECKPOINT_DIR):
        print(f"Resuming training from checkpoint: {CHECKPOINT_DIR}")
        trainer.train(resume_from_checkpoint=CHECKPOINT_DIR)
    else:
        print("No checkpoint found. Starting fresh training.")
        trainer.train()

    #Save trained model & tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Training complete. Model saved at: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
