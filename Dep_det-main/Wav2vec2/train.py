import os
from torch.cuda.amp import GradScaler
import preprocess_data
import build_model
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import load_dataset
from typing import Any, Dict, Union
import torch
from torch import nn
from collections import Counter
from torch.nn.utils import clip_grad_norm_

configuration = {
    "seed": 42,
    "corpora": "daic_woz",
    "data_path": "/workspace/Depression_Recognition-Code/Preprocessing_code/audio_combine/",
    "processor_name_or_path": "facebook/wav2vec2-base",
    "pooling_mode": "mean",
    "return_attention_mask": False,
    "freeze_feature_extractor": True,
    "output_dir": "/workspace/Depression/models/daic-woz/d2-c2-rmse-roc-mean10-frozen",
    "num_train_epochs": 10,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "eval_steps": 30,
    "save_steps": 30,
    "logging_steps": 30,
    "learning_rate": 3e-5,
    "save_total_limit": 2,
    "fp16": False,
    "bf16": True,
    "cache_dir": "/workspace/Depression/content/cache/",
    "checkpoint": "/workspace/Depression/models/daic-woz/d2-c2-rmse-roc-mean10-frozen/checkpoint-3420",
    "resume_from_checkpoint": True,
    "test_corpora": None,
    "test_corpora_path": None,
    "report_to": None,
}

set_seed(configuration['seed'])

#Prepare Data Paths
train_filepath = os.path.join(configuration['output_dir'], "splits/train.csv")
test_filepath = os.path.join(configuration['output_dir'], "splits/test.csv")
valid_filepath = os.path.join(configuration['output_dir'], "splits/valid.csv")

if not os.path.exists(train_filepath) or not os.path.exists(test_filepath) or not os.path.exists(valid_filepath):
    import prepare_data
    df = prepare_data.df(configuration['corpora'], configuration['data_path'])
    prepare_data.prepare_splits(df, configuration)

#Load Preprocessed Data
train_dataset, eval_dataset, input_column, output_column, label_list, num_labels = preprocess_data.training_data(configuration)
config, processor, target_sampling_rate = preprocess_data.load_processor(configuration, label_list, num_labels)
train_dataset, eval_dataset = preprocess_data.preprocess_data(
    configuration, processor, target_sampling_rate, train_dataset, eval_dataset, input_column, output_column, label_list
)

#Define Data Collator
data_collator = build_model.data_collator(processor)

#Compute Class Weights
print("Computing class weights...")
train_labels = [example[output_column] for example in train_dataset if example[output_column] is not None]
class_counts = Counter(train_labels)
total_samples = sum(class_counts.values())
num_classes = len(class_counts)

# Normalise class weights
class_weights = torch.tensor([
    total_samples / (num_classes * class_counts.get(cls, 1)) for cls in range(num_classes)
], dtype=torch.float32).to("cuda")

print("Updated Class Weights:", class_weights)

# Load Model with Class Weights
model = build_model.load_pretrained_checkpoint(config, configuration['processor_name_or_path'], class_weights=class_weights)

if configuration['freeze_feature_extractor']:
    model.freeze_feature_extractor()
    print("Feature extractor frozen")

#Define Training Arguments
training_args = TrainingArguments(
    output_dir=configuration['output_dir'],
    per_device_train_batch_size=configuration['per_device_train_batch_size'],
    per_device_eval_batch_size=configuration['per_device_eval_batch_size'],
    gradient_accumulation_steps=configuration['gradient_accumulation_steps'],
    evaluation_strategy="steps",
    num_train_epochs=configuration['num_train_epochs'],
    bf16=configuration['bf16'],
    save_steps=configuration['save_steps'],
    eval_steps=configuration['eval_steps'],
    logging_steps=configuration['logging_steps'],
    learning_rate=float(configuration['learning_rate']),
    save_total_limit=configuration['save_total_limit'],
    seed=configuration['seed'],
    data_seed=configuration['seed'],
    report_to=configuration.get('report_to', None),
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,
)

# Custom Trainer Class with AMP and Label Smoothing
class CTCTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = GradScaler() if self.args.fp16 else None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(self.model.device)
        outputs = model(**inputs)
        logits = outputs.get("logits", None)
        if logits is None:
            raise ValueError("Expected 'logits' key in model output but did not find it.")
        class_weights_on_device = class_weights.to(self.model.device)
        loss = nn.functional.cross_entropy(logits, labels.long(), weight=class_weights_on_device, label_smoothing=0.1)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: int) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16 if self.args.bf16 else torch.float16):
            loss = self.compute_loss(model, inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        if self.args.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
        return loss.detach()

# Initialise Trainer
trainer = CTCTrainer(
    model=model,``
    args=training_args,
    data_collator=data_collator,
    compute_metrics=build_model.compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
)

# Train Model
print("Starting training...")
try:
    trainer.train(resume_from_checkpoint=configuration.get('resume_from_checkpoint', False))
except RuntimeError as e:
    if "out of memory" in str(e):
        print("Out of memory error")
        torch.cuda.empty_cache()
        training_args.per_device_train_batch_size = max(1, configuration['per_device_train_batch_size'] // 2)
        trainer.args = training_args
        trainer.train(resume_from_checkpoint=False)
    else:
        raise e

# Debugging Test Dataset
if os.path.exists(test_filepath):
    test_dataset = load_dataset("csv", data_files={"test": test_filepath}, delimiter="\t", cache_dir=configuration['cache_dir'])["test"]
    print("Inspecting a batch from test_dataset:")
    for i in range(min(3, len(test_dataset))):
        print(f"Test Example {i}: {test_dataset[i]}")
else:
    print(f"Test dataset not found at {test_filepath}")

# Evaluate Model
print("Starting evaluation...")
result = trainer.evaluate(eval_dataset)
print("Evaluation Results:", result)
