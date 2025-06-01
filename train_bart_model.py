# train_bart_model.py

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import load_dataset
import os
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 1. 数据准备 (GrammarCorrectionDataset) ---
class GrammarCorrectionDataset(Dataset):
    def __init__(self, tokenizer, hf_dataset, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.hf_dataset = hf_dataset

        logger.info("Preprocessing dataset: selecting first correction and filtering empty entries...")
        processed_dataset = self.hf_dataset.map(
            self._preprocess_example,
            num_proc=os.cpu_count() // 2 or 1,
            load_from_cache_file=False  # Keep False during debugging
        )

        initial_filtered_count = len(processed_dataset)
        logger.info(f"Initial samples after selecting first correction: {initial_filtered_count}")

        self.hf_dataset = processed_dataset.filter(
            lambda example: example.get('sentence') is not None and len(str(example['sentence']).strip()) > 0 and \
                            example.get('parsed_correction') is not None and len(
                str(example['parsed_correction']).strip()) > 0
        )
        logger.info(f"Dataset preprocessed. Remaining samples after filtering: {len(self.hf_dataset)}")

    def _preprocess_example(self, example):
        """Helper function to extract the first correction from the list."""
        parsed_correction = ""
        # The 'corrections' column is already a list of strings
        corrections_list = example.get('corrections', [])  # Default to empty list if column is missing

        if corrections_list and isinstance(corrections_list, list) and len(corrections_list) > 0:
            parsed_correction = str(corrections_list[0])  # Take the first correction
        else:
            logger.debug(f"Corrections list was empty or not a list for example: {example.get('sentence', '')[:50]}...")

        example['parsed_correction'] = parsed_correction
        return example

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        original_text = str(self.hf_dataset[idx]['sentence'])
        corrected_text = str(self.hf_dataset[idx]['parsed_correction'])

        input_text = original_text

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = self.tokenizer(
            corrected_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": labels["input_ids"].flatten()
        }


def train_grammar_model(hf_dataset_path="Prajapat/Grammer_Correction_train", model_name="facebook/bart-base",
                        output_dir="./finetuned_bart_grammar_model"):
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    logger.info(f"Loading dataset from Hugging Face Hub: {hf_dataset_path}")
    hf_dataset = load_dataset(hf_dataset_path, split='train')

    logger.info(f"Original dataset size from Hugging Face Hub: {len(hf_dataset)}")

    full_dataset = GrammarCorrectionDataset(tokenizer, hf_dataset)

    logger.info(f"Final dataset size after all preprocessing and filtering: {len(full_dataset)}")

    if len(full_dataset) == 0:
        logger.error(
            "Dataset is empty after preprocessing and filtering. Cannot train model. Please check your data or preprocessing logic.")
        return

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    if train_size == 0 and len(full_dataset) > 0:
        train_size = 1
        val_size = len(full_dataset) - 1
    if val_size == 0 and len(full_dataset) > 0 and train_size == 0:
        val_size = 1
        train_size = len(full_dataset) - 1
    if train_size < 0 or val_size < 0:
        logger.error("Error: Negative dataset split size. Something is wrong with dataset splitting.")
        return

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        dataloader_num_workers=os.cpu_count() // 2 or 1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")

    logger.info(f"Saving final model and tokenizer to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model and tokenizer saved.")


if __name__ == "__main__":
    HF_DATASET_NAME = "Prajapat/Grammer_Correction_train"
    MODEL_TO_FINETUNE = "facebook/bart-base"
    FINE_TUNED_MODEL_DIR = f"./finetuned_{MODEL_TO_FINETUNE.replace('/', '_')}_grammar_model"

    train_grammar_model(
        hf_dataset_path=HF_DATASET_NAME,
        model_name=MODEL_TO_FINETUNE,
        output_dir=FINE_TUNED_MODEL_DIR
    )

    logger.info(f"\n训练过程完成。你的微调模型保存在: {FINE_TUNED_MODEL_DIR}")
    logger.info("现在，更新你的 model.py 以便加载这个新模型进行推理。")