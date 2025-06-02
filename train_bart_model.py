import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import load_dataset, Dataset as HFDataset # Keep HFDataset import for potential future use or clarity
import os
import logging
import sys

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Configure logging for datasets library (optional, but good for debugging)
datasets_logger = logging.getLogger("datasets")
datasets_logger.setLevel(logging.DEBUG)

# --- 1. 数据准备 (GrammarCorrectionDataset) ---
class GrammarCorrectionDataset(Dataset):
    def __init__(self, tokenizer, hf_dataset, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.hf_dataset = hf_dataset

        logger.info("Preprocessing dataset: mapping column names and filtering empty entries...")
        processed_dataset = self.hf_dataset.map(
            self._preprocess_example,
            num_proc=os.cpu_count() // 2 or 1,
            load_from_cache_file=False # Keep False during debugging
        )

        initial_filtered_count = len(processed_dataset)
        logger.info(f"Initial samples after column mapping: {initial_filtered_count}")

        # Filter out examples where either 'sentence' or 'parsed_correction' is empty
        self.hf_dataset = processed_dataset.filter(
            lambda example: example.get('sentence') is not None and len(str(example['sentence']).strip()) > 0 and \
                            example.get('parsed_correction') is not None and len(str(example['parsed_correction']).strip()) > 0
        )
        logger.info(f"Dataset preprocessed. Remaining samples after filtering: {len(self.hf_dataset)}")


    def _preprocess_example(self, example):
        """
        Helper function to map dataset column names to internal 'sentence' and 'parsed_correction'.
        This is for 'agentlans/grammar-correction' which uses 'input' and 'output'.
        """
        example['sentence'] = example.get('input', '')  # 原始文本列
        example['parsed_correction'] = example.get('output', '') # 纠正后文本列
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


def train_grammar_model(hf_dataset_name="agentlans/grammar-correction", model_name="facebook/bart-base", output_dir="./finetuned_bart_grammar_model"):
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    logger.info(f"Loading dataset from Hugging Face Hub: {hf_dataset_name}")
    try:
        ds = load_dataset(hf_dataset_name)
        logger.info(f"Dataset {hf_dataset_name} loaded successfully from the Hub.")
    except Exception as e:
        logger.error(f"Failed to load dataset {hf_dataset_name} from the Hub. Please verify the dataset name and your internet connection.")
        logger.error(f"Error details: {e}")
        # 保留这个提示，以防万一
        if "Invalid pattern: '**' can only be an entire path component" in str(e):
             logger.warning("This error often indicates an issue with the dataset's file structure or fsspec. Trying an alternative dataset loader or upgrading libraries might help.")
             logger.warning("Please ensure your 'datasets' library is updated to the latest version.")
        raise # Re-raise the exception to stop execution


    # Access the specific splits
    if 'train' not in ds:
        logger.error(f"Dataset {hf_dataset_name} does not contain a 'train' split.")
        return
    if 'validation' not in ds:
        logger.error(f"Dataset {hf_dataset_name} does not contain a 'validation' split.")
        return

    hf_train_dataset_raw = ds['train']
    hf_val_dataset_raw = ds['validation']

    logger.info(f"Original train dataset size from Hub: {len(hf_train_dataset_raw)}")
    logger.info(f"Original validation dataset size from Hub: {len(hf_val_dataset_raw)}")

    # Initialize GrammarCorrectionDataset for both train and validation splits
    full_train_dataset = GrammarCorrectionDataset(tokenizer, hf_train_dataset_raw)
    full_val_dataset = GrammarCorrectionDataset(tokenizer, hf_val_dataset_raw)

    logger.info(f"Final training dataset size after all preprocessing and filtering: {len(full_train_dataset)}")
    logger.info(f"Final validation dataset size after all preprocessing and filtering: {len(full_val_dataset)}")


    if len(full_train_dataset) == 0:
        logger.error("Training dataset is empty after preprocessing and filtering. Cannot train model. Please check your data or preprocessing logic.")
        return
    if len(full_val_dataset) == 0:
        logger.error("Validation dataset is empty after preprocessing and filtering. Cannot train model. Please check your data or preprocessing logic.")
        return

    # Use the preprocessed datasets directly
    train_dataset = full_train_dataset
    val_dataset = full_val_dataset

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
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
    HF_DATASET_NAME = "agentlans/grammar-correction"
    MODEL_TO_FINETUNE = "facebook/bart-base"
    FINE_TUNED_MODEL_DIR = "./finetuned_bart_model_agentlans_local" # 本地默认目录，Colab需修改

    train_grammar_model(
        hf_dataset_name=HF_DATASET_NAME,
        model_name=MODEL_TO_FINETUNE,
        output_dir=FINE_TUNED_MODEL_DIR
    )
# AUC, F1 scores,  and other metrics can be added to the training loop for model evaluation.
    logger.info(f"\n训练过程完成。你的微调模型保存在: {FINE_TUNED_MODEL_DIR}")
    logger.info("现在，更新你的 model.py 以便加载这个新模型进行推理。")