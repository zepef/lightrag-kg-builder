"""
LoRA Fine-Tuning Trainer

Trains a LoRA adapter on fine-tuning pairs generated from the Knowledge Graph.
Uses Unsloth for efficient training on consumer GPUs.

Adapted from backend/training/finetune_lora.py for the kg-builder pipeline.
"""

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

# Disable wandb before any ML imports
os.environ["WANDB_DISABLED"] = "true"

from unsloth import FastLanguageModel

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from ..utils.config import TrainingConfig


class LoRATrainer:
    """
    LoRA fine-tuning trainer for Mistral models.

    Pipeline: load_model → load_dataset → format_dataset → train → save
    """

    def __init__(self, config: TrainingConfig, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model with Unsloth optimizations and add LoRA adapters."""
        print(f"  Loading model: {self.config.model}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model,
            max_seq_length=self.config.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=self.config.load_in_4bit,
        )

        print(f"    Device: {model.device}, Dtype: {model.dtype}")

        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        print("    LoRA adapters added")

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def load_dataset(self, path: Path) -> Dataset:
        """Load OpenAI JSONL fine-tuning pairs into a HuggingFace Dataset."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        print(f"    Loaded {len(data)} training examples from {path.name}")

        dataset = Dataset.from_list(data)
        return dataset

    def format_dataset(self, dataset: Dataset) -> Dataset:
        """Format dataset using the tokenizer's chat template."""
        tokenizer = self.tokenizer

        def apply_template(examples):
            texts = []
            for messages in examples["messages"]:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(text)
            return {"text": texts}

        formatted = dataset.map(apply_template, batched=True)
        print(f"    Formatted {len(formatted)} examples with chat template")
        return formatted

    def train(self, dataset: Dataset) -> "SFTTrainer":
        """Run LoRA fine-tuning with SFTTrainer."""
        effective_batch = self.config.batch_size * self.config.gradient_accumulation_steps
        print(f"    Epochs: {self.config.epochs}, Batch: {self.config.batch_size} "
              f"(effective {effective_batch}), LR: {self.config.learning_rate}")

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=2,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            args=training_args,
            packing=False,
        )

        print("\n    Training started...")
        trainer.train()
        print("    Training complete")

        return trainer

    def save(self, formats: List[str]):
        """Save model in specified formats (lora, merged, gguf)."""
        for fmt in formats:
            if fmt == "lora":
                lora_dir = self.output_dir / "lora_adapters"
                self.model.save_pretrained(str(lora_dir))
                self.tokenizer.save_pretrained(str(lora_dir))
                print(f"    Saved LoRA adapters: {lora_dir}")

            elif fmt == "merged":
                merged_dir = self.output_dir / "merged_model"
                self.model.save_pretrained_merged(
                    str(merged_dir),
                    self.tokenizer,
                    save_method="merged_16bit",
                )
                print(f"    Saved merged model: {merged_dir}")

            elif fmt == "gguf":
                gguf_dir = self.output_dir / "gguf"
                self.model.save_pretrained_gguf(
                    str(gguf_dir),
                    self.tokenizer,
                    quantization_method="q4_k_m",
                )
                print(f"    Saved GGUF (Q4_K_M): {gguf_dir}")

            else:
                print(f"    Unknown save format: {fmt}")

    def run(self, dataset_path: Path) -> dict:
        """Full training pipeline: load → format → train → save → report."""
        start = time.perf_counter()

        print("\n  Phase 1: Loading model...")
        self.load_model()

        print("\n  Phase 2: Loading dataset...")
        dataset = self.load_dataset(dataset_path)

        print("\n  Phase 3: Formatting dataset...")
        formatted = self.format_dataset(dataset)

        print("\n  Phase 4: Training...")
        trainer = self.train(formatted)

        print(f"\n  Phase 5: Saving ({', '.join(self.config.save_formats)})...")
        self.save(self.config.save_formats)

        duration_s = time.perf_counter() - start

        # Extract loss from training history
        loss_history = []
        if hasattr(trainer, 'state') and trainer.state.log_history:
            loss_history = [
                {"step": entry.get("step"), "loss": entry.get("loss")}
                for entry in trainer.state.log_history
                if "loss" in entry
            ]

        # Build training report
        report = {
            "model": self.config.model,
            "dataset": str(dataset_path),
            "dataset_size": len(dataset),
            "duration_s": round(duration_s, 1),
            "duration_min": round(duration_s / 60, 2),
            "config": asdict(self.config),
            "save_formats": self.config.save_formats,
            "output_dir": str(self.output_dir),
            "final_loss": loss_history[-1]["loss"] if loss_history else None,
            "loss_history": loss_history,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }

        report_path = self.output_dir / "training_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n    Report saved: {report_path}")

        return report
