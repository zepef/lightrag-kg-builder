"""Tests for the LoRA trainer (logic only, no GPU required)."""

import json

import pytest

from src.utils.config import TrainingConfig


# ============================================================================
# TrainingConfig defaults
# ============================================================================

class TestTrainingConfig:

    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.model == "mistralai/Mistral-7B-Instruct-v0.3"
        assert cfg.max_seq_length == 4096
        assert cfg.load_in_4bit is True
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 16
        assert cfg.lora_dropout == 0.05
        assert cfg.epochs == 3
        assert cfg.batch_size == 2
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.learning_rate == 2e-4
        assert cfg.warmup_steps == 5
        assert "lora" in cfg.save_formats
        assert "gguf" in cfg.save_formats

    def test_custom_values(self):
        cfg = TrainingConfig(
            model="meta-llama/Llama-2-7b",
            epochs=5,
            batch_size=4,
            learning_rate=1e-5,
            lora_r=32,
        )
        assert cfg.model == "meta-llama/Llama-2-7b"
        assert cfg.epochs == 5
        assert cfg.batch_size == 4
        assert cfg.learning_rate == 1e-5
        assert cfg.lora_r == 32

    def test_effective_batch_size(self):
        cfg = TrainingConfig(batch_size=2, gradient_accumulation_steps=4)
        effective = cfg.batch_size * cfg.gradient_accumulation_steps
        assert effective == 8


# ============================================================================
# LoRATrainer - dataset loading (no GPU)
# ============================================================================

class TestTrainerDatasetLoading:

    def _write_dataset(self, path, pairs):
        with open(path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    def test_load_dataset_valid(self, tmp_path):
        # Import here to avoid top-level unsloth import
        try:
            from src.finetune.trainer import LoRATrainer
        except ImportError:
            pytest.skip("Training dependencies (unsloth/torch) not installed")

        dataset_path = tmp_path / "train.jsonl"
        pairs = [
            {"messages": [
                {"role": "system", "content": "Tu es un expert."},
                {"role": "user", "content": "Qu'est-ce que le capital ?"},
                {"role": "assistant", "content": "Le capital est..."},
            ]},
            {"messages": [
                {"role": "system", "content": "Tu es un expert."},
                {"role": "user", "content": "Qu'est-ce que la dette ?"},
                {"role": "assistant", "content": "La dette est..."},
            ]},
        ]
        self._write_dataset(dataset_path, pairs)

        trainer = LoRATrainer(TrainingConfig(), tmp_path / "output")
        dataset = trainer.load_dataset(dataset_path)
        assert len(dataset) == 2
        assert "messages" in dataset.column_names

    def test_load_dataset_file_not_found(self, tmp_path):
        try:
            from src.finetune.trainer import LoRATrainer
        except ImportError:
            pytest.skip("Training dependencies (unsloth/torch) not installed")

        trainer = LoRATrainer(TrainingConfig(), tmp_path / "output")
        with pytest.raises(FileNotFoundError):
            trainer.load_dataset(tmp_path / "nonexistent.jsonl")

    def test_load_dataset_empty_lines_skipped(self, tmp_path):
        try:
            from src.finetune.trainer import LoRATrainer
        except ImportError:
            pytest.skip("Training dependencies (unsloth/torch) not installed")

        dataset_path = tmp_path / "train.jsonl"
        content = (
            '{"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]}\n'
            '\n'
            '{"messages": [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]}\n'
            '\n'
        )
        dataset_path.write_text(content)

        trainer = LoRATrainer(TrainingConfig(), tmp_path / "output")
        dataset = trainer.load_dataset(dataset_path)
        assert len(dataset) == 2


# ============================================================================
# LoRATrainer - output directory management
# ============================================================================

class TestTrainerOutputDir:

    def test_output_dir_created(self, tmp_path):
        try:
            from src.finetune.trainer import LoRATrainer
        except ImportError:
            pytest.skip("Training dependencies (unsloth/torch) not installed")

        out = tmp_path / "new" / "nested" / "output"
        trainer = LoRATrainer(TrainingConfig(), out)
        assert out.exists()
        assert out.is_dir()

    def test_output_dir_already_exists(self, tmp_path):
        try:
            from src.finetune.trainer import LoRATrainer
        except ImportError:
            pytest.skip("Training dependencies (unsloth/torch) not installed")

        out = tmp_path / "existing"
        out.mkdir()
        # Should not raise
        trainer = LoRATrainer(TrainingConfig(), out)
        assert out.exists()
