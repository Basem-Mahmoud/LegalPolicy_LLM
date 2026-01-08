"""
Fine-tuning script for legal policy explainer using LoRA/QLoRA.
Uses PEFT (Parameter-Efficient Fine-Tuning) for efficient adaptation.
"""

import os
import torch
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegalExplainerFineTuner:
    """
    Fine-tune a language model for legal policy explanation using LoRA.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize fine-tuner with configuration.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.ft_config = self.config["fine_tuning"]
        self.base_model_name = self.ft_config["base_model"]
        self.output_dir = Path(self.config["model"]["fine_tuned_model_path"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def prepare_model_and_tokenizer(self):
        """
        Prepare model and tokenizer with quantization if enabled.
        """
        logger.info(f"Loading model: {self.base_model_name}")

        # Configure quantization
        compute_dtype = torch.float16

        if self.ft_config["use_4bit"]:
            logger.info("Using 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif self.ft_config["use_8bit"]:
            logger.info("Using 8-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=compute_dtype,
            )
        else:
            bnb_config = None

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare model for k-bit training if quantized
        if bnb_config:
            model = prepare_model_for_kbit_training(model)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return model, tokenizer

    def setup_lora(self, model):
        """
        Set up LoRA configuration and apply to model.

        Args:
            model: Base model

        Returns:
            PEFT model with LoRA
        """
        logger.info("Setting up LoRA configuration")

        lora_config = LoraConfig(
            r=self.ft_config["lora_r"],
            lora_alpha=self.ft_config["lora_alpha"],
            target_modules=self.ft_config["target_modules"],
            lora_dropout=self.ft_config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model

    def load_training_data(self, data_path: str = "data/training/legal_qa.json") -> Dataset:
        """
        Load training dataset.

        Args:
            data_path: Path to training data

        Returns:
            HuggingFace Dataset
        """
        logger.info(f"Loading training data from {data_path}")

        if not Path(data_path).exists():
            logger.warning(f"Training data not found at {data_path}, creating sample dataset")
            return self._create_sample_dataset()

        # Load from JSON
        dataset = load_dataset("json", data_files=data_path, split="train")
        logger.info(f"Loaded {len(dataset)} training examples")

        return dataset

    def _create_sample_dataset(self) -> Dataset:
        """Create a sample dataset for demonstration."""
        sample_data = [
            {
                "instruction": "Explain what a non-disclosure agreement is.",
                "input": "",
                "output": "A Non-Disclosure Agreement (NDA) is a legal contract between parties that outlines confidential information they wish to share for specific purposes while restricting access by third parties. It ensures that sensitive information remains private and establishes legal consequences if confidentiality is breached."
            },
            {
                "instruction": "What does 'liability' mean in legal terms?",
                "input": "",
                "output": "In legal terms, 'liability' refers to legal responsibility or obligation. When someone is liable, they are legally responsible for something, typically for damages, costs, or consequences that result from their actions or failures. Liability can be contractual, tort-based, or statutory."
            },
            {
                "instruction": "Explain the concept of 'force majeure'.",
                "input": "",
                "output": "Force majeure is a contract clause that frees parties from liability or obligation when an extraordinary event or circumstance beyond their control prevents them from fulfilling the contract. These events typically include natural disasters, war, terrorism, or other 'acts of God' that make performance impossible."
            }
        ]

        return Dataset.from_list(sample_data)

    def format_prompt(self, example: Dict[str, str]) -> str:
        """
        Format training example into prompt.

        Args:
            example: Dictionary with instruction, input, output

        Returns:
            Formatted prompt string
        """
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        if input_text:
            prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""

        return prompt

    def train(self, dataset: Optional[Dataset] = None):
        """
        Train the model with LoRA.

        Args:
            dataset: Optional dataset, otherwise loads from config
        """
        # Prepare model and tokenizer
        model, tokenizer = self.prepare_model_and_tokenizer()

        # Setup LoRA
        model = self.setup_lora(model)

        # Load dataset
        if dataset is None:
            dataset = self.load_training_data()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.ft_config["num_epochs"],
            per_device_train_batch_size=self.ft_config["batch_size"],
            gradient_accumulation_steps=self.ft_config["gradient_accumulation_steps"],
            learning_rate=self.ft_config["learning_rate"],
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            optim="paged_adamw_8bit",
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            report_to="none",
        )

        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=self.ft_config["max_seq_length"],
            formatting_func=self.format_prompt,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(self.output_dir)

        logger.info("Training completed!")

    def load_fine_tuned_model(self):
        """
        Load the fine-tuned model for inference.

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading fine-tuned model from {self.output_dir}")

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Load PEFT adapters
        model = PeftModel.from_pretrained(model, str(self.output_dir))

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(self.output_dir))

        return model, tokenizer

    def compare_outputs(self, prompt: str):
        """
        Compare outputs from base and fine-tuned models.

        Args:
            prompt: Test prompt
        """
        # Base model
        logger.info("Generating with base model...")
        base_model, base_tokenizer = self.prepare_model_and_tokenizer()
        base_pipeline = pipeline(
            "text-generation",
            model=base_model,
            tokenizer=base_tokenizer,
            max_new_tokens=256
        )
        base_output = base_pipeline(prompt)[0]["generated_text"]

        # Fine-tuned model
        logger.info("Generating with fine-tuned model...")
        ft_model, ft_tokenizer = self.load_fine_tuned_model()
        ft_pipeline = pipeline(
            "text-generation",
            model=ft_model,
            tokenizer=ft_tokenizer,
            max_new_tokens=256
        )
        ft_output = ft_pipeline(prompt)[0]["generated_text"]

        print("\n" + "="*80)
        print("BASE MODEL OUTPUT:")
        print("="*80)
        print(base_output)
        print("\n" + "="*80)
        print("FINE-TUNED MODEL OUTPUT:")
        print("="*80)
        print(ft_output)
        print("="*80)


def main():
    """Main training function."""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize fine-tuner
    fine_tuner = LegalExplainerFineTuner()

    # Train model
    fine_tuner.train()

    # Test comparison
    test_prompt = "### Instruction:\nWhat is a breach of contract?\n\n### Response:\n"
    fine_tuner.compare_outputs(test_prompt)


if __name__ == "__main__":
    main()
