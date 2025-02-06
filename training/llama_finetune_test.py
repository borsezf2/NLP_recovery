import torch
from datasets import load_dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import numpy as np
from typing import Dict, List

# Sample data for fine-tuning
SAMPLE_DATA = [
    {"instruction": "Explain what photosynthesis is.", 
     "response": "Photosynthesis is the process by which plants convert sunlight into energy. They use this energy to turn carbon dioxide and water into glucose and oxygen."},
    {"instruction": "What is the capital of France?",
     "response": "Paris is the capital of France. It's known as the City of Light and is famous for landmarks like the Eiffel Tower."},
    # Add more examples as needed
]

def prepare_dataset():
    """Create a sample dataset for fine-tuning"""
    dataset_dict = {
        "instruction": [item["instruction"] for item in SAMPLE_DATA],
        "response": [item["response"] for item in SAMPLE_DATA]
    }
    return dataset_dict

def format_instruction(example: Dict) -> str:
    """Format the instruction and response into a single string"""
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}\n\n"

class LlamaFineTuner:
    def __init__(
        self,
        model_name: str = "huggingface/llama-3b",
        output_dir: str = "./llama-ft-output",
        max_length: int = 512
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        
        # Initialize tokenizer and model
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Add special tokens if needed
        special_tokens = {"pad_token": "[PAD]"}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def preprocess_function(self, examples: Dict) -> Dict:
        """Tokenize and format the examples"""
        formatted_texts = [format_instruction({"instruction": instr, "response": resp}) 
                         for instr, resp in zip(examples["instruction"], examples["response"])]
        
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    def train(
        self,
        train_dataset: Dict,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100
    ):
        """Fine-tune the model"""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=10,
            save_strategy="epoch",
            logging_dir="./logs",
            fp16=True,
            gradient_accumulation_steps=4,
        )

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        # Start training
        trainer.train()
        
        # Save the fine-tuned model
        self.model.save_pretrained(f"{self.output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{self.output_dir}/final_model")

def main():
    # Prepare dataset
    dataset_dict = prepare_dataset()
    
    # Initialize fine-tuner
    fine_tuner = LlamaFineTuner()
    
    # Preprocess dataset
    tokenized_dataset = fine_tuner.preprocess_function(dataset_dict)
    
    # Start fine-tuning
    fine_tuner.train(tokenized_dataset)

if __name__ == "__main__":
    main()