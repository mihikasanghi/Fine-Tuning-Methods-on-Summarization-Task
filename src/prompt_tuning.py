import os
import shutil
import torch
import pandas as pd
import evaluate
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class SummarizationConfig:
    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_workers: int = 6
    max_input_length: int = 954
    max_target_length: int = 1024
    checkpoint_dir: str = '/checkpoints/prompt_tuning/'
    soft_prompt_length: int = 70
    model_name: str = "gpt2"
    testing_mode: bool = True
    test_checkpoint_path: str = '/checkpoints/prompt_tuning/best_model.pt'
    dataset_limits: Dict[str, int] = field(default_factory=lambda: {
        'train': 21000,
        'val': 6000,
        'test': 3000
    })

class PromptTunedModel(nn.Module):
    """Enhanced prompt tuning implementation for language models"""
    def __init__(self, base_model: AutoModelForCausalLM, config: SummarizationConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self._freeze_base_model()
        self.soft_prompts = self._initialize_soft_prompts()
        
    def _freeze_base_model(self):
        """Freeze all parameters of the base model"""
        for param in self.base_model.parameters():
            param.requires_grad = False
            
    def _initialize_soft_prompts(self) -> nn.Embedding:
        """Initialize learnable soft prompt embeddings"""
        return nn.Embedding(
            self.config.soft_prompt_length,
            self.base_model.config.hidden_size
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with soft prompt prepending"""
        batch_size = input_ids.size(0)
        
        # Get input embeddings from base model
        input_embeds = self.base_model.transformer.wte(input_ids)
        
        # Generate and prepend soft prompts
        soft_prompt_ids = torch.arange(
            self.config.soft_prompt_length,
            device=input_ids.device
        ).expand(batch_size, -1)
        soft_prompt_embeds = self.soft_prompts(soft_prompt_ids)
        
        # Combine embeddings and adjust attention mask
        combined_embeds = torch.cat((soft_prompt_embeds, input_embeds), dim=1)
        prompt_attention_mask = torch.ones(
            (batch_size, self.config.soft_prompt_length),
            device=attention_mask.device
        )
        combined_attention_mask = torch.cat(
            (prompt_attention_mask, attention_mask),
            dim=1
        )
        
        return self.base_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask
        )

class SummarizationDataset(Dataset):
    """Dataset for text summarization with prompt tuning"""
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, config: SummarizationConfig):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self._setup_tokenizer()

    def _setup_tokenizer(self):
        """Configure tokenizer settings"""
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Process and return a single training example"""
        article, summary = self._get_text_pair(idx)
        
        article_encoding = self._encode_text(
            article,
            max_length=self.config.max_input_length,
            with_attention_mask=True
        )
        
        summary_encoding = self._encode_text(
            summary,
            max_length=self.config.max_target_length,
            with_attention_mask=False
        )

        return {
            "input_ids": article_encoding["input_ids"][0],
            "attention_mask": article_encoding["attention_mask"][0],
            "target_ids": summary_encoding["input_ids"][0]
        }

    def _get_text_pair(self, idx: int) -> Tuple[str, str]:
        """Retrieve article and summary pair"""
        row = self.data.iloc[idx]
        return row['article'], row['highlights']

    def _encode_text(
        self,
        text: str,
        max_length: int,
        with_attention_mask: bool
    ) -> Dict[str, torch.Tensor]:
        """Encode text with specified parameters"""
        return self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=with_attention_mask,
            add_special_tokens=True
        )

class TrainingManager:
    """Manages the training process for prompt-tuned summarization"""
    def __init__(self, model: PromptTunedModel, config: SummarizationConfig):
        self.model = model
        self.config = config
        self.device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        self.criterion = nn.CrossEntropyLoss(ignore_index=model.base_model.config.eos_token_id)
        self.rouge_metric = evaluate.load('rouge')
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: AutoTokenizer
    ):
        """Execute training loop"""
        model = self._prepare_distributed_model()
        optimizer = self._create_optimizer()
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            train_metrics = self._train_epoch(
                model, optimizer, train_loader, tokenizer, epoch
            )
            val_metrics = self._validate(model, val_loader, tokenizer)
            
            if int(os.environ["LOCAL_RANK"]) == 0:
                self._handle_checkpointing(
                    model, optimizer, epoch, val_metrics["loss"], best_val_loss
                )
                best_val_loss = min(val_metrics["loss"], best_val_loss)
                self._log_metrics(epoch, train_metrics, val_metrics)

    def test(
        self,
        test_loader: DataLoader,
        tokenizer: AutoTokenizer,
        checkpoint_path: str
    ):
        """Execute testing loop"""
        model = self._load_checkpoint(checkpoint_path)
        test_metrics = self._validate(model, test_loader, tokenizer)
        self._log_test_metrics(test_metrics)

class DistributedRunner:
    """Manages distributed training setup and execution"""
    def __init__(self, config: SummarizationConfig):
        self.config = config
        self._setup_distributed()
        
    def _setup_distributed(self):
        """Initialize distributed training environment"""
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
    def run(self):
        """Execute the training or testing pipeline"""
        datasets = self._load_datasets()
        model, tokenizer = self._initialize_model()
        dataloaders = self._create_dataloaders(datasets, tokenizer)
        
        training_manager = TrainingManager(model, self.config)
        
        if self.config.testing_mode:
            training_manager.test(
                dataloaders['test'],
                tokenizer,
                self.config.test_checkpoint_path
            )
        else:
            training_manager.train(
                dataloaders['train'],
                dataloaders['val'],
                tokenizer
            )
            
        dist.destroy_process_group()

    def _load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load and preprocess datasets"""
        return {
            split: pd.read_csv(f"cnn_dailymail/{split}.csv")[:limit]
            for split, limit in self.config.dataset_limits.items()
        }

    def _initialize_model(self) -> Tuple[PromptTunedModel, AutoTokenizer]:
        """Initialize model and tokenizer"""
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = PromptTunedModel(base_model, self.config).cuda()
        return model, tokenizer

    def _create_dataloaders(
        self,
        datasets: Dict[str, pd.DataFrame],
        tokenizer: AutoTokenizer
    ) -> Dict[str, DataLoader]:
        """Create DataLoader instances for all splits"""
        return {
            split: DataLoader(
                SummarizationDataset(data, tokenizer, self.config),
                batch_size=self.config.batch_size,
                sampler=DistributedSampler(data),
                num_workers=self.config.num_workers,
                pin_memory=True
            )
            for split, data in datasets.items()
        }

if __name__ == "__main__":
    config = SummarizationConfig()
    runner = DistributedRunner(config)
    runner.run()
