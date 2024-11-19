import os
import shutil
import torch
import pandas as pd
import evaluate
import torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass
from typing import Dict, List, Tuple
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

@dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_sequence_length: int = 1024
    checkpoint_dir: str = '/checkpoints/summarization/'
    num_workers: int = 6
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["c_attn", "c_proj"]

class SummarizationDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, config: TrainingConfig):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self._setup_tokenizer()

    def _setup_tokenizer(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[index]
        
        source_encoding = self._encode_text(row['article'])
        target_encoding = self._encode_text(row['highlights'], return_attention_mask=False)
        
        return {
            "source_ids": source_encoding.input_ids[0],
            "source_mask": source_encoding.attention_mask[0],
            "target_ids": target_encoding.input_ids[0]
        }

    def _encode_text(self, text: str, return_attention_mask: bool = True) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_sequence_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=return_attention_mask,
            add_special_tokens=True
        )

class ModelManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        self.rouge_metric = evaluate.load('rouge')

    def initialize_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        base_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)
        
        self._freeze_base_model(base_model)
        lora_model = self._apply_lora_adaptation(base_model)
        
        return lora_model, tokenizer

    def _freeze_base_model(self, model: AutoModelForCausalLM):
        for param in model.parameters():
            param.requires_grad = False

    def _apply_lora_adaptation(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules
        )
        return get_peft_model(model, lora_config)

class SummarizationTrainer:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.rouge_metric = evaluate.load('rouge')
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        self.model = DDP(self.model, device_ids=[int(os.environ["LOCAL_RANK"])])
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            train_metrics = self._train_epoch(train_loader, optimizer, epoch)
            val_metrics = self._validate(val_loader)
            
            if int(os.environ["LOCAL_RANK"]) == 0:
                self._log_metrics(epoch, train_metrics, val_metrics)
                self._save_checkpoint(val_metrics['loss'], best_val_loss, optimizer)
                best_val_loss = min(val_metrics['loss'], best_val_loss)

    def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int) -> Dict:
        self.model.train()
        train_loader.sampler.set_epoch(epoch)
        
        accumulated_loss = 0
        predictions, references = [], []
        
        progress_bar = self._create_progress_bar(train_loader, epoch)
        
        for step, batch in progress_bar:
            loss, batch_preds, batch_refs = self._process_batch(batch)
            
            scaled_loss = loss / self.config.gradient_accumulation_steps
            scaled_loss.backward()
            
            accumulated_loss += loss.item()
            predictions.extend(batch_preds)
            references.extend(batch_refs)
            
            if self._should_optimize(step):
                self._optimize_step(optimizer)
                
        metrics = self._compute_metrics(accumulated_loss, predictions, references, len(train_loader))
        return metrics

    def _validate(self, val_loader: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0
        predictions, references = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                loss, batch_preds, batch_refs = self._process_batch(batch)
                total_loss += loss.item()
                predictions.extend(batch_preds)
                references.extend(batch_refs)
        
        metrics = self._compute_metrics(total_loss, predictions, references, len(val_loader))
        return metrics

    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[str], List[str]]:
        source_ids = batch["source_ids"].to(self.device)
        source_mask = batch["source_mask"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        
        outputs = self.model(input_ids=source_ids, attention_mask=source_mask)
        loss = self.criterion(outputs.logits.view(-1, outputs.logits.size(-1)), target_ids.view(-1))
        
        predictions = self.tokenizer.batch_decode(outputs.logits.argmax(-1), skip_special_tokens=True)
        references = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)
        
        return loss, predictions, references

class SummarizationSystem:
    def __init__(self, config: TrainingConfig):
        self.config = config
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
    def run(self):
        data = self._load_datasets()
        model_manager = ModelManager(self.config)
        model, tokenizer = model_manager.initialize_model()
        
        dataloaders = self._create_dataloaders(data, tokenizer)
        trainer = SummarizationTrainer(model, tokenizer, self.config)
        
        trainer.train(dataloaders['train'], dataloaders['val'])
        dist.destroy_process_group()

    def _load_datasets(self) -> Dict[str, pd.DataFrame]:
        return {
            'train': pd.read_csv("cnn_dailymail/train.csv")[:21000],
            'val': pd.read_csv("cnn_dailymail/validation.csv")[:6000],
            'test': pd.read_csv("cnn_dailymail/test.csv")[:3000]
        }

    def _create_dataloaders(
        self, 
        data: Dict[str, pd.DataFrame], 
        tokenizer: AutoTokenizer
    ) -> Dict[str, DataLoader]:
        datasets = {
            split: SummarizationDataset(df, tokenizer, self.config)
            for split, df in data.items()
        }
        
        return {
            split: DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                sampler=DistributedSampler(dataset),
                num_workers=self.config.num_workers,
                pin_memory=True
            )
            for split, dataset in datasets.items()
        }

if __name__ == "__main__":
    config = TrainingConfig()
    system = SummarizationSystem(config)
    system.run()
