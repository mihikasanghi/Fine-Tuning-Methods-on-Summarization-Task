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
class TrainingConfiguration:
    """Configuration for the training process"""
    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_sequence_length: int = 1024
    num_workers: int = 6
    model_name: str = "gpt2"
    checkpoint_dir: str = '/checkpoints/fine_tuning/'
    test_mode: bool = True
    test_checkpoint_path: str = '/checkpoints/fine_tuning/best_model.pt'
    dataset_sizes: Dict[str, int] = field(default_factory=lambda: {
        'train': 21000,
        'validation': 6000,
        'test': 3000
    })

class SummarizationDataset(Dataset):
    """Dataset handler for text summarization"""
    def __init__(
        self,
        documents: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int
    ):
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._configure_tokenizer()

    def _configure_tokenizer(self):
        """Set up tokenizer special tokens"""
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.documents)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get a single training instance"""
        document = self.documents.iloc[index]
        source_text, target_text = document['article'], document['highlights']

        source_tokens = self._tokenize_text(source_text)
        target_tokens = self._tokenize_text(target_text, needs_attention_mask=False)

        return {
            "source_ids": source_tokens["input_ids"][0],
            "source_mask": source_tokens["attention_mask"][0],
            "target_ids": target_tokens["input_ids"][0]
        }

    def _tokenize_text(
        self,
        text: str,
        needs_attention_mask: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Tokenize input text with specified parameters"""
        return self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=needs_attention_mask,
            add_special_tokens=True
        )

class ModelHandler:
    """Handles model initialization and modification"""
    def __init__(self, config: TrainingConfiguration):
        self.config = config
        self.device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    def initialize_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Initialize and prepare the model for training"""
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        self._freeze_model_layers(model)
        self._unfreeze_lm_head(model)
        
        return model, tokenizer

    def _freeze_model_layers(self, model: AutoModelForCausalLM):
        """Freeze all model parameters"""
        for param in model.parameters():
            param.requires_grad = False

    def _unfreeze_lm_head(self, model: AutoModelForCausalLM):
        """Unfreeze only the language modeling head"""
        for param in model.lm_head.parameters():
            param.requires_grad = True

class MetricsTracker:
    """Tracks and computes training metrics"""
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.rouge_metric = evaluate.load('rouge')

    def compute_metrics(
        self,
        predictions: torch.Tensor,
        references: torch.Tensor
    ) -> Dict[str, float]:
        """Compute ROUGE scores for generated summaries"""
        decoded_preds = self.tokenizer.batch_decode(
            predictions.argmax(-1),
            skip_special_tokens=True
        )
        decoded_refs = self.tokenizer.batch_decode(
            references,
            skip_special_tokens=True
        )
        return self.rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_refs
        )

class CheckpointManager:
    """Manages model checkpointing"""
    def __init__(self, config: TrainingConfiguration):
        self.config = config
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        state: Dict,
        is_best: bool
    ):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, 'checkpoint.pt')
        best_model_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
        
        torch.save(state, checkpoint_path)
        if is_best:
            shutil.copyfile(checkpoint_path, best_model_path)

    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint"""
        return torch.load(path)

class TrainingEngine:
    """Manages the training process"""
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: TrainingConfiguration
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])]")
        self.metrics_tracker = MetricsTracker(tokenizer)
        self.checkpoint_manager = CheckpointManager(config)
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        """Execute training loop"""
        model = self._prepare_distributed_model()
        optimizer = self._create_optimizer(model)
        criterion = self._setup_criterion()
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            train_metrics = self._train_epoch(
                model, optimizer, criterion, train_loader, epoch
            )
            val_metrics = self._validate(model, criterion, val_loader)
            
            if int(os.environ["LOCAL_RANK"]) == 0:
                self._handle_checkpoint(
                    model, optimizer, epoch, val_metrics["loss"], best_val_loss
                )
                best_val_loss = min(val_metrics["loss"], best_val_loss)
                self._log_metrics(epoch, train_metrics, val_metrics)

    def test(
        self,
        test_loader: DataLoader,
        checkpoint_path: str
    ):
        """Execute testing loop"""
        model = self._load_for_testing(checkpoint_path)
        test_metrics = self._validate(model, self._setup_criterion(), test_loader)
        self._log_test_metrics(test_metrics)

class DistributedTrainingManager:
    """Manages distributed training setup and execution"""
    def __init__(self, config: TrainingConfiguration):
        self.config = config
        self._setup_distributed()
        
    def run(self):
        """Execute the training or testing pipeline"""
        model_handler = ModelHandler(self.config)
        model, tokenizer = model_handler.initialize_model()
        
        datasets = self._load_datasets()
        dataloaders = self._create_dataloaders(datasets, tokenizer)
        
        engine = TrainingEngine(model, tokenizer, self.config)
        
        if self.config.test_mode:
            engine.test(dataloaders['test'], self.config.test_checkpoint_path)
        else:
            engine.train(dataloaders['train'], dataloaders['validation'])
            
        dist.destroy_process_group()

    def _setup_distributed(self):
        """Initialize distributed training environment"""
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    def _load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load and prepare datasets"""
        return {
            split: pd.read_csv(f"cnn_dailymail/{split}.csv")[:size]
            for split, size in self.config.dataset_sizes.items()
        }

    def _create_dataloaders(
        self,
        datasets: Dict[str, pd.DataFrame],
        tokenizer: AutoTokenizer
    ) -> Dict[str, DataLoader]:
        """Create DataLoader instances for all splits"""
        return {
            split: DataLoader(
                SummarizationDataset(
                    data,
                    tokenizer,
                    self.config.max_sequence_length
                ),
                batch_size=self.config.batch_size,
                sampler=DistributedSampler(data),
                num_workers=self.config.num_workers,
                pin_memory=True
            )
            for split, data in datasets.items()
        }

if __name__ == "__main__":
    config = TrainingConfiguration()
    manager = DistributedTrainingManager(config)
    manager.run()
