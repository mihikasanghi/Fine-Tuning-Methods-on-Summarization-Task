# Advanced NLP Assignment 3
### Mihika Sanghi 2021113014
## Comparing Fine-tuning Methods for GPT-2 on Summarization Tasks

This repository implements and compares three different fine-tuning methods for the GPT-2 small model on text summarization:
1. Prompt Tuning
2. LoRA (Low-Rank Adaptation)
3. Traditional Fine-tuning (Last Layers)

### Setup Requirements

Required packages:
- transformers
- torch
- datasets
- rouge_score
- peft (for LoRA implementation)
- tqdm
- pandas
- numpy

### Dataset

The project uses the CNN/Daily Mail dataset for text summarization. Due to computational constraints, we use a subset of the data:
- Training: 21,000 samples
- Validation: 3,000 samples
- Testing: 6,000 samples

### Directory Structure

```
assignment3/
├── src/
│   ├── prompt_tuning.py
│   ├── lora_tuning.py
│   ├── traditional_tuning.py
├── report.pdf
└── README.md
```

### Running the Code

Train the models:
```bash
torchrun --nproc_per_node=num_gpus src/prompt_tuning.py
torchrun --nproc_per_node=num_gpus src/traditional_tuning.py
torchrun --nproc_per_node=num_gpus src/lora_tuning.py
```
### Hyperparameters

- Learning rate: 1e-4
- Loss function: Cross Entropy
- Number of epochs: 10
- Optimizer: AdamW
- Gradient accumulation steps: 4
- Soft prompt length: 70 tokens
- LoRA rank: 8
- LoRA alpha: 32
- LoRA dropout: 0.1

### Results

#### Parameter Efficiency

| Method | Trainable Params | % of Total |
|--------|-----------------|------------|
| Prompt Tuning | 53,760 | 0.043% |
| LoRA | 811,008 | 0.648% |
| Traditional Fine-tuning | 38,597,376 | 31.02% |

#### Performance Metrics

| Metric | Prompt Tuning | LoRA | Traditional |
|--------|---------------|------|-------------|
| Train Loss | 1.070 | 1.820 | 1.085 |
| Val Loss | 1.066 | 1.798 | 1.079 |
| Test ROUGE-1 | 0.063 | 0.114 | 0.155 |
| Test ROUGE-2 | 0.007 | 0.009 | 0.010 |
| Test ROUGE-L | 0.050 | 0.095 | 0.110 |
| Train ROUGE-1 | 0.055 | 0.077 | 0.123 |
| Train ROUGE-2 | 0.006 | 0.007 | 0.015 |
| Train ROUGE-L | 0.042 | 0.065 | 0.075 |
| Val ROUGE-1 | 0.075 | 0.125 | 0.183 |
| Val ROUGE-2 | 0.009 | 0.006 | 0.025 |
| Val ROUGE-L | 0.055 | 0.100 | 0.110 |

#### Training Time & Resources

1. **Prompt Tuning**
   - Hardware: 2x RTX 3090 (24GB VRAM)
   - Time per epoch: 8m 30s
   - VRAM usage: 16.5GB/GPU
   - Batch size: 8 (effective: 64)

2. **LoRA**
   - Hardware: 4x RTX 2080 (12GB VRAM)
   - Time per epoch: 9m 40s
   - VRAM usage: 10.2GB/GPU
   - Batch size: 4 (effective: 64)

3. **Traditional Fine-tuning**
   - Hardware: 2x RTX 3090 (24GB VRAM)
   - Time per epoch: 11m 50s
   - VRAM usage: 16.5GB/GPU
   - Batch size: 8 (effective: 64)

### Key Findings

1. Traditional fine-tuning achieves the best performance across all ROUGE metrics, likely due to the larger number of trainable parameters (31% of model).

2. LoRA provides a good balance between efficiency and performance, achieving reasonable results while training only 0.648% of parameters.

3. Prompt tuning, while most parameter-efficient (0.043%), shows lower performance but could be viable for resource-constrained scenarios.

### Notes

- All models use gradient accumulation to achieve an effective batch size of 64
- Models are evaluated using ROUGE-1, ROUGE-2, and ROUGE-L metrics
- Early stopping is implemented based on validation loss
- Checkpoints are saved for best performing models (https://iiitaphyd-my.sharepoint.com/:f:/g/personal/mihika_sanghi_research_iiit_ac_in/EqoU99ECp4RLkywnMlb8TpcBP95SuUoKV0FNbGk7jTbu2Q?e=mIQPhT) 
