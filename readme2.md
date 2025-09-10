# DREEAM-RU: Document-level Relation Extraction and Evidence Aggregation Model for Russian

An adaptation of the DREEAM (Document-level Relation Extraction and Evidence Aggregation Model) framework to the Russian language, with several enhancements and improvements.

## Overview

DREEAM-RU is a neural model for document-level relation extraction that simultaneously:
- **Extracts relations** between entities across an entire document
- **Identifies evidence sentences** that support the predicted relations
- **Handles long documents** through specialized attention mechanisms
- **Supports Russian language** using RuBERT as the backbone transformer

The model is specifically designed to work with documents where entities and their relations span multiple sentences, making it particularly suitable for complex document understanding tasks.

## Architecture

### Core Components

1. **Transformer Encoder**: Uses RuBERT (`DeepPavlov/rubert-base-cased`) for encoding document sequences
2. **Entity Representation**: Combines mention-level embeddings using log-sum-exp pooling
3. **Relation Classification**: 
   - Bilinear classifier with grouped block structure
   - Head/tail extractors with localized context pooling
4. **Evidence Extraction**: Sentence-level attention mechanism for identifying supporting evidence
5. **Long Sequence Processing**: Handles documents longer than 512 tokens through overlapping chunks

### Key Features

- **Multi-task Learning**: Joint training of relation extraction and evidence identification
- **Attention-based Evidence**: Automatically learns which sentences provide evidence for relations
- **Knowledge Distillation**: Support for self-training with teacher attention
- **Flexible Evaluation**: Single-pass and fusion-based evaluation modes

## Dataset Format

The model expects data in DocRED format with the following structure:

```json
{
  "title": "Document title",
  "sents": [["sentence", "tokens", "as", "lists"], ...],
  "vertexSet": [
    [{"name": "entity", "sent_id": 0, "pos": [0, 1], "type": "PERSON"}],
    ...
  ],
  "labels": [
    {"h": 0, "t": 1, "r": "relation_type", "evidence": [0, 1]}
  ]
}
```

## Configuration

The main configuration is stored in `dreeam-config.json`:

```json
{
  "data_dir": "path/to/data",
  "transformer_type": "bert",
  "model_name_or_path": "DeepPavlov/rubert-base-cased",
  "max_seq_length": 1600,
  "max_sent_num": 101,
  "num_labels": 2,
  "num_class": 48,
  "evi_thresh": 0.2
}
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd dreeam-ru
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

```bash
python run.py \
    --do_train \
    --data_dir ./dataset/your_data \
    --train_file train.json \
    --dev_file dev.json \
    --save_path ./logs/experiment_name \
    --num_train_epochs 30 \
    --train_batch_size 4 \
    --eval_mode single
```

### Inference

```bash
python run.py \
    --load_path ./logs/experiment_name \
    --test_file test.json \
    --pred_file results.json \
    --test_batch_size 8
```

### Batch Scripts

- **Windows**: Use `run_rel.bat` for training and `run_inf.bat` for inference
- **Linux/Mac**: Use `run_rel.sh` for training and `run_inf.sh` for inference

## Key Parameters

### Training Parameters
- `--lr_transformer`: Learning rate for transformer layers (default: 5e-5)
- `--lr_added`: Learning rate for added modules (default: 1e-4)  
- `--evi_lambda`: Weight for evidence loss (default: 0.1)
- `--attn_lambda`: Weight for attention distillation loss (default: 1.0)
- `--gradient_accumulation_steps`: Steps to accumulate gradients (default: 1)

### Model Parameters
- `--max_seq_length`: Maximum sequence length (default: 1024)
- `--max_sent_num`: Maximum sentences per document (default: 25)
- `--num_labels`: Maximum relation labels per entity pair (default: 4)
- `--evi_thresh`: Evidence threshold for sentence selection (default: 0.2)

## Scripts and Tools

### Data Processing Scripts (`scripts/`)

- `brat2docred.py`: Convert BRAT annotation format to DocRED format
- `brat2docred_split.py`: Convert BRAT format with document splitting
- `docred2brat.py`: Convert DocRED predictions back to BRAT format
- `docred2view.py`: Generate human-readable view of relations
- `relations_stats.py`: Analyze relation statistics in datasets
- `collect_rel_labels.py`: Extract relation labels from data

### Core Modules

- `model.py`: Main DocREModel implementation
- `prepro.py`: Data preprocessing and feature generation
- `evaluation.py`: Evaluation metrics and official scoring
- `losses.py`: Custom loss functions (ATLoss)
- `long_seq.py`: Long sequence processing utilities
- `utils.py`: General utility functions
- `args.py`: Command-line argument definitions

## Model Architecture Details

### Entity Representation
```python
# Multiple mentions per entity are aggregated using log-sum-exp
e_emb = torch.logsumexp(torch.stack(mention_embs, dim=0), dim=0)
```

### Relation Classification
```python
# Grouped bilinear classification
hs = head_extractor(torch.cat([head_emb, context_emb], dim=-1))
ts = tail_extractor(torch.cat([tail_emb, context_emb], dim=-1))
logits = bilinear(grouped_interaction(hs, ts))
```

### Evidence Extraction
```python
# Sentence-level attention for evidence
s_attn = sentence_pooling(token_attention, sentence_positions)
evidence_pred = s_attn > evidence_threshold
```

## Evaluation Metrics

The model is evaluated using:
- **Precision, Recall, F1** for relation extraction
- **Evidence F1** for evidence sentence identification  
- **Ignore F1** excluding "no relation" predictions

Results are reported in both micro and macro averaged formats.

## Technical Requirements

- **Python**: 3.7+
- **PyTorch**: 1.11.0+
- **Transformers**: 4.14.1+
- **GPU**: Recommended for training (supports CUDA)
- **Memory**: 16GB+ RAM recommended for large documents

## File Structure

```
dreeam-ru/
├── run.py                 # Main training/inference script
├── model.py              # Core model implementation
├── prepro.py             # Data preprocessing
├── evaluation.py         # Evaluation utilities
├── args.py               # Argument definitions
├── dreeam-config.json    # Configuration file
├── requirements.txt      # Dependencies
├── scripts/              # Data processing tools
├── dataset/              # Data directory
├── logs/                 # Training logs and checkpoints
└── wandb/                # Weights & Biases logs
```

## Output Files

After training/inference, the model generates:
- `best.ckpt`: Best model checkpoint based on dev F1
- `last.ckpt`: Final epoch checkpoint
- `results.json`: Relation predictions in official format
- `scores.csv`: Evaluation metrics
- `topk_results.json`: Top-k predictions for fusion

## Citation

If you use this code, please cite the original DREEAM paper and acknowledge this Russian adaptation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about this Russian adaptation, please open an issue in the repository. 