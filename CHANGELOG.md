# DreeamInference Interface Changelog

## [2024-12-19] - Interface Simplification

### Changed
- **Simplified DreeamInference constructor**: Now only takes `config_path` and `device` arguments
- **Moved parameters to config file**: `batch_size`, `num_labels`, `model_path`, `max_seq_length`, `max_sent_num`, and `evi_thresh` are now read from the configuration file
- **Updated configuration file**: Added `model_path` and `batch_size` parameters to `nerel-config.json`

### Migration Guide

#### Before:
```python
inference = DreeamInference(
    model_path="./logs/nerel-ckpt",
    config_path="./nerel-config.json", 
    device="auto",
    max_seq_length=1024,
    max_sent_num=25,
    evi_thresh=0.2,
    num_labels=4,
    batch_size=8
)
```

#### After:
```python
inference = DreeamInference(
    config_path="./nerel-config.json",
    device="auto"
)
```

### Configuration File Changes

The configuration file now includes all parameters that were previously passed as constructor arguments:

```json
{
    "transformer_type": "bert",
    "model_name_or_path": "DeepPavlov/rubert-base-cased",
    "model_path": "./logs/nerel-ckpt",
    "load_path": "./ckpt/NEREL-rubert-ep40-single.ckpt",
    "rel2id": "./rel2id/nerel-rel2id.json",
    "num_labels": 2,
    "num_class": 48,
    "evi_thresh": 0.2,
    "max_seq_length": 1600,
    "max_sent_num": 101,
    "batch_size": 8
}
```

### Updated Files
- `dreeam_inference.py` - Simplified constructor and parameter loading
- `nerel-config.json` - Added `model_path` and `batch_size` parameters
- `example_inference.py` - Updated to use new interface
- `test_simple.py` - Updated to use new interface
- `nerel_inference_eval.py` - Updated to use new interface

### Benefits
- **Cleaner interface**: Fewer parameters to remember and pass
- **Configuration-driven**: All model parameters are centralized in config file
- **Easier maintenance**: Changing model parameters no longer requires code changes
- **Better reproducibility**: Complete model configuration is stored in a single file 