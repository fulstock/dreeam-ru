#!/usr/bin/env python3
"""
Diagnostic script to check DREEAM model compatibility and configuration.

This script helps diagnose issues with model loading and configuration.
"""

import os
import json
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def check_model_checkpoint(model_path):
    """Check if model checkpoint exists and inspect its contents."""
    print(f"Checking model path: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model path '{model_path}' does not exist")
        return False
    
    # Check for checkpoint files
    best_ckpt = os.path.join(model_path, "best.ckpt")
    last_ckpt = os.path.join(model_path, "last.ckpt")
    
    checkpoint_path = None
    if os.path.exists(best_ckpt):
        checkpoint_path = best_ckpt
        print(f"✅ Found best.ckpt")
    elif os.path.exists(last_ckpt):
        checkpoint_path = last_ckpt
        print(f"✅ Found last.ckpt")
    else:
        print(f"❌ No checkpoint files found (best.ckpt or last.ckpt)")
        return False
    
    # Try to load checkpoint and inspect its keys
    try:
        print(f"\nInspecting checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        print(f"✅ Checkpoint loaded successfully")
        print(f"Number of parameters: {len(state_dict)}")
        
        # Check for problematic keys
        problematic_keys = [k for k in state_dict.keys() if "position_ids" in k]
        if problematic_keys:
            print(f"⚠️  Found position_ids keys: {problematic_keys}")
        else:
            print(f"✅ No position_ids keys found (good for compatibility)")
        
        # Show first few keys
        keys = list(state_dict.keys())
        print(f"First 5 keys: {keys[:5]}")
        print(f"Last 5 keys: {keys[-5:]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False


def check_config_files(config_path="./dreeam-config.json"):
    """Check configuration files."""
    print(f"\nChecking configuration files...")
    
    # Check main config
    if os.path.exists(config_path):
        print(f"✅ Found config file: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            print(f"Configuration contents:")
            for key, value in config_data.items():
                print(f"  {key}: {value}")
            
            # Check data directory
            data_dir = config_data.get("data_dir", ".")
            rel2id_path = os.path.join(data_dir, "rel2id.json")
            
            if os.path.exists(rel2id_path):
                print(f"✅ Found rel2id.json at: {rel2id_path}")
                with open(rel2id_path, 'r', encoding='utf-8') as f:
                    rel2id = json.load(f)
                print(f"Number of relations: {len(rel2id)}")
                print(f"Relations: {list(rel2id.keys())[:10]}...")  # Show first 10
            else:
                print(f"❌ rel2id.json not found at: {rel2id_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error reading config: {e}")
            return False
    else:
        print(f"❌ Config file not found: {config_path}")
        return False
    
    return True


def check_transformers_model(model_name="DeepPavlov/rubert-base-cased"):
    """Check if transformers model can be loaded."""
    print(f"\nChecking transformers model: {model_name}")
    
    try:
        config = AutoConfig.from_pretrained(model_name)
        print(f"✅ Config loaded successfully")
        print(f"Model type: {config.model_type}")
        print(f"Hidden size: {config.hidden_size}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ Tokenizer loaded successfully")
        print(f"Vocab size: {tokenizer.vocab_size}")
        
        model = AutoModel.from_pretrained(model_name)
        print(f"✅ Model loaded successfully")
        
        # Check if position_ids exist
        if hasattr(model.embeddings, 'position_ids'):
            print(f"✅ Model has position_ids")
        else:
            print(f"⚠️  Model missing position_ids (will be added automatically)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading transformers model: {e}")
        return False


def list_available_models(logs_dir="./logs"):
    """List available trained models."""
    print(f"\nLooking for available models in: {logs_dir}")
    
    if not os.path.exists(logs_dir):
        print(f"❌ Logs directory does not exist: {logs_dir}")
        return []
    
    available_models = []
    for item in os.listdir(logs_dir):
        item_path = os.path.join(logs_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains model files
            has_best = os.path.exists(os.path.join(item_path, "best.ckpt"))
            has_last = os.path.exists(os.path.join(item_path, "last.ckpt"))
            
            if has_best or has_last:
                available_models.append(item)
                status = "✅" if has_best else "⚠️ (only last.ckpt)"
                print(f"  {status} {item}")
    
    if not available_models:
        print(f"❌ No trained models found in {logs_dir}")
    
    return available_models


def main():
    """Main diagnostic function."""
    print("=== DREEAM Model Diagnostics ===\n")
    
    # Check available models
    available_models = list_available_models()
    
    # Check configuration files
    config_ok = check_config_files()
    
    # Check transformers model
    transformers_ok = check_transformers_model()
    
    # Check specific model if available
    if available_models:
        model_path = os.path.join("./logs", available_models[0])
        print(f"\nChecking first available model: {model_path}")
        model_ok = check_model_checkpoint(model_path)
    else:
        print(f"\nNo models available to check")
        model_ok = False
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Configuration files: {'✅' if config_ok else '❌'}")
    print(f"Transformers model: {'✅' if transformers_ok else '❌'}")
    print(f"Trained model: {'✅' if model_ok else '❌'}")
    print(f"Available models: {len(available_models)}")
    
    if config_ok and transformers_ok and model_ok:
        print(f"\n🎉 Everything looks good! You should be able to run inference.")
    else:
        print(f"\n⚠️  Some issues found. Please address them before running inference.")


if __name__ == "__main__":
    main() 