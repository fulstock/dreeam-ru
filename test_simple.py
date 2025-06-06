#!/usr/bin/env python3
"""
Simple test script to debug the DreeamInference step by step.
"""

from dreeam_inference import DreeamInference
import os

def test_simple():
    """Test with the simplest possible case."""
    
    # Check if we have a config file
    config_path = "./nerel-config.json"
    if not os.path.exists(config_path):
        # Try alternative config files
        alternative_configs = ["./dreeam-config.json", "./config.json"]
        for alt_config in alternative_configs:
            if os.path.exists(alt_config):
                config_path = alt_config
                break
        else:
            print(f"No config file found. Tried: {[config_path] + alternative_configs}")
            return
    
    print(f"Using config: {config_path}")
    
    try:
        # Initialize inference
        print("Initializing inference model...")
        inference = DreeamInference(
            config_path=config_path,
            device="auto"
        )
        print("Model loaded successfully!")
        
        # Test with very simple text
        print("\nTesting with simple text...")
        text = "Иван работает в Газпром."
        entities = [
            {"text": "Иван", "start": 0, "end": 4, "type": "PERSON"},
            {"text": "Газпром", "start": 14, "end": 21, "type": "ORGANIZATION"}
        ]
        
        print(f"Text: '{text}'")
        print(f"Entities: {entities}")
        
        # Test conversion to DocRED format
        print("\nTesting DocRED conversion...")
        sample = inference._convert_to_docred_format(text, entities, "test")
        print(f"DocRED sample: {sample}")
        
        # Test feature creation
        print("\nTesting feature creation...")
        features = inference._create_features([sample])
        print(f"Features created: {len(features)}")
        
        if features:
            feature = features[0]
            print(f"Feature keys: {feature.keys()}")
            print(f"Input IDs length: {len(feature['input_ids'])}")
            print(f"Entity positions: {feature['entity_pos']}")
            print(f"HTS pairs: {feature['hts']}")
            print(f"Labels shape: {len(feature['labels'])}")
        
        # Try actual prediction
        print("\nTesting prediction...")
        relations = inference.predict_single(text, entities, "test")
        print(f"Predicted relations: {relations}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple() 