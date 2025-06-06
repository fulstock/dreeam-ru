#!/usr/bin/env python3
"""
Example usage of the DreeamInference class for relation extraction.

This script demonstrates how to use the DreeamInference class to perform
document-level relation extraction on raw text with named entities.
"""

from dreeam_inference import DreeamInference
import json
import os


def main():
    """Main function demonstrating DreeamInference usage."""
    
    # Initialize the inference model
    # Use config file - model path is now specified in the config
    config_path = "./nerel-config.json"
    if not os.path.exists(config_path):
        # Try alternative config files
        alternative_configs = ["./dreeam-config.json", "./config.json"]
        for alt_config in alternative_configs:
            if os.path.exists(alt_config):
                config_path = alt_config
                break
        else:
            raise FileNotFoundError(f"Configuration file not found. Tried: {[config_path] + alternative_configs}")
    
    inference = DreeamInference(
        config_path=config_path,
        device="auto"
    )
    
    print("Model loaded successfully!")
    print(f"Available relations: {list(inference.rel2id.keys())}")
    print("-" * 50)
    
    # Example 1: Single document inference
    print("Example 1: Single document")
    
    text1 = """Иван Петров работает в компании Газпром. 
    Он является директором отдела продаж. 
    Компания Газпром была основана в 1989 году."""
    
    entities1 = [
        {"text": "Иван Петров", "start": 0, "end": 11, "type": "PERSON"},
        {"text": "Газпром", "start": 33, "end": 40, "type": "ORGANIZATION"},
        {"text": "1989", "start": 115, "end": 119, "type": "DATE"}
    ]
    
    relations1 = inference.predict_single(text1, entities1, title="Документ 1")
    
    print(f"Text: {text1}")
    print(f"Entities: {[e['text'] for e in entities1]}")
    print("Predicted relations:")
    for head, tail, relation in relations1:
        print(f"  {head} -> {relation} -> {tail}")
    print()
    
    # Example 2: Batch inference
    print("Example 2: Batch inference")
    
    texts = [
        "Владимир Путин является президентом России. Москва - столица России.",
        "Apple была основана Стивом Джобсом. Компания находится в Калифорнии."
    ]
    
    entities_list = [
        [
            {"text": "Владимир Путин", "start": 0, "end": 14, "type": "PERSON"},
            {"text": "России", "start": 36, "end": 42, "type": "COUNTRY"},
            {"text": "Москва", "start": 44, "end": 50, "type": "CITY"}
        ],
        [
            {"text": "Apple", "start": 0, "end": 5, "type": "ORGANIZATION"},
            {"text": "Стивом Джобсом", "start": 19, "end": 33, "type": "PERSON"},
            {"text": "Калифорнии", "start": 57, "end": 67, "type": "LOCATION"}
        ]
    ]
    
    titles = ["Документ 2", "Документ 3"]
    
    batch_relations = inference.predict_relations(texts, entities_list, titles)
    
    for i, (text, entities, relations) in enumerate(zip(texts, entities_list, batch_relations)):
        print(f"Document {i+1}: {text}")
        print(f"Entities: {[e['text'] for e in entities]}")
        print("Predicted relations:")
        for head, tail, relation in relations:
            print(f"  {head} -> {relation} -> {tail}")
        print()
    
    # Example 3: Processing from JSON file
    print("Example 3: Processing from JSON file")
    
    # Create example data
    example_data = [
        {
            "title": "Новость 1",
            "text": "Сергей Лавров встретился с министром иностранных дел Германии.",
            "entities": [
                {"text": "Сергей Лавров", "start": 0, "end": 13, "type": "PERSON"},
                {"text": "Германии", "start": 56, "end": 64, "type": "COUNTRY"}
            ]
        },
        {
            "title": "Новость 2", 
            "text": "Компания Яндекс объявила о новом проекте в области искусственного интеллекта.",
            "entities": [
                {"text": "Яндекс", "start": 9, "end": 15, "type": "ORGANIZATION"}
            ]
        }
    ]
    
    # Save example data to file
    with open("example_data.json", "w", encoding="utf-8") as f:
        json.dump(example_data, f, ensure_ascii=False, indent=2)
    
    # Process data from file
    with open("example_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts_from_file = [item["text"] for item in data]
    entities_from_file = [item["entities"] for item in data]
    titles_from_file = [item["title"] for item in data]
    
    file_relations = inference.predict_relations(texts_from_file, entities_from_file, titles_from_file)
    
    print("Results from JSON file:")
    for i, (item, relations) in enumerate(zip(data, file_relations)):
        print(f"{item['title']}: {item['text']}")
        print(f"Entities: {[e['text'] for e in item['entities']]}")
        print("Predicted relations:")
        if relations:
            for head, tail, relation in relations:
                print(f"  {head} -> {relation} -> {tail}")
        else:
            print("  No relations found")
        print()


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Ensure the configuration file (nerel-config.json) exists and has the correct model_path")
        print("2. Ensure the model checkpoint directory exists at the specified model_path")
        print("3. Ensure the rel2id.json file exists at the path specified in the config")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc() 