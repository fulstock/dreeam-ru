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
    # Check for available models and prefer nerel-ckpt
    logs_dir = "./logs"
    if os.path.exists(logs_dir):
        available_models = [d for d in os.listdir(logs_dir) 
                          if os.path.isdir(os.path.join(logs_dir, d))]
        
        # Prefer nerel-ckpt if available, otherwise use first available
        if "nerel-ckpt" in available_models:
            model_path = os.path.join(logs_dir, "nerel-ckpt")
        elif available_models:
            model_path = os.path.join(logs_dir, available_models[0])
            print(f"Available models: {available_models}")
            print(f"Using: {model_path}")
        else:
            raise FileNotFoundError(f"No models found in {logs_dir}")
    else:
        raise FileNotFoundError(f"Logs directory {logs_dir} not found")
    
    inference = DreeamInference(
        model_path=model_path,
        config_path="./dreeam-config.json",
        device="auto",
        batch_size=4
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
        print("1. Update the model_path in the script to point to your trained model")
        print("2. Ensure the dreeam-config.json file exists")
        print("3. Ensure the rel2id.json file exists in the data directory")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc() 