#!/usr/bin/env python3
"""
NEREL1.1 Inference and Evaluation Script

This script processes NEREL1.1 BRAT format files, runs relation extraction inference,
and evaluates the predictions against gold standard relations.
"""

import os
import json
import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
from dreeam_inference import DreeamInference
import argparse


class BRATParser:
    """Parser for BRAT annotation format files."""
    
    def __init__(self):
        self.entities = {}
        self.relations = []
        self.text = ""
    
    def parse_file(self, txt_path: str, ann_path: str) -> Dict:
        """
        Parse a BRAT file pair (.txt and .ann).
        
        Args:
            txt_path: Path to the text file
            ann_path: Path to the annotation file
            
        Returns:
            Dictionary with parsed data
        """
        # Read text file with proper encoding
        try:
            with open(txt_path, 'r', encoding='utf-8', newline='') as f:
                self.text = f.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['cp1251', 'latin-1', 'utf-8-sig']:
                try:
                    with open(txt_path, 'r', encoding=encoding, newline='') as f:
                        self.text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
        
        # Parse annotation file
        self.entities = {}
        self.entityspan2id = {}
        self.relations = []
        
        with open(ann_path, 'r', encoding='utf-8', newline='') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('T'):  # Entity
                    self._parse_entity(line)
                elif line.startswith('R'):  # Relation
                    self._parse_relation(line)
        
        return {
            'text': self.text,
            'entities': self.entities,
            'entityspan2id' : self.entityspan2id,
            'relations': self.relations,
            'file_id': os.path.splitext(os.path.basename(txt_path))[0]
        }
    
    def _parse_entity(self, line: str):
        """Parse entity annotation line."""
        parts = line.split('\t')
        if len(parts) < 3:
            return
        
        entity_id = parts[0]
        entity_info = parts[1].split()
        entity_text = parts[2] if len(parts) > 2 else ""
        
        if len(entity_info) >= 3:
            entity_type = entity_info[0]
            start_pos = int(entity_info[1])
            end_pos = int(entity_info[2])
            
            self.entities[entity_id] = {
                'id': entity_id,
                'type': entity_type,
                'start': start_pos,
                'end': end_pos,
                'text': entity_text
            }
            if entity_text not in self.entityspan2id:
                self.entityspan2id[entity_text] = [entity_id]
            else:
                self.entityspan2id[entity_text].append(entity_id)
    
    def _parse_relation(self, line: str):
        """Parse relation annotation line."""
        parts = line.split('\t')
        if len(parts) < 2:
            return
        
        relation_id = parts[0]
        relation_info = parts[1].split()
        
        if len(relation_info) >= 3:
            relation_type = relation_info[0]
            arg1 = relation_info[1].split(':')[1] if ':' in relation_info[1] else relation_info[1]
            arg2 = relation_info[2].split(':')[1] if ':' in relation_info[2] else relation_info[2]
            
            # Skip some special relations
            if relation_type in ['ALTERNATIVE_NAME']:
                return
            
            self.relations.append({
                'id': relation_id,
                'type': relation_type,
                'arg1': arg1,
                'arg2': arg2
            })


def convert_brat_to_inference_format(parsed_data: Dict) -> Tuple[str, List[Dict]]:
    """
    Convert BRAT parsed data to DreeamInference format.
    
    Args:
        parsed_data: Output from BRATParser.parse_file()
        
    Returns:
        Tuple of (text, entities_list)
    """
    text = parsed_data['text']
    entities = []
    
    for entity_id, entity_info in parsed_data['entities'].items():
        entities.append({
            'text': entity_info['text'],
            'start': entity_info['start'],
            'end': entity_info['end'],
            'type': entity_info['type'],
            'id': entity_id
        })
    
    return text, entities

def process_files(test_dir: str, 
                  output_dir: str,
                  inference_model: DreeamInference,
                  max_files: int = None,
                  verbose: bool = True) -> Dict:
    """
    Process all NEREL test files and evaluate predictions.
    
    Args:
        test_dir: Path to NEREL test directory
        inference_model: DreeamInference instance
        max_files: Maximum number of files to process (for testing)
        verbose: Whether to print detailed progress
        
    Returns:
        Evaluation results
    """
    parser = BRATParser()
    
    # Find all text files
    txt_files = [f for f in os.listdir(test_dir) if f.endswith('.txt')]
    
    if max_files:
        txt_files = txt_files[:max_files]
    
    processed_files = 0
    total_gold_relations = 0
    total_pred_relations = 0
    
    print(f"Processing {len(txt_files)} files...")
    
    for txt_file in txt_files:
        base_name = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(test_dir, txt_file)
        ann_path = os.path.join(test_dir, base_name + '.ann')
        
        if not os.path.exists(ann_path):
            if verbose:
                print(f"Warning: No annotation file for {txt_file}")
            continue
        parsed_data = parser.parse_file(txt_path, ann_path)
        
        # Convert to inference format
        text, entities = convert_brat_to_inference_format(parsed_data)
        
        if len(entities) < 2:  # Need at least 2 entities for relations
            if verbose:
                print(f"Skipping {txt_file}: insufficient entities ({len(entities)})")
            continue

        # Run inference
        pred_relations_list = inference_model.predict_single(text, entities, base_name)
        pred_relations = set(pred_relations_list)
        
        total_pred_relations += len(pred_relations)

        with open(os.path.join(output_dir, txt_file), "w", encoding = "UTF-8", newline='') as tf:
            print("Writing text of file", txt_file, "into", os.path.join(output_dir, txt_file))
            tf.write(text)

        with open(os.path.join(output_dir, txt_file.replace('.txt','.ann')), "w", encoding = "UTF-8", newline='') as af:
            print("Writing ner golds and rel perds of file", txt_file.replace('.txt','.ann'), "into", os.path.join(output_dir, txt_file.replace('.txt','.ann')))
            for p_idx, p in enumerate(sorted(list(entities), key = lambda e : (e["start"], e["type"], e["end"]))):
                af.write("T" + str(p_idx + 1) + "\t" + p["type"] + " " + str(p["start"]) + " " + str(p["end"]) + "\t" + p["text"] + "\n")
            p_idx = 0
            for p in sorted(list(pred_relations)):
                # Handle both (s,e,t) and (s,e,t,text) formats
                h, t, r = p

                for h1 in parsed_data["entityspan2id"][h]:
                    for t2 in parsed_data["entityspan2id"][t]:
                        af.write("R" + str(p_idx + 1) + "\t" + r + " Arg1:" + str(h1) + " Arg2:" + str(t2) + "\n")
                        p_idx += 1
        
        processed_files += 1
        
        if verbose:
            print(f"Processed {base_name}: {len(entities)} entities, {len(pred_relations)} predicted relations")
            
            if len(pred_relations) > 0:
                print(f"  Sample predictions: {list(pred_relations)[:3]}")
    
    print(f"\nProcessed {processed_files} files")
    print(f"Total predicted relations: {total_pred_relations}")
    
    return 

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='NEREL1.1 Inference and Evaluation')
    parser.add_argument('--test_dir', default='S:/HRCode/data/NEREL1.1/test',
                      help='Path to test directory')
    parser.add_argument('--output_dir', default='S:/HRCode/data/NEREL1.1/test',
                      help='Path to prediction directory')
    parser.add_argument('--model_path', default='./ckpt/NEREL-rubert-ep40-single.ckpt',
                      help='Path to trained model')
    parser.add_argument('--config_path', default='./brat-inf-config.json',
                      help='Path to configuration file')
    parser.add_argument('--max_files', type=int, default=None,
                      help='Maximum number of files to process (for testing)')
    parser.add_argument('--verbose', action='store_true',
                      help='Print verbose output')
    
    args = parser.parse_args()
    
    # Check if test directory exists
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory not found: {args.test_dir}")
        return
    
    # Check for available models if default doesn't exist
    if not os.path.exists(args.model_path):
        logs_dir = "./logs"
        if os.path.exists(logs_dir):
            available_models = [d for d in os.listdir(logs_dir) 
                              if os.path.isdir(os.path.join(logs_dir, d))]
            if "nerel-ckpt" in available_models:
                args.model_path = os.path.join(logs_dir, "nerel-ckpt")
            elif available_models:
                args.model_path = os.path.join(logs_dir, available_models[0])
                print(f"Using model: {args.model_path}")
    
    print("Initializing DREEAM inference model...")
    try:
        inference_model = DreeamInference(
            config_path=args.config_path,
            device="auto"
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process files and evaluate
    print(f"\nProcessing test files from: {args.test_dir}")
    
    process_files(
        test_dir=args.test_dir,
        output_dir = args.output_dir,
        inference_model=inference_model,
        max_files=args.max_files,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main() 