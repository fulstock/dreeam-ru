#!/usr/bin/env python3
"""
NEREL1.1 Inference and Evaluation Script

This script processes NEREL1.1 BRAT format files, runs relation extraction inference,
and evaluates the predictions against gold standard relations.
"""

import os
import json
import re
import csv
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
            with open(txt_path, 'r', encoding='utf-8') as f:
                self.text = f.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['cp1251', 'latin-1', 'utf-8-sig']:
                try:
                    with open(txt_path, 'r', encoding=encoding) as f:
                        self.text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
        
        # Parse annotation file
        self.entities = {}
        self.relations = []
        
        with open(ann_path, 'r', encoding='utf-8') as f:
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


class NERELEvaluator:
    """Evaluator for NEREL relation extraction."""

    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

        self.relation_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

        # Store detailed predictions for error analysis
        self.detailed_predictions = []
        
    def add_predictions(self, gold_relations: Set[Tuple], pred_relations: Set[Tuple], file_id: str = None, text: str = None):
        """
        Add predictions for evaluation.

        Args:
            gold_relations: Set of gold relation tuples (head_text, tail_text, relation_type)
            pred_relations: Set of predicted relation tuples (head_text, tail_text, relation_type)
            file_id: File identifier for tracking predictions
            text: Original text for context
        """

        # Calculate metrics
        tp_relations = gold_relations & pred_relations
        fp_relations = pred_relations - gold_relations
        fn_relations = gold_relations - pred_relations

        self.true_positives += len(tp_relations)
        self.false_positives += len(fp_relations)
        self.false_negatives += len(fn_relations)

        # Store detailed predictions for error analysis
        file_prediction = {
            'file_id': file_id,
            'text': text[:500] + '...' if text and len(text) > 500 else text,  # Truncate long texts
            'gold_relations': list(gold_relations),
            'predicted_relations': list(pred_relations),
            'true_positives': list(tp_relations),
            'false_positives': list(fp_relations),
            'false_negatives': list(fn_relations),
            'metrics': {
                'tp_count': len(tp_relations),
                'fp_count': len(fp_relations),
                'fn_count': len(fn_relations),
                'precision': len(tp_relations) / len(pred_relations) if pred_relations else 0,
                'recall': len(tp_relations) / len(gold_relations) if gold_relations else 0
            }
        }
        self.detailed_predictions.append(file_prediction)

        # Per-relation statistics
        for rel in tp_relations:
            self.relation_stats[rel[2]]['tp'] += 1

        for rel in fp_relations:
            self.relation_stats[rel[2]]['fp'] += 1

        for rel in fn_relations:
            self.relation_stats[rel[2]]['fn'] += 1
    
    def get_metrics(self) -> Dict:
        """Calculate and return evaluation metrics."""
        
        # Overall metrics
        precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
        recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Per-relation metrics
        relation_metrics = {}
        for rel_type, stats in self.relation_stats.items():
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            rel_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            rel_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            rel_f1 = 2 * rel_precision * rel_recall / (rel_precision + rel_recall) if (rel_precision + rel_recall) > 0 else 0

            relation_metrics[rel_type] = {
                'precision': rel_precision,
                'recall': rel_recall,
                'f1': rel_f1,
                'support': tp + fn,  # Number of true instances
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        return {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': self.true_positives,
                'false_positives': self.false_positives,
                'false_negatives': self.false_negatives
            },
            'per_relation': relation_metrics
        }


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


def extract_gold_relations(parsed_data: Dict) -> Set[Tuple[str, str, str]]:
    """
    Extract gold relations from BRAT parsed data.
    
    Args:
        parsed_data: Output from BRATParser.parse_file()
        
    Returns:
        Set of relation tuples (head_text, tail_text, relation_type)
    """
    gold_relations = set()
    
    for relation in parsed_data['relations']:
        arg1_id = relation['arg1']
        arg2_id = relation['arg2']
        rel_type = relation['type']
        
        if arg1_id in parsed_data['entities'] and arg2_id in parsed_data['entities']:
            head_text = parsed_data['entities'][arg1_id]['text']
            tail_text = parsed_data['entities'][arg2_id]['text']
            
            gold_relations.add((head_text, tail_text, rel_type))
    
    return gold_relations


def process_nerel_files(test_dir: str, 
                       inference_model: DreeamInference,
                       max_files: int = None,
                       verbose: bool = False) -> Dict:
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
    evaluator = NERELEvaluator()
    
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
        
        try:
            # Parse BRAT files
            parsed_data = parser.parse_file(txt_path, ann_path)
            
            # Convert to inference format
            text, entities = convert_brat_to_inference_format(parsed_data)
            
            if len(entities) < 2:  # Need at least 2 entities for relations
                if verbose:
                    print(f"Skipping {txt_file}: insufficient entities ({len(entities)})")
                continue
            
            # Extract gold relations
            gold_relations = extract_gold_relations(parsed_data)
            
            # Run inference
            pred_relations_list = inference_model.predict_single(text, entities, base_name)
            pred_relations = set(pred_relations_list)
            
            # Evaluate
            evaluator.add_predictions(gold_relations, pred_relations, base_name, text)
            
            total_gold_relations += len(gold_relations)
            total_pred_relations += len(pred_relations)
            
            processed_files += 1
            
            if verbose:
                print(f"Processed {base_name}: {len(entities)} entities, {len(gold_relations)} gold relations, {len(pred_relations)} predicted relations")
                
                if len(pred_relations) > 0:
                    print(f"  Sample predictions: {list(pred_relations)[:3]}")
                
        except Exception as e:
            if verbose:
                print(f"Error processing {txt_file}: {e}")
            continue
    
    print(f"\nProcessed {processed_files} files")
    print(f"Total gold relations: {total_gold_relations}")
    print(f"Total predicted relations: {total_pred_relations}")
    
    return evaluator.get_metrics()


def print_results(metrics: Dict):
    """Print evaluation results in a nice format."""
    
    overall = metrics['overall']
    per_relation = metrics['per_relation']
    
    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall:    {overall['recall']:.4f}")
    print(f"F1 Score:  {overall['f1']:.4f}")
    print(f"")
    print(f"True Positives:  {overall['true_positives']}")
    print(f"False Positives: {overall['false_positives']}")
    print(f"False Negatives: {overall['false_negatives']}")
    
    print("\n" + "="*90)
    print("PER-RELATION RESULTS")
    print("="*90)
    print(f"{'Relation':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'TP':<8} {'FP':<8} {'FN':<8} {'Support':<10}")
    print("-" * 90)

    # Sort relations by F1 score
    sorted_relations = sorted(per_relation.items(), key=lambda x: x[1]['f1'], reverse=True)

    for rel_type, metrics_rel in sorted_relations:
        print(f"{rel_type:<20} {metrics_rel['precision']:<10.4f} {metrics_rel['recall']:<10.4f} "
              f"{metrics_rel['f1']:<10.4f} {metrics_rel['tp']:<8} {metrics_rel['fp']:<8} {metrics_rel['fn']:<8} {metrics_rel['support']:<10}")

    # Calculate macro and micro averages
    if per_relation:
        macro_precision = sum(m['precision'] for m in per_relation.values()) / len(per_relation)
        macro_recall = sum(m['recall'] for m in per_relation.values()) / len(per_relation)
        macro_f1 = sum(m['f1'] for m in per_relation.values()) / len(per_relation)

        print("-" * 90)
        print(f"{'MACRO AVG':<20} {macro_precision:<10.4f} {macro_recall:<10.4f} {macro_f1:<10.4f}")
        print(f"{'MICRO AVG':<20} {overall['precision']:<10.4f} {overall['recall']:<10.4f} {overall['f1']:<10.4f}")


def write_csv_results(metrics: Dict, output_path: str):
    """Write evaluation results to CSV format for easy copying to Excel."""

    overall = metrics['overall']
    per_relation = metrics['per_relation']

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Overall metrics section
        writer.writerow(['OVERALL METRICS'])
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Precision', f"{overall['precision']:.4f}"])
        writer.writerow(['Recall', f"{overall['recall']:.4f}"])
        writer.writerow(['F1 Score', f"{overall['f1']:.4f}"])
        writer.writerow(['True Positives', overall['true_positives']])
        writer.writerow(['False Positives', overall['false_positives']])
        writer.writerow(['False Negatives', overall['false_negatives']])
        writer.writerow([])  # Empty row for separation

        # Per-relation metrics section
        writer.writerow(['PER-RELATION METRICS'])
        writer.writerow(['Relation', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN', 'Support'])

        # Sort relations by F1 score
        sorted_relations = sorted(per_relation.items(), key=lambda x: x[1]['f1'], reverse=True)

        for rel_type, metrics_rel in sorted_relations:
            writer.writerow([
                rel_type,
                f"{metrics_rel['precision']:.4f}",
                f"{metrics_rel['recall']:.4f}",
                f"{metrics_rel['f1']:.4f}",
                metrics_rel['tp'],
                metrics_rel['fp'],
                metrics_rel['fn'],
                metrics_rel['support']
            ])

        # Macro and micro averages
        if per_relation:
            macro_precision = sum(m['precision'] for m in per_relation.values()) / len(per_relation)
            macro_recall = sum(m['recall'] for m in per_relation.values()) / len(per_relation)
            macro_f1 = sum(m['f1'] for m in per_relation.values()) / len(per_relation)

            writer.writerow([])  # Empty row
            writer.writerow(['MACRO AVG', f"{macro_precision:.4f}", f"{macro_recall:.4f}", f"{macro_f1:.4f}", '', '', '', ''])
            writer.writerow(['MICRO AVG', f"{overall['precision']:.4f}", f"{overall['recall']:.4f}", f"{overall['f1']:.4f}",
                           overall['true_positives'], overall['false_positives'], overall['false_negatives'], ''])


def save_detailed_predictions(evaluator: NERELEvaluator, output_path: str):
    """Save detailed predictions for manual error analysis."""

    predictions_data = {
        'summary': {
            'total_files': len(evaluator.detailed_predictions),
            'total_gold_relations': sum(p['metrics']['tp_count'] + p['metrics']['fn_count'] for p in evaluator.detailed_predictions),
            'total_predicted_relations': sum(p['metrics']['tp_count'] + p['metrics']['fp_count'] for p in evaluator.detailed_predictions),
            'overall_metrics': {
                'precision': evaluator.true_positives / (evaluator.true_positives + evaluator.false_positives) if (evaluator.true_positives + evaluator.false_positives) > 0 else 0,
                'recall': evaluator.true_positives / (evaluator.true_positives + evaluator.false_negatives) if (evaluator.true_positives + evaluator.false_negatives) > 0 else 0,
                'tp': evaluator.true_positives,
                'fp': evaluator.false_positives,
                'fn': evaluator.false_negatives
            }
        },
        'file_predictions': evaluator.detailed_predictions
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, indent=2, ensure_ascii=False)


def save_error_analysis_csv(evaluator: NERELEvaluator, output_path: str):
    """Save error analysis in CSV format for easy manual review."""

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Header
        writer.writerow(['File_ID', 'Error_Type', 'Head_Entity', 'Tail_Entity', 'Relation_Type', 'Text_Context'])

        for prediction in evaluator.detailed_predictions:
            file_id = prediction['file_id']
            text_context = prediction['text']

            # False Positives (predicted but not in gold)
            for fp_relation in prediction['false_positives']:
                head, tail, rel_type = fp_relation
                writer.writerow([file_id, 'FALSE_POSITIVE', head, tail, rel_type, text_context])

            # False Negatives (in gold but not predicted)
            for fn_relation in prediction['false_negatives']:
                head, tail, rel_type = fn_relation
                writer.writerow([file_id, 'FALSE_NEGATIVE', head, tail, rel_type, text_context])


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Inference and Evaluation')
    parser.add_argument('--test_dir', default='/home/student1/data/relations/NEREL1.1_sent/test',
                      help='Path to test directory')
    parser.add_argument('--config_path', default='./nerel-config.json',
                      help='Path to configuration file')
    parser.add_argument('--max_files', type=int, default=None,
                      help='Maximum number of files to process (for testing)')
    parser.add_argument('--output_file', default='nerel_evaluation_results2.json',
                      help='Output file for detailed results')
    parser.add_argument('--csv_output', default='nerel_evaluation_results.csv',
                      help='Output CSV file for metrics (easy to copy to Excel)')
    parser.add_argument('--predictions_output', default='nerel_detailed_predictions.json',
                      help='Output JSON file with detailed predictions for error analysis')
    parser.add_argument('--errors_csv', default='nerel_error_analysis.csv',
                      help='Output CSV file with error analysis (FP/FN breakdown)')
    parser.add_argument('--verbose', action='store_true',
                      help='Print verbose output')
    
    args = parser.parse_args()
    
    # Check if test directory exists
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory not found: {args.test_dir}")
        return
    
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
    print(f"\nProcessing NEREL1.1 test files from: {args.test_dir}")

    evaluator = NERELEvaluator()

    # Process files manually to get access to evaluator
    parser_brat = BRATParser()
    txt_files = [f for f in os.listdir(args.test_dir) if f.endswith('.txt')]

    if args.max_files:
        txt_files = txt_files[:args.max_files]

    processed_files = 0
    total_gold_relations = 0
    total_pred_relations = 0

    print(f"Processing {len(txt_files)} files...")

    for txt_file in txt_files:
        base_name = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(args.test_dir, txt_file)
        ann_path = os.path.join(args.test_dir, base_name + '.ann')

        if not os.path.exists(ann_path):
            if args.verbose:
                print(f"Warning: No annotation file for {txt_file}")
            continue

        try:
            # Parse BRAT files
            parsed_data = parser_brat.parse_file(txt_path, ann_path)

            # Convert to inference format
            text, entities = convert_brat_to_inference_format(parsed_data)

            if len(entities) < 2:  # Need at least 2 entities for relations
                if args.verbose:
                    print(f"Skipping {txt_file}: insufficient entities ({len(entities)})")
                continue

            # Extract gold relations
            gold_relations = extract_gold_relations(parsed_data)

            # Run inference
            pred_relations_list = inference_model.predict_single(text, entities, base_name)
            pred_relations = set(pred_relations_list)

            # Evaluate
            evaluator.add_predictions(gold_relations, pred_relations, base_name, text)

            total_gold_relations += len(gold_relations)
            total_pred_relations += len(pred_relations)

            processed_files += 1

            if args.verbose:
                print(f"Processed {base_name}: {len(entities)} entities, {len(gold_relations)} gold relations, {len(pred_relations)} predicted relations")

                if len(pred_relations) > 0:
                    print(f"  Sample predictions: {list(pred_relations)[:3]}")

        except Exception as e:
            if args.verbose:
                print(f"Error processing {txt_file}: {e}")
            continue

    print(f"\nProcessed {processed_files} files")
    print(f"Total gold relations: {total_gold_relations}")
    print(f"Total predicted relations: {total_pred_relations}")

    # Get metrics from evaluator
    metrics = evaluator.get_metrics()

    # Print results
    print_results(metrics)

    # Save detailed results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Save CSV results
    write_csv_results(metrics, args.csv_output)

    # Save detailed predictions for error analysis
    save_detailed_predictions(evaluator, args.predictions_output)

    # Save error analysis CSV
    save_error_analysis_csv(evaluator, args.errors_csv)

    print(f"\nDetailed results saved to: {args.output_file}")
    print(f"CSV results saved to: {args.csv_output}")
    print(f"Detailed predictions saved to: {args.predictions_output}")
    print(f"Error analysis CSV saved to: {args.errors_csv}")
    print(f"\nFor manual error analysis:")
    print(f"- Open {args.errors_csv} to see all false positives and false negatives")
    print(f"- Open {args.predictions_output} for detailed per-file predictions")
    print(f"- CSV files can be opened in Excel for easy analysis")


if __name__ == "__main__":
    main() 