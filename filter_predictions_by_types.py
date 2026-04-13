#!/usr/bin/env python3
"""
Filter predictions based on NEREL annotation.conf entity type constraints.

This script:
1. Loads NEREL annotation rules (which entity types can have which relations)
2. Reads predictions from results.json
3. Reads gold labels from test.json (to get entity types)
4. Filters out predictions that violate type constraints
5. Recalculates metrics (precision, recall, F1) before and after filtering

Usage:
    python filter_predictions_by_types.py \
        --predictions ./logs/model/results.json \
        --gold_file ./data/test.json \
        --annotation_conf ./data/NEREL1.1/annotation.conf \
        --output_csv filtered_metrics.csv
"""

import json
import argparse
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set


class AnnotationRuleParser:
    """Parser for NEREL annotation.conf file."""

    def __init__(self, conf_path: str):
        self.conf_path = conf_path
        self.relation_rules = {}  # {relation: (arg1_types, arg2_types)}
        self._parse()

    def _parse(self):
        """Parse annotation.conf [relations] section."""
        in_relations_section = False

        with open(self.conf_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Check for [relations] section
                if line == '[relations]':
                    in_relations_section = True
                    continue

                # Check for next section
                if line.startswith('[') and in_relations_section:
                    break

                # Skip comments and empty lines
                if not in_relations_section or not line or line.startswith('#'):
                    continue

                # Parse relation line: RELATION_NAME  Arg1:TYPE1|TYPE2, Arg2:TYPE3|TYPE4
                self._parse_relation_line(line)

    def _parse_relation_line(self, line: str):
        """Parse a single relation constraint line."""
        # Format: RELATION_NAME  Arg1:TYPE1|TYPE2, Arg2:TYPE3|TYPE4
        parts = line.split('\t')
        if len(parts) < 2:
            parts = re.split(r'\s{2,}', line)  # Split on multiple spaces

        if len(parts) < 2:
            return

        relation_name = parts[0].strip()
        args_str = parts[1].strip()

        # Parse arguments
        arg_parts = args_str.split(',')
        if len(arg_parts) != 2:
            return

        # Parse Arg1 types
        arg1_match = re.search(r'Arg1:(.+)', arg_parts[0])
        arg2_match = re.search(r'Arg2:(.+)', arg_parts[1])

        if not arg1_match or not arg2_match:
            return

        arg1_types_str = arg1_match.group(1).strip()
        arg2_types_str = arg2_match.group(1).strip()

        # Parse type lists (e.g., "PERSON|PROFESSION" or "<ENTITY>")
        arg1_types = self._parse_type_list(arg1_types_str)
        arg2_types = self._parse_type_list(arg2_types_str)

        self.relation_rules[relation_name] = (arg1_types, arg2_types)

    def _parse_type_list(self, types_str: str) -> Set[str]:
        """Parse type list like 'PERSON|PROFESSION' or '<ENTITY>'."""
        # Handle special case: <ENTITY> means any type
        if '<ENTITY>' in types_str or '<ANY>' in types_str:
            return {'<ANY>'}

        # Split by | and clean
        types = set()
        for t in types_str.split('|'):
            t = t.strip()
            # Remove <OVL-TYPE>: prefix if present
            if ':' in t:
                t = t.split(':')[1]
            types.add(t)

        return types

    def is_valid_relation(self, relation: str, arg1_type: str, arg2_type: str) -> bool:
        """Check if a relation is valid given entity types."""
        if relation not in self.relation_rules:
            # Unknown relation - allow by default
            return True

        allowed_arg1, allowed_arg2 = self.relation_rules[relation]

        # Check if types are allowed
        arg1_valid = '<ANY>' in allowed_arg1 or arg1_type in allowed_arg1
        arg2_valid = '<ANY>' in allowed_arg2 or arg2_type in allowed_arg2

        return arg1_valid and arg2_valid

    def get_violation_reason(self, relation: str, arg1_type: str, arg2_type: str) -> str:
        """Get human-readable reason why a relation is invalid."""
        if relation not in self.relation_rules:
            return "Unknown relation"

        allowed_arg1, allowed_arg2 = self.relation_rules[relation]

        reasons = []
        if '<ANY>' not in allowed_arg1 and arg1_type not in allowed_arg1:
            allowed_str = ', '.join(sorted(allowed_arg1))
            reasons.append(f"Arg1 type '{arg1_type}' not in allowed types: {allowed_str}")

        if '<ANY>' not in allowed_arg2 and arg2_type not in allowed_arg2:
            allowed_str = ', '.join(sorted(allowed_arg2))
            reasons.append(f"Arg2 type '{arg2_type}' not in allowed types: {allowed_str}")

        return '; '.join(reasons) if reasons else "Valid"


class PredictionFilter:
    """Filter predictions based on annotation rules."""

    def __init__(self, rules: AnnotationRuleParser, gold_data: List[Dict]):
        self.rules = rules
        self.gold_data = gold_data

        # Build lookup: {title: {entity_idx: type}}
        self.title_to_entity_types = {}
        self._build_entity_type_lookup()

    def _build_entity_type_lookup(self):
        """Build mapping from (title, entity_idx) to entity type."""
        for doc in self.gold_data:
            title = doc['title']
            entity_types = {}

            for entity_idx, entity_mentions in enumerate(doc['vertexSet']):
                # Get type from first mention (all mentions have same type)
                if entity_mentions:
                    entity_type = entity_mentions[0].get('type', 'UNKNOWN')
                    entity_types[entity_idx] = entity_type

            self.title_to_entity_types[title] = entity_types

    def filter_predictions(self, predictions: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Filter predictions by annotation rules.

        Returns:
            (valid_predictions, invalid_predictions, violation_details)
        """
        valid = []
        invalid = []
        violations = []

        for pred in predictions:
            title = pred['title']
            h_idx = pred['h_idx']
            t_idx = pred['t_idx']
            relation = pred['r']

            # Get entity types
            if title not in self.title_to_entity_types:
                # Unknown document - skip
                invalid.append(pred)
                violations.append({
                    **pred,
                    'violation': 'Document not in gold data',
                    'h_type': 'UNKNOWN',
                    't_type': 'UNKNOWN'
                })
                continue

            entity_types = self.title_to_entity_types[title]
            h_type = entity_types.get(h_idx, 'UNKNOWN')
            t_type = entity_types.get(t_idx, 'UNKNOWN')

            # Check if relation is valid
            if self.rules.is_valid_relation(relation, h_type, t_type):
                valid.append(pred)
            else:
                invalid.append(pred)
                reason = self.rules.get_violation_reason(relation, h_type, t_type)
                violations.append({
                    **pred,
                    'violation': reason,
                    'h_type': h_type,
                    't_type': t_type
                })

        return valid, invalid, violations


def load_gold_labels(gold_file: str) -> Tuple[List[Dict], Set[Tuple]]:
    """
    Load gold labels from test.json.

    Returns:
        (gold_data, gold_relations_set)
        gold_relations_set: {(title, h_idx, t_idx, relation), ...}
    """
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)

    gold_relations = set()
    for doc in gold_data:
        title = doc['title']
        if 'labels' not in doc:
            continue

        for label in doc['labels']:
            h = label['h']
            t = label['t']
            r = label['r']
            gold_relations.add((title, h, t, r))

    return gold_data, gold_relations


def calculate_metrics(predictions: List[Dict], gold_relations: Set[Tuple]) -> Dict:
    """Calculate precision, recall, F1 for predictions."""
    pred_relations = {(p['title'], p['h_idx'], p['t_idx'], p['r']) for p in predictions}

    tp = len(pred_relations & gold_relations)
    fp = len(pred_relations - gold_relations)
    fn = len(gold_relations - pred_relations)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'total_predictions': len(predictions),
        'total_gold': len(gold_relations)
    }


def calculate_per_relation_metrics(predictions: List[Dict], gold_relations: Set[Tuple]) -> Dict:
    """
    Calculate per-relation metrics (like official_evaluate).

    Returns:
        {
            'per_relation': {relation: {'precision': ..., 'recall': ..., 'f1': ..., 'tp': ..., 'fp': ..., 'fn': ..., 'support': ...}},
            'micro_avg': {'precision': ..., 'recall': ..., 'f1': ..., 'tp': ..., 'fp': ..., 'fn': ...},
            'macro_avg': {'precision': ..., 'recall': ..., 'f1': ...}
        }
    """
    # Convert predictions to set of tuples
    pred_relations = {(p['title'], p['h_idx'], p['t_idx'], p['r']) for p in predictions}

    # Initialize per-relation statistics
    relation_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    # Collect all relation types from both predictions and gold
    all_relations = set()
    for _, _, _, r in pred_relations:
        all_relations.add(r)
    for _, _, _, r in gold_relations:
        all_relations.add(r)

    # Calculate TP and FP for each relation
    for title, h, t, r in pred_relations:
        if (title, h, t, r) in gold_relations:
            relation_stats[r]['tp'] += 1
        else:
            relation_stats[r]['fp'] += 1

    # Calculate FN for each relation
    for title, h, t, r in gold_relations:
        if (title, h, t, r) not in pred_relations:
            relation_stats[r]['fn'] += 1

    # Calculate per-relation metrics
    per_relation_metrics = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    f1_sum, precision_sum, recall_sum = 0, 0, 0
    valid_relations_for_macro = 0

    for rel in all_relations:
        stats = relation_stats[rel]
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn  # Total gold instances

        # Only include relations that appear in gold for macro avg
        if support > 0:
            f1_sum += f1
            precision_sum += precision
            recall_sum += recall
            valid_relations_for_macro += 1

        per_relation_metrics[rel] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'support': support
        }

    # Calculate micro average (aggregate all TP/FP/FN)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Calculate macro average (mean of per-relation metrics)
    macro_precision = precision_sum / valid_relations_for_macro if valid_relations_for_macro > 0 else 0.0
    macro_recall = recall_sum / valid_relations_for_macro if valid_relations_for_macro > 0 else 0.0
    macro_f1 = f1_sum / valid_relations_for_macro if valid_relations_for_macro > 0 else 0.0

    return {
        'per_relation': per_relation_metrics,
        'micro_avg': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        },
        'macro_avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        }
    }


def analyze_violations(violations: List[Dict]) -> Dict:
    """Analyze violation patterns."""
    violation_by_relation = defaultdict(int)
    violation_by_type = defaultdict(int)

    for v in violations:
        relation = v['r']
        h_type = v.get('h_type', 'UNKNOWN')
        t_type = v.get('t_type', 'UNKNOWN')

        violation_by_relation[relation] += 1
        # Convert tuple key to string for JSON serialization
        type_pair_key = f"{relation}|{h_type}→{t_type}"
        violation_by_type[type_pair_key] += 1

    return {
        'by_relation': dict(violation_by_relation),
        'by_type_pair': dict(violation_by_type)
    }


def main():
    parser = argparse.ArgumentParser(description='Filter predictions by entity type constraints')
    parser.add_argument('--predictions', required=True, help='Path to results.json (predictions)')
    parser.add_argument('--gold_file', required=True, help='Path to test.json (gold labels)')
    parser.add_argument('--annotation_conf', required=True, help='Path to annotation.conf')
    parser.add_argument('--output_csv', default='filtered_metrics.csv', help='Output CSV file')
    parser.add_argument('--violations_json', default='violations.json', help='Output violations JSON')

    args = parser.parse_args()

    print("="*80)
    print("NEREL Prediction Filtering by Entity Type Constraints")
    print("="*80)

    # Load annotation rules
    print(f"\n1. Loading annotation rules from: {args.annotation_conf}")
    rules = AnnotationRuleParser(args.annotation_conf)
    print(f"   Loaded {len(rules.relation_rules)} relation type constraints")

    # Load gold data
    print(f"\n2. Loading gold labels from: {args.gold_file}")
    gold_data, gold_relations = load_gold_labels(args.gold_file)
    print(f"   Loaded {len(gold_data)} documents")
    print(f"   Total gold relations: {len(gold_relations)}")

    # Load predictions
    print(f"\n3. Loading predictions from: {args.predictions}")
    with open(args.predictions, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    print(f"   Total predictions: {len(predictions)}")

    # Filter predictions
    print(f"\n4. Filtering predictions by type constraints...")
    pred_filter = PredictionFilter(rules, gold_data)
    valid_preds, invalid_preds, violations = pred_filter.filter_predictions(predictions)

    print(f"   Valid predictions: {len(valid_preds)}")
    print(f"   Invalid predictions: {len(invalid_preds)} ({len(invalid_preds)/len(predictions)*100:.2f}%)")

    # Calculate metrics
    print(f"\n5. Calculating metrics...")

    metrics_all = calculate_metrics(predictions, gold_relations)
    metrics_filtered = calculate_metrics(valid_preds, gold_relations)

    # Calculate per-relation metrics
    print(f"   Calculating per-relation metrics...")
    per_rel_all = calculate_per_relation_metrics(predictions, gold_relations)
    per_rel_filtered = calculate_per_relation_metrics(valid_preds, gold_relations)

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print("\n--- ALL PREDICTIONS (baseline) ---")
    print(f"Precision: {metrics_all['precision']:.4f}")
    print(f"Recall:    {metrics_all['recall']:.4f}")
    print(f"F1 Score:  {metrics_all['f1']:.4f}")
    print(f"TP: {metrics_all['tp']}, FP: {metrics_all['fp']}, FN: {metrics_all['fn']}")

    print("\n--- FILTERED PREDICTIONS (valid types only) ---")
    print(f"Precision: {metrics_filtered['precision']:.4f}")
    print(f"Recall:    {metrics_filtered['recall']:.4f}")
    print(f"F1 Score:  {metrics_filtered['f1']:.4f}")
    print(f"TP: {metrics_filtered['tp']}, FP: {metrics_filtered['fp']}, FN: {metrics_filtered['fn']}")

    print("\n--- IMPACT OF FILTERING ---")
    precision_delta = metrics_filtered['precision'] - metrics_all['precision']
    recall_delta = metrics_filtered['recall'] - metrics_all['recall']
    f1_delta = metrics_filtered['f1'] - metrics_all['f1']

    print(f"Precision change: {precision_delta:+.4f} ({precision_delta/metrics_all['precision']*100:+.2f}%)")
    print(f"Recall change:    {recall_delta:+.4f} ({recall_delta/metrics_all['recall']*100:+.2f}%)")
    print(f"F1 change:        {f1_delta:+.4f} ({f1_delta/metrics_all['f1']*100:+.2f}%)")

    # Print macro/micro averages
    print("\n--- MACRO/MICRO AVERAGES ---")
    print(f"Baseline  - Micro: {per_rel_all['micro_avg']['f1']:.4f}, Macro: {per_rel_all['macro_avg']['f1']:.4f}")
    print(f"Filtered  - Micro: {per_rel_filtered['micro_avg']['f1']:.4f}, Macro: {per_rel_filtered['macro_avg']['f1']:.4f}")

    # Print top-10 relations by F1 (filtered)
    print("\n--- TOP 10 RELATIONS BY F1 (Filtered) ---")
    sorted_rels = sorted(per_rel_filtered['per_relation'].items(),
                        key=lambda x: x[1]['f1'], reverse=True)[:10]

    print(f"{'Relation':<25} {'Baseline F1':<12} {'Filtered F1':<12} {'Delta':<10} {'Support':<8}")
    print("-" * 75)
    for rel, metrics_f in sorted_rels:
        metrics_b = per_rel_all['per_relation'].get(rel, {'f1': 0})
        delta = metrics_f['f1'] - metrics_b['f1']
        print(f"{rel:<25} {metrics_b['f1']:<12.4f} {metrics_f['f1']:<12.4f} {delta:<10.4f} {metrics_f['support']:<8}")

    # Analyze violations
    if violations:
        print(f"\n--- VIOLATION ANALYSIS ---")
        violation_stats = analyze_violations(violations)

        print(f"\nTop 10 relations with violations:")
        sorted_violations = sorted(violation_stats['by_relation'].items(),
                                   key=lambda x: x[1], reverse=True)
        for relation, count in sorted_violations[:10]:
            print(f"  {relation:<30} {count:>5} violations")

    # Save results to CSV
    print(f"\n6. Saving results to: {args.output_csv}")
    import csv
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        writer.writerow(['METRIC COMPARISON'])
        writer.writerow(['Metric', 'All Predictions', 'Filtered (Valid Types)', 'Delta', 'Delta %'])
        writer.writerow(['Precision', f"{metrics_all['precision']:.4f}", f"{metrics_filtered['precision']:.4f}",
                        f"{precision_delta:+.4f}", f"{precision_delta/metrics_all['precision']*100:+.2f}%"])
        writer.writerow(['Recall', f"{metrics_all['recall']:.4f}", f"{metrics_filtered['recall']:.4f}",
                        f"{recall_delta:+.4f}", f"{recall_delta/metrics_all['recall']*100:+.2f}%"])
        writer.writerow(['F1 Score', f"{metrics_all['f1']:.4f}", f"{metrics_filtered['f1']:.4f}",
                        f"{f1_delta:+.4f}", f"{f1_delta/metrics_all['f1']*100:+.2f}%"])
        writer.writerow([])
        writer.writerow(['TP', metrics_all['tp'], metrics_filtered['tp'], metrics_filtered['tp'] - metrics_all['tp']])
        writer.writerow(['FP', metrics_all['fp'], metrics_filtered['fp'], metrics_filtered['fp'] - metrics_all['fp']])
        writer.writerow(['FN', metrics_all['fn'], metrics_filtered['fn'], metrics_filtered['fn'] - metrics_all['fn']])
        writer.writerow(['Total Predictions', metrics_all['total_predictions'], metrics_filtered['total_predictions']])
        writer.writerow([])

        # Per-relation metrics comparison table
        writer.writerow(['PER-RELATION METRICS COMPARISON'])
        writer.writerow(['Relation', 'Baseline_Precision', 'Baseline_Recall', 'Baseline_F1',
                        'Filtered_Precision', 'Filtered_Recall', 'Filtered_F1',
                        'Delta_F1', 'TP', 'FP', 'FN', 'Support'])

        # Sort by filtered F1 score
        sorted_relations = sorted(per_rel_filtered['per_relation'].items(),
                                 key=lambda x: x[1]['f1'], reverse=True)

        for rel, metrics_f in sorted_relations:
            metrics_b = per_rel_all['per_relation'].get(rel, {
                'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'support': 0
            })
            delta_f1 = metrics_f['f1'] - metrics_b['f1']

            writer.writerow([
                rel,
                f"{metrics_b['precision']:.4f}",
                f"{metrics_b['recall']:.4f}",
                f"{metrics_b['f1']:.4f}",
                f"{metrics_f['precision']:.4f}",
                f"{metrics_f['recall']:.4f}",
                f"{metrics_f['f1']:.4f}",
                f"{delta_f1:+.4f}",
                metrics_f['tp'],
                metrics_f['fp'],
                metrics_f['fn'],
                metrics_f['support']
            ])

        # Add macro/micro rows
        writer.writerow([])
        writer.writerow(['MACRO AVERAGE',
                        f"{per_rel_all['macro_avg']['precision']:.4f}",
                        f"{per_rel_all['macro_avg']['recall']:.4f}",
                        f"{per_rel_all['macro_avg']['f1']:.4f}",
                        f"{per_rel_filtered['macro_avg']['precision']:.4f}",
                        f"{per_rel_filtered['macro_avg']['recall']:.4f}",
                        f"{per_rel_filtered['macro_avg']['f1']:.4f}",
                        f"{per_rel_filtered['macro_avg']['f1'] - per_rel_all['macro_avg']['f1']:+.4f}",
                        '', '', '', ''])
        writer.writerow(['MICRO AVERAGE',
                        f"{per_rel_all['micro_avg']['precision']:.4f}",
                        f"{per_rel_all['micro_avg']['recall']:.4f}",
                        f"{per_rel_all['micro_avg']['f1']:.4f}",
                        f"{per_rel_filtered['micro_avg']['precision']:.4f}",
                        f"{per_rel_filtered['micro_avg']['recall']:.4f}",
                        f"{per_rel_filtered['micro_avg']['f1']:.4f}",
                        f"{per_rel_filtered['micro_avg']['f1'] - per_rel_all['micro_avg']['f1']:+.4f}",
                        per_rel_filtered['micro_avg']['tp'],
                        per_rel_filtered['micro_avg']['fp'],
                        per_rel_filtered['micro_avg']['fn'],
                        ''])
        writer.writerow([])

        writer.writerow(['VIOLATIONS BY RELATION'])
        writer.writerow(['Relation', 'Violation Count'])
        for relation, count in sorted_violations:
            writer.writerow([relation, count])

    # Save violations to JSON
    print(f"7. Saving violations to: {args.violations_json}")
    with open(args.violations_json, 'w', encoding='utf-8') as f:
        json.dump({
            'total_violations': len(violations),
            'violation_rate': len(violations) / len(predictions),
            'violations': violations,
            'statistics': violation_stats,
            'metrics': {
                'baseline': {
                    'overall': metrics_all,
                    'per_relation': per_rel_all['per_relation'],
                    'micro_avg': per_rel_all['micro_avg'],
                    'macro_avg': per_rel_all['macro_avg']
                },
                'filtered': {
                    'overall': metrics_filtered,
                    'per_relation': per_rel_filtered['per_relation'],
                    'micro_avg': per_rel_filtered['micro_avg'],
                    'macro_avg': per_rel_filtered['macro_avg']
                }
            }
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  - {args.output_csv}")
    print(f"  - {args.violations_json}")


if __name__ == "__main__":
    main()
