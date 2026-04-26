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
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
from dreeam_inference import DreeamInference
import argparse


class AnnotationRuleParser:
    """Parse NEREL annotation.conf [relations] section into {relation: set of (h_type, t_type) pairs}.

    A relation can have multiple lines in the conf; each line adds (Arg1 type) x (Arg2 type)
    pairs to that relation's allowed set. This preserves cross-type vs same-type structure
    (e.g. ABBREVIATION's 8 same-type lines do NOT mean cross-type combos are valid).
    """

    def __init__(self, conf_path: str, verbose: bool = False):
        self.relation_rules: Dict[str, Set[Tuple[str, str]]] = {}
        self.malformed_lines: List[Tuple[int, str]] = []
        in_relations = False
        with open(conf_path, 'r', encoding='utf-8') as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.strip()
                if line == '[relations]':
                    in_relations = True
                    continue
                if line.startswith('[') and in_relations:
                    break
                if not in_relations or not line or line.startswith('#'):
                    continue
                # Skip the special <OVERLAP> entry — it uses <ENTITY> and isn't a real relation
                if line.startswith('<'):
                    continue
                self._parse_line(line, lineno)

        if verbose:
            self.print_summary()

    def _parse_line(self, line: str, lineno: int):
        # Split into (relation_name, args_string) on first whitespace gap.
        # This works for both tab-separated and multi-space-separated lines.
        m = re.match(r'^(\S+)\s+(\S.*)$', line)
        if not m:
            self.malformed_lines.append((lineno, line))
            return
        rel_name = m.group(1).strip()
        args_str = m.group(2).strip()
        arg_parts = args_str.split(',')
        if len(arg_parts) != 2:
            self.malformed_lines.append((lineno, line))
            return
        arg1_match = re.search(r'Arg1:(.+)', arg_parts[0])
        arg2_match = re.search(r'Arg2:(.+)', arg_parts[1])
        if not arg1_match or not arg2_match:
            self.malformed_lines.append((lineno, line))
            return
        arg1_types = self._parse_type_list(arg1_match.group(1).strip())
        arg2_types = self._parse_type_list(arg2_match.group(1).strip())
        pairs = self.relation_rules.setdefault(rel_name, set())
        for a1 in arg1_types:
            for a2 in arg2_types:
                pairs.add((a1, a2))

    def _parse_type_list(self, s: str) -> Set[str]:
        if '<ENTITY>' in s or '<ANY>' in s:
            return {'<ANY>'}
        out = set()
        for t in s.split('|'):
            t = t.strip()
            if ':' in t:
                t = t.split(':')[1]
            if t:
                out.add(t)
        return out

    def is_valid(self, relation: str, h_type: str, t_type: str) -> bool:
        """True if this (h_type, relation, t_type) combination is allowed by the schema."""
        if relation not in self.relation_rules:
            return True  # unknown relation — don't filter
        pairs = self.relation_rules[relation]
        return (
            (h_type, t_type) in pairs
            or ('<ANY>', t_type) in pairs
            or (h_type, '<ANY>') in pairs
            or ('<ANY>', '<ANY>') in pairs
        )

    def print_summary(self):
        """Print all parsed rules — useful for verifying the conf was loaded correctly."""
        print("=" * 80)
        print(f"Parsed {len(self.relation_rules)} relation types:")
        print("=" * 80)
        for rel in sorted(self.relation_rules.keys()):
            pairs = sorted(self.relation_rules[rel])
            print(f"\n{rel}  ({len(pairs)} allowed type-pairs):")
            for h, t in pairs:
                print(f"  [{h} → {t}]")
        if self.malformed_lines:
            print("\n" + "=" * 80)
            print(f"WARNING: {len(self.malformed_lines)} malformed line(s) skipped:")
            for ln, content in self.malformed_lines:
                print(f"  line {ln}: {content!r}")
        print("=" * 80)


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
            # Skip discontinuous spans (BRAT syntax: "121;144 154 200")
            if ';' in entity_info[1] or ';' in entity_info[2]:
                return
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
        
    def add_predictions(self, gold_relations: Set[Tuple], pred_relations: Set[Tuple], file_id: str = None, text: str = None, entity_types_by_text: Dict[str, str] = None):
        """
        Add predictions for evaluation.

        Args:
            gold_relations: Set of gold relation tuples (head_text, tail_text, relation_type)
            pred_relations: Set of predicted relation tuples (head_text, tail_text, relation_type)
            file_id: File identifier for tracking predictions
            text: Original text for context
            entity_types_by_text: Optional {entity_text: entity_type} lookup for readable error reports
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
            'entity_types_by_text': entity_types_by_text or {},
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

        # Skip mention-merge relations (not predicted by the model)
        if rel_type in ('ABBREVIATION', 'ALTERNATIVE_NAME'):
            continue

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
            
            # Build {entity_text: entity_type} lookup for readable error report
            entity_types_by_text = {e['text']: e['type'] for e in entities}

            # Evaluate
            evaluator.add_predictions(gold_relations, pred_relations, base_name, text, entity_types_by_text)
            
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


def save_error_analysis_txt(evaluator: NERELEvaluator, output_path: str,
                            rules: Optional['AnnotationRuleParser'] = None):
    """
    Save error analysis in a human-readable text format, grouped per file
    and within each file by (head_type → tail_type, relation_type).

    If `rules` is provided:
      - FALSE POSITIVES are split into "schema-valid" (review-worthy) and
        "schema-violating" (clearly model errors, hidden in summary count).
      - GOLD VIOLATIONS are listed separately — gold relations whose type
        combination violates the schema (likely annotation errors).
    """
    UNKNOWN_TYPE = '?'

    def group_relations(rel_set, types_by_text):
        # group: {(head_type, tail_type, rel_type): [(head_text, tail_text), ...]}
        groups = defaultdict(list)
        for head, tail, rel_type in rel_set:
            ht = types_by_text.get(head, UNKNOWN_TYPE)
            tt = types_by_text.get(tail, UNKNOWN_TYPE)
            groups[(ht, tt, rel_type)].append((head, tail))
        return groups

    def write_section(f, title, rel_set, types_by_text):
        f.write(f"\n>>> {title} — {len(rel_set)} total\n\n")
        if not rel_set:
            f.write("  (none)\n")
            return
        groups = group_relations(rel_set, types_by_text)
        # sort groups: by relation, then head_type, then tail_type
        for (ht, tt, rel_type) in sorted(groups.keys(), key=lambda k: (k[2], k[0], k[1])):
            instances = sorted(groups[(ht, tt, rel_type)])
            head_pad = max(len(h) for h, _ in instances)
            f.write(f"  [{ht} → {tt}]   {rel_type}\n")
            for head, tail in instances:
                f.write(f"    {head:<{head_pad}}   ->   {tail}\n")
            f.write("\n")

    def partition_by_validity(rel_set, types_by_text):
        """Return (valid_set, invalid_set) per the rules."""
        if rules is None:
            return rel_set, set()
        valid, invalid = set(), set()
        for head, tail, rel_type in rel_set:
            ht = types_by_text.get(head, UNKNOWN_TYPE)
            tt = types_by_text.get(tail, UNKNOWN_TYPE)
            if ht == UNKNOWN_TYPE or tt == UNKNOWN_TYPE:
                valid.add((head, tail, rel_type))  # can't judge -> keep
            elif rules.is_valid(rel_type, ht, tt):
                valid.add((head, tail, rel_type))
            else:
                invalid.add((head, tail, rel_type))
        return valid, invalid

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("FP/FN ERROR ANALYSIS — per file, grouped by (head_type → tail_type, relation)\n")
        f.write("Each `===` line marks the start of a new file's section.\n")
        if rules is not None:
            f.write("\nSchema filtering ON — predictions violating annotation.conf type constraints\n")
            f.write("are dropped from FALSE POSITIVES (clearly model errors, not annotation gaps).\n")
            f.write("Gold relations violating type constraints are listed under GOLD VIOLATIONS\n")
            f.write("(likely real annotation errors that should be reviewed).\n")

        for prediction in evaluator.detailed_predictions:
            file_id = prediction['file_id']
            tp = prediction['metrics']['tp_count']
            fp = prediction['metrics']['fp_count']
            fn = prediction['metrics']['fn_count']
            types_by_text = prediction.get('entity_types_by_text', {})

            fps = set(map(tuple, prediction['false_positives']))
            fns = set(map(tuple, prediction['false_negatives']))
            golds = set(map(tuple, prediction['gold_relations']))

            fps_valid, fps_invalid = partition_by_validity(fps, types_by_text)
            golds_valid, golds_invalid = partition_by_validity(golds, types_by_text)

            # If schema filtering on, FNs that violate schema are also surfaced
            # (those are gold errors, not model misses)
            fns_for_review = fns - golds_invalid if rules is not None else fns

            review_fp = len(fps_valid)
            review_fn = len(fns_for_review)
            n_gold_violations = len(golds_invalid)
            n_dropped_fp = len(fps_invalid)

            # Skip files with nothing to review
            if review_fp == 0 and review_fn == 0 and n_gold_violations == 0:
                continue

            f.write("\n\n" + "=" * 80 + "\n")
            header = f"FILE: {file_id}     (TP={tp}, FP={fp}, FN={fn}"
            if rules is not None:
                header += f"  |  review_FP={review_fp}, review_FN={review_fn}, gold_violations={n_gold_violations}, dropped_FP={n_dropped_fp}"
            header += ")"
            f.write(header + "\n")

            write_section(f, "FALSE POSITIVES (model predicted, not in gold; schema-valid)" if rules else "FALSE POSITIVES (model predicted, not in gold)",
                          fps_valid, types_by_text)
            write_section(f, "FALSE NEGATIVES (in gold, model missed)",
                          fns_for_review, types_by_text)
            if rules is not None and n_gold_violations > 0:
                write_section(f, "GOLD VIOLATIONS (gold relations violating annotation.conf — review)",
                              golds_invalid, types_by_text)


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
    parser.add_argument('--errors_txt', default='nerel_error_analysis.txt',
                      help='Output TXT file with human-readable FP/FN breakdown grouped by type-pair + relation')
    parser.add_argument('--annotation_conf', default=None,
                      help='Path to annotation.conf — drop schema-violating FPs and flag gold violations')
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

            # Build {entity_text: entity_type} lookup for readable error report
            entity_types_by_text = {e['text']: e['type'] for e in entities}

            # Evaluate
            evaluator.add_predictions(gold_relations, pred_relations, base_name, text, entity_types_by_text)

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

    # Save human-readable error analysis TXT (optionally schema-filtered)
    rules = None
    if args.annotation_conf:
        print(f"\nLoading annotation rules from {args.annotation_conf}")
        rules = AnnotationRuleParser(args.annotation_conf, verbose=True)
        print(f"Loaded {len(rules.relation_rules)} relation rules.")
    save_error_analysis_txt(evaluator, args.errors_txt, rules=rules)

    print(f"\nDetailed results saved to: {args.output_file}")
    print(f"CSV results saved to: {args.csv_output}")
    print(f"Detailed predictions saved to: {args.predictions_output}")
    print(f"Error analysis CSV saved to: {args.errors_csv}")
    print(f"Readable error analysis saved to: {args.errors_txt}")
    print(f"\nFor manual error analysis:")
    print(f"- Open {args.errors_txt} for grouped, human-readable FP/FN inspection")
    print(f"- Open {args.errors_csv} to filter/sort all FP/FN in Excel")
    print(f"- Open {args.predictions_output} for detailed per-file predictions")


if __name__ == "__main__":
    main() 