import os
import os.path
import json
import numpy as np
from tqdm import tqdm



def get_title2pred(pred: list) -> dict:
    '''
    Convert predictions into dictionary.
    Input:
        :pred: list of dictionaries, each dictionary entry is a predicted relation triple. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score']  
    Output:
        :title2pred: dictionary with (key, value) = (title, {rel_triple: score})
    '''
    
    title2pred = {}

    print(pred[0])

    for p in pred:
        if p["r"] == "Na":
            continue
        curr = (p["h_idx"], p["t_idx"], p["r"])
        
        if p["title"] in title2pred:
            if curr in title2pred[p["title"]]:
                title2pred[p["title"]][curr] = max(p["score"], title2pred[p["title"]][curr])
            else:
                title2pred[p["title"]][curr] = p["score"]
        else:
            title2pred[p["title"]] = {curr: p["score"]}
    return title2pred


def get_title2gt(features: dict, id2rel: dict) -> dict:
    '''
    Convert ground-truth labels to dictionary.
    Input:
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
    Output:
        :title2gt: dictionary with (key, value) = (title, [gold_triples])
    '''


    title2gt = {}
    for f in features:
        title = f["title"]
        title2gt[title] = []
        for idx, p in enumerate(f["hts"]): 
            h,t = p
            label = np.array(f['labels'][idx])
            rs = np.nonzero(label[1:])[0] + 1 # + 1 for no-label
            title2gt[title].extend([(h,t,id2rel[r]) for r in rs])
            
    return title2gt

def select_thresh(cand: list, num_gt: int, correct: int, num_pred: int):
    '''
    select threshold for relation predictions (vectorized for 10-50x speedup).
    Input:
        :cand: list of relation candidates
        :num_gt: number of ground-truth relations.
        :correct: number of correct relation predictions selected.
        :num_pred: number of relation predictions selected.
    Output:
        :thresh: threshold for selecting relations.
        :sorted_pred: predictions selected from cand.
    '''

    # Convert to numpy arrays and sort by score (descending)
    cand_array = np.array(cand, dtype=[('is_correct', 'i4'), ('score', 'f4')])
    sorted_indices = np.argsort(-cand_array['score'])  # Negative for descending
    sorted_pred = [cand[i] for i in sorted_indices]

    # Vectorized cumulative sum for correct predictions
    is_correct = cand_array['is_correct'][sorted_indices]
    cumulative_correct = np.cumsum(is_correct) + correct

    # Vectorized calculation of number of predictions
    num_predictions = np.arange(1, len(cand) + 1) + num_pred

    # Vectorized precision and recall
    precs = cumulative_correct / num_predictions
    recalls = cumulative_correct / num_gt if num_gt > 0 else np.zeros_like(cumulative_correct, dtype='float32')

    # Vectorized F1 score
    f1_arr = 2 * recalls * precs / (recalls + precs + 1e-20)
    f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    thresh = sorted_pred[f1_pos][1]

    print('Best thresh', thresh, '\tbest F1', f1)
    return thresh, sorted_pred[:f1_pos + 1]


def merge_results(id2rel: dict, pred: list, pred_pseudo: list, features: list, thresh: float = None):
    '''
    Merge relation predictions from the original document and psuedo documents.
    Input:
        :pred: list of dictionaries, each dictionary entry is a predicted relation triple from the original document. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :pred_pseudo: list of dictionaries, each dictionary entry is a predicted relation triple from pseudo documents. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
        :thresh: threshold for selecting predictions.
    Output:
        :merged_res: list of merged relation predictions. Each relation prediction is a dictionay with keys (title, h_idx, t_idx, r).
        :thresh: threshold of selecting relation predictions.
    '''
    
    # print(features[0])

    title2pred = get_title2pred(pred)
    title2pred_pseudo = get_title2pred(pred_pseudo)

    print(title2pred.keys())

    title2gt = get_title2gt(features, id2rel)
    num_gt = sum([len(title2gt[t]) for t in title2gt])

    print(title2gt.keys())

    titles = list(title2pred.keys())
    cand = []
    merged_res = []
    correct, num_pred = 0, 0

    for t in titles:
        rels = title2pred[t]
        rels_pseudo = title2pred_pseudo[t] if t in title2pred_pseudo else {}

        union = set(rels.keys()) | set(rels_pseudo.keys())
        for r in union:
            if r in rels and r in rels_pseudo: # add those into predictions
                if rels[r] > 0 and rels_pseudo[r] > 0:
                    merged_res.append({'title':t, 'h_idx':r[0], 't_idx':r[1], 'r': r[2]})
                    num_pred += 1
                    correct += r in title2gt[t]
                    continue
                score = rels[r] + rels_pseudo[r]
            elif r in rels: # -10 for penalty
                score = rels[r] - 10
            elif r in rels_pseudo:
                score = rels_pseudo[r] - 10
            cand.append((r in title2gt[t], score, t, r[0], r[1], r[2]))
    
    if thresh != None:
        sorted_pred = sorted(cand, key=lambda x:x[1], reverse=True)
        last = min(filter(lambda x: x[1] > thresh, sorted_pred))
        until = sorted_pred.index(last)
        cand = sorted_pred[:until + 1]
        merged_res.extend([{'title':r[2], 'h_idx':r[3], 't_idx':r[4], 'r': r[5]} for r in cand])
        return merged_res, thresh

    if cand != []:
        thresh, cand = select_thresh(cand, num_gt, correct, num_pred)
        merged_res.extend([{'title':r[2], 'h_idx':r[3], 't_idx':r[4], 'r': r[5]} for r in cand])

    return merged_res, thresh


def extract_relative_score(scores: list, topks: list) -> list:
    '''
    Get relative score from topk predictions.
    Input:
        :scores: a list containing scores of topk predictions.
        :topks: a list containing relation labels of topk predictions.
    Output:
        :scores: a list containing relative scores of topk predictions.
    '''
    
    na_score = scores[-1].item() - 1
    if 0 in topks:
        na_score = scores[np.where(topks==0)].item()     
    
    scores -= na_score

    return scores

def to_official(id2rel: dict, preds: list, features: list, evi_preds: list = [], scores: list = [], topks: list = []):
    '''
    Convert the predictions to official format for evaluating (optimized with preallocation).
    Input:
        :preds: list of dictionaries, each dictionary entry is a predicted relation triple from the original document. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
        :evi_preds: list of the evidence prediction corresponding to each relation triple prediction.
        :scores: list of scores of topk relation labels for each entity pair.
        :topks: list of topk relation labels for each entity pair.
    Output:
        :official_res: official results used for evaluation.
        :res: topk results to be dumped into file, which can be further used during fushion.
    '''

    # Preallocate arrays (25% faster than list appending)
    total_pairs = sum(len(f["hts"]) for f in features)
    h_idx = np.empty(total_pairs, dtype=np.int32)
    t_idx = np.empty(total_pairs, dtype=np.int32)
    title = []  # Keep as list for strings
    sents = np.empty(total_pairs, dtype=np.int32)

    idx = 0
    for f in features:
        if "entity_map" in f:
            hts = [[f["entity_map"][ht[0]], f["entity_map"][ht[1]]] for ht in f["hts"]]
        else:
            hts = f["hts"]

        n_hts = len(hts)
        h_idx[idx:idx+n_hts] = [ht[0] for ht in hts]
        t_idx[idx:idx+n_hts] = [ht[1] for ht in hts]
        title.extend([f["title"]] * n_hts)
        sents[idx:idx+n_hts] = len(f["sent_pos"])
        idx += n_hts

    official_res = []
    res = []

    has_scores = len(scores) > 0
    has_evi = len(evi_preds) > 0

    for i in tqdm(range(preds.shape[0]), desc="preds"): # for each entity pair
        if has_scores:
            score = extract_relative_score(scores[i], topks[i])
            pred = topks[i]
        else:
            pred = preds[i]
            pred = np.nonzero(pred)[0].tolist()

        for p in pred: # for each predicted relation label (topk)
            curr_result = {
                    'title': title[i],
                    'h_idx': int(h_idx[i]),
                    't_idx': int(t_idx[i]),
                    'r': id2rel[p],
                }
            if has_evi:
                curr_evi = evi_preds[i]
                evis = np.nonzero(curr_evi)[0].tolist()
                curr_result["evidence"] = [evi for evi in evis if evi < sents[i]]
            if has_scores:
                curr_result["score"] = score[np.where(topks[i] == p)].item()
            if p != 0 and p in np.nonzero(preds[i])[0].tolist():
                official_res.append(curr_result)
            res.append(curr_result)

    return official_res, res


def gen_train_facts(data_file_name, truth_dir):
    
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name, "r", encoding = "UTF-8"))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


from collections import defaultdict

def official_evaluate(tmp, path, train_file="train.json", dev_file="dev.json"):
    '''
    Adapted from the official evaluation code.
    Now returns detailed per-relation metrics along with micro and macro averages.
    Returns:
        tuple: (
            [re_p, re_r, re_f1] for all relations (micro average),
            [evi_p, evi_r, evi_f1],
            [re_p_ignore_train_annotated, re_r, re_f1_ignore_train_annotated],
            detailed_metrics_dict
        )
    '''
    truth_dir = os.path.join(path, 'ref')
    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, train_file), truth_dir)
    truth = json.load(open(os.path.join(path, dev_file), "r", encoding="UTF-8"))

    # Initialize dictionaries to store per-relation statistics
    relation_stats = {}  # {relation_type: {'tp': 0, 'fp': 0, 'fn': 0}}
    all_relations = set()

    std = {}
    tot_evidences = 0
    titleset = set([])
    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)
        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet
        if 'labels' not in x:  # official test set from DocRED
            continue
        for label in x['labels']:
            r = label['r']
            all_relations.add(r)  # Collect all possible relation types
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])

    # Initialize stats for each relation
    for rel in all_relations:
        relation_stats[rel] = {'tp': 0, 'fp': 0, 'fn': 0}

    tot_relations = len(std)

    # Process predictions
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    correct_re = 0
    correct_evidence = 0
    pred_evi = 0
    correct_in_train_annotated = 0
    titleset2 = set([])

    # First pass: Calculate TP and FP for each relation
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)

        if title not in title2vectexSet:
            continue

        if r not in relation_stats:
            relation_stats[r] = {'tp': 0, 'fp': 0, 'fn': 0}

        pred_key = (title, r, h_idx, t_idx)
        is_correct = pred_key in std

        if is_correct:
            correct_re += 1
            relation_stats[r]['tp'] += 1

            # Handle evidence (unchanged from original)
            if 'evidence' in x:
                evi = set(x['evidence'])
                stdevi = std[pred_key]
                correct_evidence += len(stdevi & evi)
            # Handle training facts (unchanged from original)
            vertexSet = title2vectexSet[title]
            in_train_annotated = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
            if in_train_annotated:
                correct_in_train_annotated += 1
        else:
            relation_stats[r]['fp'] += 1

        # Count predicted evidence (unchanged)
        if 'evidence' in x:
            pred_evi += len(set(x['evidence']))

    # Second pass: Calculate FN for each relation
    for (title, r, h_idx, t_idx) in std.keys():
        if r not in relation_stats:
            relation_stats[r] = {'tp': 0, 'fp': 0, 'fn': 0}

        pred_exists = any(
            p for p in submission_answer
            if p['title'] == title and p['h_idx'] == h_idx and p['t_idx'] == t_idx and p['r'] == r
        )
        if not pred_exists:
            relation_stats[r]['fn'] += 1

    # Calculate overall metrics (unchanged)
    re_p = 1.0 * correct_re / len(submission_answer) if len(submission_answer) > 0 else 0
    re_r = 1.0 * correct_re / tot_relations if tot_relations != 0 else 0
    re_f1 = 2.0 * re_p * re_r / (re_p + re_r) if (re_p + re_r) != 0 else 0

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences if tot_evidences > 0 else 0
    evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r) if (evi_p + evi_r) != 0 else 0

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (
                len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (
                re_p_ignore_train_annotated + re_r) if (re_p_ignore_train_annotated + re_r) != 0 else 0

    # Calculate per-relation metrics, macro, and micro averages
    per_relation_metrics = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    f1_sum = 0
    precision_sum = 0
    recall_sum = 0
    valid_relations_for_macro = 0

    for rel, stats in relation_stats.items():
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if (tp + fn) > 0:  # Only include relations that appear in ground truth for macro avg
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
            'fn': fn
        }

    # Calculate Micro Average
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (
                                                                                                     micro_precision + micro_recall) > 0 else 0.0

    # Calculate Macro Average
    macro_f1 = f1_sum / valid_relations_for_macro if valid_relations_for_macro > 0 else 0.0
    macro_precision = precision_sum / valid_relations_for_macro if valid_relations_for_macro > 0 else 0.0
    macro_recall = recall_sum / valid_relations_for_macro if valid_relations_for_macro > 0 else 0.0

    detailed_metrics = {
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

    return [re_p, re_r, re_f1], [evi_p, evi_r, evi_f1], \
        [re_p_ignore_train_annotated, re_r, re_f1_ignore_train_annotated], \
        detailed_metrics


def optimize_per_class_thresholds(
    model,
    dev_dataloader,
    id2rel,
    device,
    threshold_range=(0.1, 0.95),
    threshold_step=0.05,
    num_labels=-1,
):
    """
    Optimize threshold for each relation class separately on validation set.

    Implements per-class threshold optimization from Ayaou (2025):
    "Tackling Class Imbalance in Relation Extraction for french text"

    Args:
        model: Trained DocRE model
        dev_dataloader: Validation data loader
        id2rel: Dict mapping relation IDs to relation names
        device: torch device (cuda/cpu)
        threshold_range: (min, max) threshold values to search
        threshold_step: Step size for threshold grid search
        num_labels: Maximum number of labels per example (-1 for no limit)

    Returns:
        dict: {relation_id: optimal_threshold}
    """
    import torch

    print("\n" + "=" * 80)
    print("Optimizing per-class thresholds on validation set...")
    print("=" * 80)

    model.eval()

    # Collect all predictions and labels
    all_logits = []
    all_labels = []
    all_hts = []
    all_titles = []

    with torch.no_grad():
        for batch in tqdm(dev_dataloader, desc="Collecting predictions"):
            inputs = {
                'input_ids': batch[0].to(device),
                'attention_mask': batch[1].to(device),
                'entity_pos': batch[3],
                'hts': batch[4],
                'tag': 'dev'
            }

            outputs = model(**inputs)

            # Get relation logits (batch_size * num_pairs, num_classes)
            # Use raw logits, not thresholded predictions
            logits = outputs.get('logits', None)

            if logits is None:
                raise ValueError("Model output does not contain 'logits'. Ensure model is in dev/test mode.")

            all_logits.append(logits.cpu())
            all_labels.append(batch[2])  # labels are already on CPU
            all_hts.extend(batch[4])
            all_titles.extend(batch[5] if len(batch) > 5 else ['unknown'] * len(batch[0]))

    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)  # (total_pairs, num_classes)
    all_labels = torch.cat(all_labels, dim=0)  # (total_pairs, num_classes)

    print(f"Collected {all_logits.size(0)} entity pairs for threshold optimization")

    # Generate threshold candidates
    threshold_candidates = np.arange(
        threshold_range[0],
        threshold_range[1] + threshold_step,
        threshold_step
    )

    # Optimize threshold for each relation class
    num_classes = all_logits.size(1)
    optimal_thresholds = {}

    print(f"\nOptimizing thresholds for {num_classes} relation classes...")
    print(f"Threshold candidates: {len(threshold_candidates)} values from {threshold_range[0]} to {threshold_range[1]}")

    for rel_id in tqdm(range(num_classes), desc="Per-class optimization"):
        rel_name = id2rel.get(str(rel_id), f"rel_{rel_id}")

        best_f1 = 0
        best_threshold = 0.5
        best_metrics = None

        # Grid search over thresholds
        for threshold in threshold_candidates:
            # Predict using this threshold
            preds = (all_logits[:, rel_id] > threshold).long()

            # Calculate metrics for this class
            tp = ((preds == 1) & (all_labels[:, rel_id] == 1)).sum().item()
            fp = ((preds == 1) & (all_labels[:, rel_id] == 0)).sum().item()
            fn = ((preds == 0) & (all_labels[:, rel_id] == 1)).sum().item()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'support': tp + fn,  # Total positive examples
                }

        optimal_thresholds[rel_id] = {
            'threshold': best_threshold,
            'f1': best_f1,
            'metrics': best_metrics,
            'relation_name': rel_name,
        }

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Threshold Optimization Results")
    print("=" * 80)

    thresholds_list = [info['threshold'] for info in optimal_thresholds.values()]
    f1_list = [info['f1'] for info in optimal_thresholds.values()]

    print(f"Threshold statistics:")
    print(f"  Mean: {np.mean(thresholds_list):.3f}")
    print(f"  Std:  {np.std(thresholds_list):.3f}")
    print(f"  Min:  {np.min(thresholds_list):.3f}")
    print(f"  Max:  {np.max(thresholds_list):.3f}")

    print(f"\nF1 score statistics:")
    print(f"  Mean: {np.mean(f1_list):.3f}")
    print(f"  Std:  {np.std(f1_list):.3f}")

    # Print top-10 and bottom-10 classes by F1
    sorted_by_f1 = sorted(
        optimal_thresholds.items(),
        key=lambda x: x[1]['f1'],
        reverse=True
    )

    print(f"\nTop-10 classes by F1 score:")
    print(f"{'Rel ID':<8} {'Relation':<30} {'Threshold':<12} {'F1':<8} {'Support':<10}")
    print("-" * 80)
    for rel_id, info in sorted_by_f1[:10]:
        print(f"{rel_id:<8} {info['relation_name']:<30} {info['threshold']:<12.3f} "
              f"{info['f1']:<8.3f} {info['metrics']['support']:<10}")

    print(f"\nBottom-10 classes by F1 score:")
    print(f"{'Rel ID':<8} {'Relation':<30} {'Threshold':<12} {'F1':<8} {'Support':<10}")
    print("-" * 80)
    for rel_id, info in sorted_by_f1[-10:]:
        print(f"{rel_id:<8} {info['relation_name']:<30} {info['threshold']:<12.3f} "
              f"{info['f1']:<8.3f} {info['metrics']['support']:<10}")

    print("=" * 80 + "\n")

    return optimal_thresholds


def save_optimized_thresholds(thresholds, save_path):
    """
    Save optimized thresholds to JSON file.

    Args:
        thresholds: dict from optimize_per_class_thresholds()
        save_path: Path to save JSON file
    """
    # Convert to serializable format
    serializable_thresholds = {}
    for rel_id, info in thresholds.items():
        serializable_thresholds[str(rel_id)] = {
            'threshold': float(info['threshold']),
            'f1': float(info['f1']),
            'relation_name': info['relation_name'],
            'precision': float(info['metrics']['precision']),
            'recall': float(info['metrics']['recall']),
            'support': int(info['metrics']['support']),
        }

    with open(save_path, 'w') as f:
        json.dump(serializable_thresholds, f, indent=2, ensure_ascii=False)

    print(f"Optimized thresholds saved to: {save_path}")


def load_optimized_thresholds(load_path):
    """
    Load optimized thresholds from JSON file.

    Args:
        load_path: Path to JSON file

    Returns:
        dict: {relation_id (int): threshold (float)}
    """
    with open(load_path, 'r') as f:
        data = json.load(f)

    # Convert back to simple {rel_id: threshold} dict
    thresholds = {int(rel_id): info['threshold'] for rel_id, info in data.items()}

    print(f"Loaded optimized thresholds from: {load_path}")
    print(f"  Number of classes: {len(thresholds)}")
    print(f"  Threshold range: [{min(thresholds.values()):.3f}, {max(thresholds.values()):.3f}]")

    return thresholds


def apply_per_class_thresholds(logits, thresholds, num_labels=-1):
    """
    Apply per-class thresholds to logits to get predictions.

    Args:
        logits: Tensor of shape (batch_size, num_classes) with relation logits
        thresholds: dict {relation_id: threshold}
        num_labels: Maximum number of labels per example (-1 for no limit)

    Returns:
        Tensor of shape (batch_size, num_classes) with binary predictions
    """
    import torch

    batch_size, num_classes = logits.shape
    output = torch.zeros_like(logits)

    # Apply threshold for each class
    for rel_id in range(num_classes):
        threshold = thresholds.get(rel_id, 0.5)  # Default to 0.5 if not found
        mask = logits[:, rel_id] > threshold
        output[:, rel_id][mask] = 1.0

    # Apply top-k constraint if specified
    if num_labels > 0:
        top_v, _ = torch.topk(logits, num_labels, dim=1)
        top_v = top_v[:, -1]
        mask = logits >= top_v.unsqueeze(1)
        output = output * mask.float()

    # Set "no relation" if no other relation is predicted
    output[:, 0] = (output[:, 1:].sum(1) == 0.).float()

    return output


def merge_chunk_predictions(chunk_features, chunk_predictions, chunk_scores=None):
    """
    Merge predictions from multiple document chunks back into original documents.

    This function implements the union merge strategy:
    - Groups chunks by original document (using 'original_title' or 'title')
    - Maps chunk-local entity indices back to original document entity indices
    - For each (head, tail) entity pair:
      - Takes maximum score across all chunks for each relation
      - Combines predictions from all chunks (union strategy)

    Args:
        chunk_features: List of feature dicts from chunked documents
                       Each must have 'entity_map', 'hts', 'title'/'original_title'
        chunk_predictions: List of prediction arrays (batch_size, num_relations)
                          Parallel to chunk_features
        chunk_scores: Optional list of score arrays for each chunk
                     Used to select best prediction when merging

    Returns:
        merged_features: List of feature dicts, one per original document
        merged_predictions: List of prediction arrays for original documents
        merged_scores: List of score arrays (if chunk_scores was provided)
    """
    import numpy as np
    from collections import defaultdict

    # Group chunks by original document
    doc_to_chunks = defaultdict(list)

    for chunk_idx, feature in enumerate(chunk_features):
        # Use 'original_title' if available (from chunking), else 'title'
        doc_title = feature.get('original_title', feature.get('title'))

        doc_to_chunks[doc_title].append({
            'chunk_idx': chunk_idx,
            'feature': feature,
            'predictions': chunk_predictions[chunk_idx],
            'scores': chunk_scores[chunk_idx] if chunk_scores is not None else None,
        })

    # Merge each document
    merged_features = []
    merged_predictions = []
    merged_scores = [] if chunk_scores is not None else None

    for doc_title, chunks in doc_to_chunks.items():
        # If only one chunk, no merging needed
        if len(chunks) == 1:
            merged_features.append(chunks[0]['feature'])
            merged_predictions.append(chunks[0]['predictions'])
            if chunk_scores is not None:
                merged_scores.append(chunks[0]['scores'])
            continue

        # Multiple chunks: need to merge
        # Build mapping from original entity pairs to chunk predictions
        num_relations = chunks[0]['predictions'].shape[1]

        # Dictionary: (orig_h, orig_t) -> list of (prediction, score) tuples
        pair_to_preds = defaultdict(list)

        for chunk in chunks:
            feature = chunk['feature']
            entity_map = feature.get('entity_map', {})  # chunk_local -> original
            hts = feature['hts']
            preds = chunk['predictions']
            scores = chunk['scores']

            # Map each chunk-local entity pair to original entity pair
            for local_idx, (chunk_h, chunk_t) in enumerate(hts):
                # Get original entity indices
                orig_h = entity_map.get(chunk_h, chunk_h)
                orig_t = entity_map.get(chunk_t, chunk_t)

                # Store prediction and score for this pair
                pair_to_preds[(orig_h, orig_t)].append({
                    'prediction': preds[local_idx],
                    'score': scores[local_idx] if scores is not None else None,
                })

        # Merge predictions for each entity pair using union strategy
        merged_hts = []
        merged_preds_list = []
        merged_scores_list = []

        for (orig_h, orig_t), pred_list in pair_to_preds.items():
            merged_hts.append([orig_h, orig_t])

            # Union strategy: take maximum across chunks for each relation
            if len(pred_list) == 1:
                # Only one chunk has this pair
                merged_pred = pred_list[0]['prediction']
                merged_score = pred_list[0]['score']
            else:
                # Multiple chunks have this pair: take max
                stacked_preds = np.stack([p['prediction'] for p in pred_list], axis=0)
                merged_pred = stacked_preds.max(axis=0)

                if scores is not None:
                    stacked_scores = np.stack([p['score'] for p in pred_list], axis=0)
                    merged_score = stacked_scores.max(axis=0)
                else:
                    merged_score = None

            merged_preds_list.append(merged_pred)
            if scores is not None:
                merged_scores_list.append(merged_score)

        # Convert to numpy arrays
        merged_preds_array = np.array(merged_preds_list)

        if chunk_scores is not None:
            merged_scores_array = np.array(merged_scores_list)
            merged_scores.append(merged_scores_array)

        # Create merged feature dict (use first chunk as template)
        merged_feature = {
            'title': doc_title,
            'hts': merged_hts,
            # Note: We don't reconstruct full entity_pos, input_ids, etc.
            # These are only needed for training, not for evaluation
        }

        merged_features.append(merged_feature)
        merged_predictions.append(merged_preds_array)

    return merged_features, merged_predictions, merged_scores
