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
    select threshold for relation predictions.
    Input:
        :cand: list of relation candidates
        :num_gt: number of ground-truth relations.
        :correct: number of correct relation predictions selected.
        :num_pred: number of relation predictions selected.
    Output:
        :thresh: threshold for selecting relations.
        :sorted_pred: predictions selected from cand. 
    '''
    
    sorted_pred = sorted(cand, key=lambda x:x[1], reverse=True)
    precs, recalls = [], []
    
    for pred in sorted_pred:     
        correct += pred[0]
        num_pred += 1
        precs.append(correct / num_pred if num_pred > 0 else 0.)  # Precision
        recalls.append(correct / num_gt if num_gt > 0 else 0.)  # Recall                             

    recalls = np.asarray(recalls, dtype='float32')
    precs = np.asarray(precs, dtype='float32')
    f1_arr = (2 * recalls * precs / (recalls + precs + 1e-20))
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
    Convert the predictions to official format for evaluating.
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
    
    
    h_idx, t_idx, title, sents = [], [], [], []

    for f in features:
        if "entity_map" in f:
            hts = [[f["entity_map"][ht[0]], f["entity_map"][ht[1]]] for ht in f["hts"]]
        else:
            hts = f["hts"]

        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]
        sents += [len(f["sent_pos"])] * len(hts)

    official_res = []
    res = []

    for i in tqdm(range(preds.shape[0]), desc="preds"): # for each entity pair
        if scores != []:
            score = extract_relative_score(scores[i], topks[i]) 
            pred = topks[i]
        else:
            pred = preds[i]
            pred = np.nonzero(pred)[0].tolist()
        
        for p in pred: # for each predicted relation label (topk)
            curr_result = {
                    'title': title[i],
                    'h_idx': h_idx[i],
                    't_idx': t_idx[i],
                    'r': id2rel[p],
                }
            if evi_preds != []:
                curr_evi = evi_preds[i]
                evis = np.nonzero(curr_evi)[0].tolist() 
                curr_result["evidence"] = [evi for evi in evis if evi < sents[i]]
            if scores != []:
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

def official_evaluate(tmp, path, train_file="train_annotated.json", dev_file="dev.json"):
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
