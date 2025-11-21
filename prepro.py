from tqdm import tqdm
import ujson as json
import numpy as np
import pickle
import os

def add_entity_markers(sample, tokenizer, entity_start, entity_end):
    ''' add entity marker (*) at the end and beginning of entities. '''

    sents = []
    sent_map = []
    sent_pos = []

    sent_start = 0
    for i_s, sent in enumerate(sample['sents']):
    # add * marks to the beginning and end of entities
        new_map = {}

        # print(sent)
        
        for i_t, token in enumerate(sent):
            tokens_wordpiece = tokenizer.tokenize(token)
            if (i_s, i_t) in entity_start:
                tokens_wordpiece = ["*"] + tokens_wordpiece
            if (i_s, i_t) in entity_end:
                tokens_wordpiece = tokens_wordpiece + ["*"]
            new_map[i_t] = len(sents)
            sents.extend(tokens_wordpiece)
        
        sent_end = len(sents)
        # [sent_start, sent_end)
        sent_pos.append((sent_start, sent_end,))
        sent_start = sent_end
        
        # update the start/end position of each token.
        new_map[i_t + 1] = len(sents)
        sent_map.append(new_map)

    return sents, sent_map, sent_pos


def get_pseudo_features(raw_feature: dict, pred_rels: list, entities: list, sent_map: dict, offset: int, tokenizer = None): 

    ''' Construct pseudo documents from predictions.'''
    
    pos_samples = 0
    neg_samples = 0
    
    sent_grps = []
    pseudo_features = []

    for pred_rel in pred_rels:
        curr_sents = pred_rel["evidence"] #evidence sentence
        if len(curr_sents) == 0:
            continue

        # check if head/tail entity presents in evidence. if not, append sentence containing the first mention of head/tail into curr_sents
        head_sents = sorted([m["sent_id"] for m in entities[pred_rel["h_idx"]]])
        tail_sents = sorted([m["sent_id"] for m in entities[pred_rel["t_idx"]]])

        if len(set(head_sents) & set(curr_sents)) == 0:
            curr_sents.append(head_sents[0])
        if len(set(tail_sents) & set(curr_sents)) == 0:
            curr_sents.append(tail_sents[0])

        curr_sents = sorted(set(curr_sents))
        if curr_sents in sent_grps: # skip if such sentence group has already been created
            continue
        sent_grps.append(curr_sents)

        # new sentence masks and input ids
        old_sent_pos = [raw_feature["sent_pos"][i] for i in curr_sents]
        new_input_ids_each = [raw_feature["input_ids"][s[0] + offset:s[1] + offset] for s in old_sent_pos]
        new_input_ids = sum(new_input_ids_each, [])
        new_input_ids = tokenizer.build_inputs_with_special_tokens(new_input_ids)
 
        new_sent_pos = []

        prev_len = 0
        for sent in old_sent_pos:
            curr_sent_pos =  (prev_len, prev_len + sent[1] - sent[0])
            new_sent_pos.append(curr_sent_pos)
            prev_len += sent[1] - sent[0]

        # iterate through all entities, keep only entities with mention in curr_sents.
        
        # obtain entity positions w.r.t whole document
        curr_entities = []  
        ent_new2old = {} # head/tail of a relation should be selected
        new_entity_pos = []

        for i, entity in enumerate(entities):
            curr = []
            curr_pos = []
            for mention in entity:
                if mention["sent_id"] in curr_sents:
                    curr.append(mention)
                    prev_len = new_sent_pos[curr_sents.index(mention["sent_id"])][0]
                    pos = [sent_map[mention["sent_id"]][pos] - sent_map[mention["sent_id"]][0] + prev_len for pos in mention['pos']]
                    curr_pos.append(pos)

            if curr != []:
                curr_entities.append(curr)
                new_entity_pos.append(curr_pos)
                ent_new2old[len(ent_new2old)] = i # update dictionary
        
        # iterate through all entities to obtain all entity pairs
        new_hts = []
        new_labels = []
        for h in range(len(curr_entities)):
            for t in range(len(curr_entities)):
                if h != t:
                    new_hts.append([h, t])
                    old_h, old_t = ent_new2old[h], ent_new2old[t]
                    curr_label = raw_feature["labels"][raw_feature["hts"].index([old_h, old_t])]
                    new_labels.append(curr_label)

                    neg_samples += curr_label[0]
                    pos_samples += 1 - curr_label[0]

        pseudo_feature = {'input_ids': new_input_ids,
                    'entity_pos': new_entity_pos,
                    'labels': new_labels,
                    'hts': new_hts,
                    'sent_pos': new_sent_pos,
                    'sent_labels': None,
                    'title': raw_feature['title'],
                    'entity_map': ent_new2old,
                    }
        pseudo_features.append(pseudo_feature)

    return pseudo_features, pos_samples, neg_samples

def read_docred(dataset_dir,
                file_in, 
                tokenizer, 
                transformer_type="bert",
                max_seq_length=1024, 
                teacher_sig_path="",
                single_results=None):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []

    max_len = 0

    docred_rel2id = json.load(open(os.path.join(dataset_dir, 'rel2id.json'), 'r'))

    if file_in == "":
        return None

    with open(file_in, "r", encoding = "UTF-8") as fh:
        data = json.load(fh)

    if teacher_sig_path != "": # load logits
        basename = os.path.splitext(os.path.basename(file_in))[0]
        attns_file = os.path.join(teacher_sig_path, f"{basename}.attns")
        attns = pickle.load(open(attns_file, 'rb'))

    if single_results != None:  
        #reorder predictions as relations by title
        pred_pos_samples = 0
        pred_neg_samples = 0
        pred_rels = single_results
        title2preds = {}
        for pred_rel in pred_rels:
            if pred_rel["title"] in title2preds:
                title2preds[pred_rel["title"]].append(pred_rel)
            else:
                title2preds[pred_rel["title"]] = [pred_rel]

    for doc_id in tqdm(range(len(data)), desc="Loading examples"):

        sample = data[doc_id]
        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        # record entities
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))

        # add entity markers
        sents, sent_map, sent_pos = add_entity_markers(sample, tokenizer, entity_start, entity_end)

        # training triples with positive examples (entity pairs with labels)
        train_triple = {}

        # print(docred_rel2id)

        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])

                # update training triples
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})
                
        # entity start, end position
        entity_pos = []

        for e in entities:
            entity_pos.append([])
            assert len(e) != 0
            for m in e:
                # print("m", m)
                # print('m["pos"]', m["pos"])
                # print('m["pos"][0]', m["pos"][0])
                # print('m["sent_id"]', m["sent_id"])
                # print('sent_map', sent_map)
                # print('len(sent_map)', len(sent_map))
                # print('sent_map[m["sent_id"]]', sent_map[m["sent_id"]])
                # print('sent_map[m["sent_id"]][m["pos"][0]]', sent_map[m["sent_id"]][m["pos"][0]])
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                label = m["type"]
                entity_pos[-1].append((start, end,))

        relations, hts, sent_labels = [], [], []

        # print(docred_rel2id)

        for h, t in train_triple.keys(): # for every entity pair with gold relation
            relation = [0] * len(docred_rel2id)
            sent_evi = [0] * len(sent_pos)

            for mention in train_triple[h, t]: # for each relation mention with head h and tail t
                # print(mention)
                relation[mention["relation"]] = 1
                for i in mention["evidence"]:
                    sent_evi[i] += 1

            relations.append(relation)
            hts.append([h, t])
            sent_labels.append(sent_evi)
            pos_samples += 1

        # print(len(relations), len(entities))

        for h in range(len(entities)):
            for t in range(len(entities)):
                # all entity pairs that do not have relation are treated as negative samples
                if h != t and [h, t] not in hts: #and [t, h] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    sent_evi = [0] * len(sent_pos)
                    relations.append(relation)

                    hts.append([h, t])
                    sent_labels.append(sent_evi)
                    neg_samples += 1
        # print(docred_rel2id)
        # print(len(relations), len(entities))
        # print(data[doc_id]["title"])
        # print(entities)
        # print(relations)
        # print(entities)
        assert len(relations) <= len(entities) * (len(entities) - 1) + 1
        assert len(relations) >= len(entities) * (len(entities) - 1)
        # assert len(sents) < max_seq_length
        if max_len < len(sents):
            max_len = len(sents)
        if len(sents) > max_seq_length:
          print(f'Warning: len(sent): {len(sents)} > max_seq_length {max_seq_length}')
        sents = sents[:max_seq_length - 2] # truncate, -2 for [CLS] and [SEP]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        # if len(sent_labels) < 1:
        #     print({'input_ids': input_ids,
        #            'entity_pos': entity_pos,
        #            'labels': relations,
        #            'hts': hts,
        #            'sent_pos': sent_pos,
        #            'sent_labels': sent_labels,
        #            'title': sample['title'],
        #            })

        feature = [{'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'sent_pos': sent_pos,
                   'sent_labels': sent_labels,
                   'title': sample['title'],
                   }]

        # print(len(sent_labels))

        if teacher_sig_path != '': # add evidence distributions from the teacher model
            feature[0]['attns'] = attns[doc_id][:, :len(input_ids)]

        if single_results != None: # get pseudo documents from predictions of the single run
            offset = 1 if transformer_type in ["bert", "roberta"] else 0
            if sample["title"] in title2preds:
                feature, pos_sample, neg_sample, = get_pseudo_features(feature[0], title2preds[sample["title"]], entities, sent_map, offset, tokenizer)
                pred_pos_samples += pos_sample
                pred_neg_samples += neg_sample

        i_line += len(feature)
        features.extend(feature)

    print("# of documents {}.".format(i_line))
    if single_results != None:
        print("# of positive examples {}.".format(pred_pos_samples))
        print("# of negative examples {}.".format(pred_neg_samples))

    else:        
        print("# of positive examples {}.".format(pos_samples))
        print("# of negative examples {}.".format(neg_samples))

    print("Maximum length: ", max_len)

    return features


def negative_sampling(features, neg_pos_ratio=3.0, seed=42):
    """
    Balance training data by sampling negative examples to achieve target neg:pos ratio.

    Implements negative sampling from Ayaou (2025):
    "Tackling Class Imbalance in Relation Extraction for french text"

    Args:
        features: List of training examples (each is a dict with 'labels', etc.)
        neg_pos_ratio: Desired negative-to-positive ratio (default: 3.0 for 3:1)
        seed: Random seed for reproducibility

    Returns:
        List of features with balanced neg:pos ratio
    """
    import random
    random.seed(seed)

    positive_features = []
    negative_features = []

    print(f"\nApplying negative sampling (target ratio {neg_pos_ratio}:1)...")

    # Count positive and negative examples at the document level
    # A document is "positive" if it contains at least one positive relation
    for feature in features:
        has_positive_relation = False

        for label in feature['labels']:
            # label[0] is "no relation" (Na), label[1:] are actual relations
            if sum(label[1:]) > 0:  # Has at least one positive relation
                has_positive_relation = True
                break

        if has_positive_relation:
            positive_features.append(feature)
        else:
            negative_features.append(feature)

    print(f"  Original: {len(positive_features)} positive, {len(negative_features)} negative documents")
    print(f"  Original ratio: {len(negative_features) / (len(positive_features) + 1e-8):.2f}:1")

    # Calculate target number of negative examples
    target_negatives = int(len(positive_features) * neg_pos_ratio)

    # Sample negatives
    if len(negative_features) > target_negatives:
        sampled_negatives = random.sample(negative_features, target_negatives)
        print(f"  Sampled {target_negatives} negative documents from {len(negative_features)}")
    else:
        sampled_negatives = negative_features
        print(f"  Using all {len(negative_features)} negative documents (less than target {target_negatives})")

    # Combine positive and sampled negative examples
    balanced_features = positive_features + sampled_negatives

    # Shuffle to mix positive and negative examples
    random.shuffle(balanced_features)

    print(f"  Final: {len(positive_features)} positive, {len(sampled_negatives)} negative documents")
    print(f"  Final ratio: {len(sampled_negatives) / (len(positive_features) + 1e-8):.2f}:1")
    print(f"  Total documents: {len(balanced_features)}\n")

    return balanced_features


def compute_relation_frequencies(features, num_classes=48):
    """
    Compute the frequency of each relation class in the dataset.

    Args:
        features: List of training examples
        num_classes: Total number of relation classes

    Returns:
        List of frequencies for each class (index = class id, value = count)
    """
    from collections import Counter

    relation_counts = Counter()

    for feature in features:
        for label in feature['labels']:
            # label is a one-hot or multi-hot vector
            for rel_id, is_present in enumerate(label):
                if is_present > 0:
                    relation_counts[rel_id] += 1

    # Convert to list format
    frequencies = [relation_counts.get(i, 0) for i in range(num_classes)]

    print("\nRelation frequency statistics:")
    print(f"  Total relations: {sum(frequencies)}")
    print(f"  Unique relation types with samples: {sum(1 for f in frequencies if f > 0)}/{num_classes}")
    print(f"  Most frequent: class {np.argmax(frequencies)} with {max(frequencies)} samples")
    print(f"  Least frequent (non-zero): {min(f for f in frequencies if f > 0)} samples")

    return frequencies

