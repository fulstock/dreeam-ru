import json
import os
import argparse

from nltk.data import load
from nltk.tokenize import NLTKWordTokenizer

from tqdm.auto import tqdm

try:
    import pymorphy3 as pymorphy2
except ImportError:
    import pymorphy2

# # ## Закомментируйте после первой загрузки пакета. 
# import nltk
# nltk.download('punkt')
# ##

word_tokenizer = NLTKWordTokenizer()
# sent_tokenizer + morph initialized after argparse based on --lang

brat2mrc_parser = argparse.ArgumentParser(description = "Brat to docred formatter script.")
brat2mrc_parser.add_argument('--brat_dataset_path', type = str, required = True, help = "Path to brat dataset. Either a parent dir with train/dev/test subdirs, or a flat dir with .txt/.ann files (use --subsets none).")
brat2mrc_parser.add_argument('--subsets', type = str, default = "train,dev,test", help = "Comma-separated subset dirs to convert (default: train,dev,test). Use 'none' if brat_dataset_path is a flat directory with .txt/.ann files directly.")
brat2mrc_parser.add_argument('--output_name', type = str, default = None, help = "Output JSON filename (without .json extension). Only used with --subsets none. Default: 'train'.")
brat2mrc_parser.add_argument('--ner_path', type = str, required = True, help = 'Path to ner tags file with format ["CLASS1", "CLASS2", ...].')
brat2mrc_parser.add_argument('--rel_path', type = str, required = True, help = 'Path to relation tags file with format ["CLASS1", "CLASS2", ...].')
brat2mrc_parser.add_argument('--docred_output_path', type = str, default = None, help = "Path, where formatted dataset would be stored. By default, same path as in --brat_dataset_path would be used.")
brat2mrc_parser.add_argument('--split_length', type = int, default = 128, help = "Split length.")
brat2mrc_parser.add_argument('--split_diff', type = int, default = 4, help = "Split difference (amount in sentences).")
brat2mrc_parser.add_argument('--lang', type=str, default='russian', choices=['russian', 'english'],
                             help='Language for sentence tokenizer + lemmatizer. Default: russian.')

args = brat2mrc_parser.parse_args()

sent_tokenizer = load(f"tokenizers/punkt/{args.lang}.pickle")
if args.lang == 'russian':
    morph = pymorphy2.MorphAnalyzer()
else:
    morph = None

brat_dataset_path = args.brat_dataset_path

split_length = args.split_length
split_diff = args.split_diff

docred_output_path = args.docred_output_path
if docred_output_path is None:
    docred_output_path = brat_dataset_path

if not os.path.exists(docred_output_path):
    os.makedirs(docred_output_path)

ner_tags_path = args.ner_path

with open(ner_tags_path, "r") as ner_tags_file:
    ner_tags = json.loads(ner_tags_file.read())

ner_tags = ["BLANK"] + sorted(ner_tags)

print(ner_tags)

rel_tags_path = args.rel_path

with open(rel_tags_path, "r") as rel_tags_file:
    rel_tags = json.loads(rel_tags_file.read())

rel_tags = ["Na"] + sorted(rel_tags)

print(rel_tags)

rel_info = {}
rel_info_inv = {}
rel2id = {}

rel2id["Na"] = 0

for rel_id, rel_tag in enumerate([r for r in rel_tags[1:] if r != "ALTERNATIVE_NAME" and r != "ABBREVIATION"]):
    rel_info["P" + str(rel_id + 1)] = rel_tag
    rel2id[rel_tag] = rel_id + 1
    rel_info_inv[rel_tag] = "P" + str(rel_id + 1)

with open(os.path.join(docred_output_path, "rel_info.json"), "w") as rel_info_file:
    json.dump(rel_info, rel_info_file, ensure_ascii = False)

with open(os.path.join(docred_output_path, "rel2id.json"), "w") as rel2id_file:
    json.dump(rel2id, rel2id_file, ensure_ascii = False)

ner2id = {tag : tid for tid, tag in enumerate(ner_tags)}

with open(os.path.join(docred_output_path, "ner2id.json"), "w") as ner2id_file:
    json.dump(ner2id, ner2id_file, ensure_ascii = False)

if args.subsets.strip().lower() == "none":
    sets = [""]
else:
    sets = args.subsets.split(',')

for ds in sets:

    dataset_path = os.path.join(brat_dataset_path, ds)

    if len(ds) > 0:
        print("Converting " + ds + " set:")
    else:
        dataset_path = brat_dataset_path
        ds = args.output_name or "train"
        print("Converting flat directory -> " + ds + ".json")

    total_relations_count = 0
    total_entity_mentions_count = 0
    total_entity_vertex_count = 0
    total_tokenizer_errors_count = 0
    total_split_errors = 0

    jsonpath = os.path.join(docred_output_path, ds + ".json")
    jsondir = os.path.dirname(jsonpath)

    if not os.path.exists(jsondir):
        os.makedirs(jsondir)

    jsonfile = open(jsonpath, "w", encoding='UTF-8') 



    docs_data = []

    for ad, dirs, files in os.walk(dataset_path):
        for f in tqdm(files):



            if f[-4:] == '.ann':
                try:

                    

                    if os.stat(dataset_path + '/' + f).st_size == 0:
                        continue

                    annfile = open(dataset_path + '/' + f, "r", encoding='UTF-8', newline = "")
                    txtfile = open(dataset_path + '/' + f[:-4] + ".txt", "r", encoding='UTF-8', newline = "")

                    global_doc_data = {}
                    global_doc_data["title"] = f[:-4]
                    doc_datas = []


                    txtdata = txtfile.read()

                    # Шаг 1. Считать все именованные сущности и отношения из файла, закрыть файл.

                    entity_mentions_ids = []
                    entity_types = []
                    entity_start_chars = []
                    entity_end_chars = []

                    unique_entity_mentions = {}

                    relation_types = []
                    head_mentions_ids = []
                    tail_mentions_ids = []

                    for line in annfile:
                        line_tokens = line.split()
                        if len(line_tokens) > 3 and len(line_tokens[0]) > 1 and line_tokens[0][0] == 'T':
                            try:
                                if line_tokens[1] in ner_tags:
                                    entity_mention_id = line_tokens[0]
                                    entity_type = line_tokens[1]
                                    start_char = int(line_tokens[2])
                                    end_char = int(line_tokens[3])
                            except ValueError:
                                continue # Все неподходящие сущности

                            total_entity_mentions_count += 1

                            entity_mentions_ids.append(entity_mention_id)
                            entity_types.append(entity_type)
                            entity_start_chars.append(start_char)
                            entity_end_chars.append(end_char)
                        elif len(line_tokens) > 3 and len(line_tokens[0]) > 1 and line_tokens[0][0] == 'R':
                            try:
                                if line_tokens[1] in rel_tags:
                                    relation_type = line_tokens[1]
                                    head_args = line_tokens[2].split('Arg1:')
                                    tail_args = line_tokens[3].split('Arg2:')
                                    head_mention_id = head_args[-1]
                                    tail_mention_id = tail_args[-1]
                            except ValueError:
                                continue

                            relation_types.append(relation_type)
                            head_mentions_ids.append(head_mention_id)
                            tail_mentions_ids.append(tail_mention_id)

                    annfile.close()
                    txtfile.close()

                    # Шаг 2. В каждом файле выделить контексты отдельно друг от друга.

                    offset_mapping = []

                    sentence_spans = list(sent_tokenizer.span_tokenize(txtdata))
                    tokenized_sentences = []
                    
                    for span in sentence_spans:

                        start, end = span
                        context = txtdata[start : end]

                        word_spans = list(word_tokenizer.span_tokenize(context))
                        tokenized_sentences.append([txtdata[s + start : e + start] for s, e in word_spans])
                        offset_mapping.extend([(s + start, e + start) for s, e in word_spans])

                    start_words, end_words = zip(*offset_mapping)

                    global_doc_data["sents"] = tokenized_sentences
 
                    total_length = sum([len(s) for s in tokenized_sentences])
                    if total_length > 512:
                        split = True
                        first_idx = 0
                        curr_len = 0
                        spk = 0
                        for s_idx, sent in enumerate(tokenized_sentences):
                            curr_len += len(sent)
                            if curr_len > split_length:
                                last_idx = s_idx
                                
                                doc_datas.append({
                                    "title" : global_doc_data["title"] + "_part" + str(spk),
                                    "part" : spk,
                                    "sents" : tokenized_sentences[first_idx : last_idx],
                                    "global_sents_ids" : list(range(first_idx, last_idx))
                                })
                                spk += 1

                                first_idx = s_idx - split_diff
                                prev_len = sum([len(prev_sent) for prev_sent in tokenized_sentences[first_idx:s_idx + 1]])
                                while prev_len > split_length:
                                    first_idx += 1
                                    prev_len = sum([len(prev_sent) for prev_sent in tokenized_sentences[first_idx:s_idx + 1]])
                                curr_len = prev_len

                        if curr_len > 0:
                            last_idx = len(tokenized_sentences)
                                    
                            doc_datas.append({
                                "title" : global_doc_data["title"] + "_part" + str(spk),
                                "part" : spk,
                                "sents" : tokenized_sentences[first_idx : last_idx],
                                "global_sents_ids" : list(range(first_idx, last_idx))
                            })
                            spk += 1

                    else:
                        split = False
                        doc_datas = [{
                            "title" : global_doc_data["title"] + "_part0",
                            "part" : 0,
                            "sents" : tokenized_sentences,
                            "global_sents_ids" : list(range(0, len(tokenized_sentences)))
                        }]

                    for entity_mentions_id, entity_type, start_char, end_char in zip(entity_mentions_ids, entity_types, entity_start_chars, entity_end_chars):

                        entity_mention = txtdata[start_char : end_char]
                        entity_lemm_mention = morph.parse(entity_mention)[0].normal_form if morph else entity_mention.lower()

                        # print(entity_mention, entity_type, start_char, end_char)

                        try:
                            sent_id = [sent_id for sent_id, (sent_start, sent_end) in enumerate(sentence_spans) if sent_start <= start_char and end_char <= sent_end][0]
                        except IndexError:
                            total_tokenizer_errors_count += 1
                            # print(txtdata)
                            # print(txtdata[start_char : end_char])
                            # print(start_char, end_char)
                            # print(sentence_spans)
                            # print([txtdata[sent_start : sent_end] for sent_start, sent_end in sentence_spans])
                            continue

                        tokenized_sentence = tokenized_sentences[sent_id]
                        sentence_start, sentence_end = sentence_spans[sent_id]
                        sentence_word_spans = [(s, e) for s, e in offset_mapping if s >= sentence_start and e <= sentence_end]
                        sentence_start_chars = [s[0] for s in sentence_word_spans]
                        sentence_end_chars = [s[1] for s in sentence_word_spans]

                        try:
                            start_word_idx = [start_idx for start_idx, s in enumerate(sentence_start_chars) if s == start_char][0]
                            end_word_idx = [end_idx for end_idx, e in enumerate(sentence_end_chars) if e == end_char][0] + 1
                        except IndexError:
                            total_tokenizer_errors_count += 1
                            continue

                        local_sent_id = sent_id
                        part = 0
                        if split:
                            for d_idx, d in enumerate(doc_datas):
                                if sent_id in d["global_sents_ids"]:
                                    sent_offset = d["global_sents_ids"][0]
                                    local_sent_id = sent_id - sent_offset
                                    part = d_idx
                                    break

                        if entity_lemm_mention not in unique_entity_mentions:
                            unique_entity_mentions[entity_lemm_mention] = [{
                                "name" : entity_mention,
                                "pos" : [start_word_idx, end_word_idx],
                                "sent_id" : local_sent_id,
                                "global_sent_id" : sent_id,
                                "part" : part,
                                "type" : entity_type,
                                "mention_id" : entity_mentions_id
                            }]
                        else:
                            unique_entity_mentions[entity_lemm_mention].append({
                                "name" : entity_mention,
                                "pos" : [start_word_idx, end_word_idx],
                                "sent_id" : local_sent_id,
                                "global_sent_id" : sent_id,
                                "part" : part,
                                "type" : entity_type,
                                "mention_id" : entity_mentions_id
                            })

                        # print({
                        #         "name" : entity_mention,
                        #         "pos" : [start_word_idx, end_word_idx],
                        #         "sent_id" : local_sent_id,
                        #         "global_sent_id" : sent_id,
                        #         "part" : part,
                        #         "type" : entity_type,
                        #         "mention_id" : entity_mentions_id
                        #     })

                    for rel_type, head_mention_id, tail_mention_id in zip(relation_types, head_mentions_ids, tail_mentions_ids):

                        if rel_type != "ALTERNATIVE_NAME" and rel_type != "ABBREVIATION":
                            continue

                        head_id = -1
                        tail_id = -1
                        head_vertex = []
                        head_lemm = ""
                        tail_vertex = []
                        tail_lemm = ""
                        for entity_id, (lemm, vertex) in enumerate(unique_entity_mentions.items()):
                            for mention in vertex:
                                if mention["mention_id"] == head_mention_id:
                                    head_id = entity_id
                                    head_lemm = lemm
                                    head_vertex = vertex
                                    
                                if mention["mention_id"] == tail_mention_id:
                                    tail_id = entity_id
                                    tail_lemm = lemm
                                    tail_vertex = vertex
                                    
                        if head_id < 0 or tail_id < 0:
                            continue
                        
                        if head_id != tail_id:
                            new_vertex = head_vertex + tail_vertex
                            unique_entity_mentions[head_lemm] = new_vertex
                            # print(tail_lemm in unique_entity_mentions)
                            # print(new_vertex)
                            del unique_entity_mentions[tail_lemm]
                            # print(tail_lemm in unique_entity_mentions)
                            # print(new_vertex)

                        # How to count evidences?

                    # print([d for lm, d in unique_entity_mentions.items()])
                    # print(doc_datas[5])

                    total_entity_vertex_count += len(unique_entity_mentions)
                    for part, doc_data in enumerate(doc_datas):
                        if "vertexSet" not in doc_datas[part]:
                            doc_datas[part]["vertexSet"] = []
                        for vertex in list(unique_entity_mentions.values()):
                            parted_vertex = [m for m in vertex if m["part"] == part and m["global_sent_id"] in doc_data["global_sents_ids"]]
                            # print(parted_vertex, len(parted_vertex))
                            # print()
                            if "vertexSet" not in doc_datas[part]:
                                if len(parted_vertex) > 0:
                                    doc_datas[part]["vertexSet"] = [parted_vertex]
                                else:
                                    doc_datas[part]["vertexSet"] = []
                            else:
                                if len(parted_vertex) > 0:
                                    doc_datas[part]["vertexSet"].append(parted_vertex)

                    # print(doc_datas[5])

                    for part, doc_data in enumerate(doc_datas):

                        curr_mention_ids = [m["mention_id"] for v in doc_datas[part]["vertexSet"] for m in v]
                        doc_datas[part]["labels"] = []

                        for rel_type, head_mention_id, tail_mention_id in zip(relation_types, head_mentions_ids, tail_mentions_ids):

                            if rel_type == "ALTERNATIVE_NAME" or rel_type == "ABBREVIATION":
                                continue

                            if head_mention_id not in curr_mention_ids or tail_mention_id not in curr_mention_ids:
                                continue

                            evidences = set()
                            head_id = -1
                            tail_id = -1
                            for entity_id, vertex in enumerate(doc_datas[part]["vertexSet"]):
                                for mention in vertex:
                                    if mention["mention_id"] == head_mention_id:
                                        head_id = entity_id
                                        evidences.add(mention["sent_id"])
                                    if mention["mention_id"] == tail_mention_id:
                                        tail_id = entity_id
                                        evidences.add(mention["sent_id"])
                            if head_id == tail_id:
                                continue

                            total_relations_count += 1
                            same_relations = [(l_idx, l) for l_idx, l in enumerate(doc_datas[part]["labels"]) if l["r"] == rel_type and l["h"] == head_id and l["t"] == tail_id]
                            assert len(same_relations) <= 1
                            if len(same_relations) > 0:
                                l_idx = same_relations[0][0]
                                ev = set(doc_datas[part]["labels"][l_idx]["evidence"])
                                ev.update(evidences)
                                doc_datas[part]["labels"][l_idx]["evidence"] = sorted(list(ev))
                            else:
                                doc_datas[part]["labels"].append({
                                    "r" : rel_type,
                                    "h" : head_id,
                                    "t" : tail_id,
                                    "evidence" : sorted(list(evidences))
                                })

                    total_split_errors += len(tokenized_sentences) - len(set([d for doc_data in doc_datas for d in doc_data["global_sents_ids"]])) 

                    docs_data.extend(doc_datas)


                except FileNotFoundError:
                    pass
            
    # Вывод результатов
    print("Total entity mentions:", total_entity_mentions_count)
    print("Total entities:", total_entity_vertex_count)
    print("Total relations:", total_relations_count)
    print("Total tokenizing errors:", total_tokenizer_errors_count)
    print("Total split errors:", total_split_errors)

    json.dump(docs_data, jsonfile, ensure_ascii = False, indent = 2)
    jsonfile.close()