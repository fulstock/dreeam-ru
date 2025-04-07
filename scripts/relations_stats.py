import json
from tqdm.auto import tqdm

dataset_path = "S:/HRCode/data/relations/seccol-full-docred/full.json"
relations_path = "S:/HRCode/data/relations/seccol.rels"

with open(dataset_path, "r", encoding = "UTF-8") as df:
    train_data = json.load(df)

with open(relations_path, "r", encoding = "UTF-8") as rf:
    rel_tags = json.load(rf)
rel_tags = sorted(rel_tags)

insentence_relations = {k : [] for k in rel_tags}
crosssentence_relations = {k : [] for k in rel_tags}

print(insentence_relations)

for doc_info in train_data:
    sents = doc_info["sents"]
    sent_lens = [len(s) for s in sents]
    for relation_info in doc_info["labels"]:
        relation = relation_info["r"]
        evidence = relation_info["evidence"]
        if len(evidence) == 1:
            insentence_relations[relation].append(sent_lens[evidence[0]])
        else:
            total_length = sum([sent_lens[i] for i in range(min(evidence), max(evidence) + 1)])
            crosssentence_relations[relation].append(total_length)

print("In-sentence relations; Cross-sentence relations")
print("Relation Count Max-length Count Max-length")
for k in rel_tags:
    print(k, len(insentence_relations[k]), max(insentence_relations[k], default = 0), len(crosssentence_relations[k]), max(crosssentence_relations[k], default = 0))