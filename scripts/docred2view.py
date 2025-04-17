import json
import os
from tqdm.auto import tqdm

results_json_path = "S:/HRCode/relations/alexey/results.json"
docred_test_path = "S:/HRCode/data/relations/seccol-div-docred/test.json"

save_view_path = "S:/HRCode/relations/alexey/test/seccol-div-docred/results_view.txt"
save_view_dir = '/'.join(save_view_path.split('/')[:-1])

if not os.path.exists(save_view_dir):
    os.makedirs(save_view_dir, exist_ok = True)

with open(results_json_path, "r", encoding = "UTF-8") as resf:
    results = json.load(resf)
with open(docred_test_path, "r", encoding = "UTF-8") as docf:
    docred = json.load(docf)

# В docred разделено всё по предложениям, а позиции в них -- по токенам, а не по символам. Чек исходный код преобразования у Алексея, чтобы восстановить символьное отношение для brat

# docred format:
# list of document count length with this dict for one doc:
# {
#     "title" : "part1", # doc name
#     "sents" : [...], # all sentences of the doc, splitted by words. Their ids are evidence numbers in results
#     "vertexSet" : [ # all entities 
#       [
#         {
#           "name": "Пи́тер (Пит) Бра́йан Хе́гсет",
#           "pos": [
#             0,
#             6
#           ],
#           "sent_id": 0, 
#           "type": "PERSON",
#           "mention_id": "T1"
#         }
#       ], # one vertex, several mentions
#       ...
#       ],
#     "labels" : [],
# }

# docred results format:
# list of these dicts:
# {
#     "title": "part1", # file name
#     "h_idx": 0, # vertex idx, head (start of relation)
#     "t_idx": 1, # vertex idx, tail (end of relation)
#     "r": "WORKS_AS", # relation type
#     "evidence": [ # sentence evidences that support this relation
#       0
#     ],
#     "score": -8.895109176635742 # score of relation. Score MUST be > 0 (if <0 that there is no relation by definition)
# }
# 

docced_results = {}
for r in results:
    # print(r)
    # if r['score'] <= 0:
    #     continue
    if r['title'] not in docced_results.keys():
        docced_results[r['title']] = [r]
    else:
        docced_results[r['title']].append(r)

svf = open(save_view_path, 'w', encoding = "UTF-8")

processed_results = [] 
all_tags = set()
all_relations = set()

for doc in tqdm(docred):

    vertex2info = [{'name' : v[0]['name'], 'tag' : v[0]['type']} for v in doc['vertexSet']]
    sents = [' '.join(s) for s in doc['sents']]

    if doc['title'] in docced_results:
        for doc_res in tqdm(docced_results[doc['title']]):

            head_idx = doc_res['h_idx']
            tail_idx = doc_res['t_idx']
            relation = doc_res['r']
            # score = doc_res['score']
            # evidences = doc_res['evidence']

            head = vertex2info[head_idx]
            tail = vertex2info[tail_idx]
            # evidences = " AND ".join([sents[k] for k in evidences])

            all_tags.update([head['tag'], tail['tag']])
            all_relations.add(relation)
            processed_results.append((head, tail, relation)) # , evidences))


max_relation_len = max([len(tag) for tag in all_relations])

processed_results = sorted(processed_results, key = lambda x : (x[0]['tag'], x[1]['tag'], -len(x[0]['name']), x[2], x[0]['name'], x[1]['name']))

curr_head_tag = ""
curr_tail_tag = ""
curr_max_head = 0

def pad(s, l):
    if len(s) < l:
        p = (l - len(s) + 1) // 2
        s = " " * ((l - len(s) + 1) // 2) + s + " " * ((l - len(s)) // 2)
    return s

svf.write("HEAD -> RELATION -> TAIL\n")
#for head, tail, relation, evidences in processed_results:
for head, tail, relation in tqdm(processed_results):
    head_tag = head['tag']
    tail_tag = tail['tag']

    if curr_head_tag != head_tag:
        curr_head_tag = head_tag
        curr_tail_tag = tail_tag
        curr_max_head = max(len(head['name']), len(head_tag))

        svf.write("===========================================\n")
#        svf.write(curr_head_tag + "\t" + "RELATION" + "\t" + curr_tail_tag + "\t" + "EVIDENCES\n")
        svf.write(pad(curr_head_tag, curr_max_head) + " " + pad("RELATION", max_relation_len)  + " " + curr_tail_tag + "\n")
    elif curr_tail_tag != tail_tag:
        curr_tail_tag = tail_tag
        curr_max_head = max(len(head['name']), len(head_tag))

        svf.write("-------------------------------------------\n")
        svf.write(pad(curr_head_tag, curr_max_head) + " " + pad("RELATION", max_relation_len) + " " + curr_tail_tag + "\n")
#    svf.write(head['name'] + "\t" + relation + "\t" + tail['name'] + "\t" + evidences + "\n")
    svf.write(pad(head['name'], curr_max_head) + " " + pad(relation, max_relation_len) + " " + tail['name'] + "\n")

svf.close()