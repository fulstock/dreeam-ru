import json
import os

from tqdm.auto import tqdm

brat_dataset_path = "S:/HRCode/data/seccol/seccol_events_texts_1500_new2-div"

sets = ["train", "dev", "test"]

rel_tags_set = set()

for ds in sets:

    print(ds + " set:")

    # jsonpath = os.path.join(hfds_output_path, ds + ".json")
    dataset_path = os.path.join(brat_dataset_path, ds)

    for ad, dirs, files in os.walk(dataset_path):
        for f in tqdm(files):

            if f[-4:] == '.ann':
                try:

                    if os.stat(dataset_path + '/' + f).st_size == 0:
                        continue

                    annfile = open(dataset_path + '/' + f, "r", encoding='UTF-8')

                    entity_types = []
                    entity_start_chars = []
                    entity_end_chars = []

                    for line in annfile:
                        line_tokens = line.split()
                        if len(line_tokens) > 3 and len(line_tokens[0]) > 1 and line_tokens[0][0] == 'R':
                            try:
                                rel_tags_set.add(line_tokens[1])
                            except ValueError:
                                continue # Все неподходящие отношения

                    annfile.close()

                except FileNotFoundError:
                    pass

rel_tags = sorted(list(rel_tags_set))
print(rel_tags)
with open("S:/HRCode/data/relations/seccol.rels", "w", encoding = "UTF-8") as rf:
    json.dump(rel_tags, rf)