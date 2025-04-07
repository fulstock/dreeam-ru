import json
import os

results_json_path = "./alexey/dreeam/logs/test1/2025-03-17_14-55-32.211326/fused_results.json"
docred_test_path = "./alexey/dreeam/dataset/docred/test.json"
original_brat_path = "S:/HRCode/data/NEREL1.1/test"

save_brat_path = "S:/HRCode/data/relations/NEREL1.1/test-predicted"

if not os.path.exists(save_brat_path):
    os.makedirs(save_brat_path, exist_ok = True)

with open(results_json_path, "r", encoding = "UTF-8") as resf:
    results = json.load(resf)
with open(docred_test_path, "r", encoding = "UTF-8") as docf:
    docred = json.load(docf)


# В docred разделено всё по предложениям, а позиции в них -- по токенам, а не по символам. Чек исходный код преобразования у Алексея, чтобы восстановить символьное отношение для brat

for e_idx, (tag, first_char, last_char, entity) in enumerate(sorted(outputs, key = lambda x : x[1])):
    try:
        assert entity == text[first_char : last_char]
    except:
        print(file_entities["id"])
        print(entity)
        print(first_char, last_char)
        print(text[first_char : last_char])
        print(text[first_char - 5 : last_char + 5])
        # raise AssertionError
    annfile.write("T" + str(e_idx + 1) + "\t" + tag + " " + str(first_char) + " " + str(last_char) + "\t" + entity + "\n")  