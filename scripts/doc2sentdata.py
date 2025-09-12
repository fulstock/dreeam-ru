import json
import sys

def convert_doc_to_sent_level(input_file, output_file):
    """
    Converts a document-level RE dataset to sentence-level by removing
    cross-sentence relations.

    Args:
        input_file (str): Path to the input JSON file (document-level data).
        output_file (str): Path to the output JSON file (sentence-level data).
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{input_file}': {e}")
        return

    converted_data = []

    for doc_idx, document in enumerate(data):
        # print(f"Processing document {doc_idx}...") # Optional progress indicator
        vertex_set = document.get("vertexSet", [])
        labels = document.get("labels", [])
        sents = document.get("sents", [])
        global_sents_ids = document.get("global_sents_ids", [])
        title = document.get("title", "")
        part = document.get("part", -1)

        num_sentences = len(sents)
        sent_id_to_entities = {i: [] for i in range(num_sentences)}
        sent_id_to_global_id = {local_id: global_sents_ids[local_id] if local_id < len(global_sents_ids) else -1 for local_id in range(num_sentences)}
        global_id_to_sent_id = {gid: lid for lid, gid in sent_id_to_global_id.items() if gid != -1}

        # Map entities to their sentence based on the first mention's sent_id
        for entity_cluster_idx, entity_mentions in enumerate(vertex_set):
            # Assuming the first mention determines the entity's primary sentence for simplicity in this filter
            primary_sent_id = entity_mentions[0].get("sent_id")
            if primary_sent_id is not None and 0 <= primary_sent_id < num_sentences:
                 sent_id_to_entities[primary_sent_id].append(entity_cluster_idx)

        # Identify intra-sentence relations
        intra_sent_relations = []
        for relation in labels:
            h_idx = relation.get("h") # Head entity cluster index
            t_idx = relation.get("t") # Tail entity cluster index
            evidence_sent_ids = relation.get("evidence", [])

            if h_idx is None or t_idx is None:
                continue # Skip malformed relation

            # Find sentences for head and tail entities (based on primary mention)
            h_primary_sent_id = vertex_set[h_idx][0].get("sent_id") if h_idx < len(vertex_set) and vertex_set[h_idx] else None
            t_primary_sent_id = vertex_set[t_idx][0].get("sent_id") if t_idx < len(vertex_set) and vertex_set[t_idx] else None

            # Check if head and tail are in the same sentence AND that sentence is valid
            if (h_primary_sent_id is not None and t_primary_sent_id is not None and
                0 <= h_primary_sent_id < num_sentences and
                0 <= t_primary_sent_id < num_sentences and
                h_primary_sent_id == t_primary_sent_id and
                evidence_sent_ids and # Ensure there is evidence
                all(evid_sent_id in global_id_to_sent_id for evid_sent_id in evidence_sent_ids) # Ensure evidence sentences are within the document
               ):
                # Check if ALL evidence sentences are the same as the entity sentence
                # This is a stricter check for intra-sentential relations based on evidence
                entity_sent_global_id = sent_id_to_global_id.get(h_primary_sent_id)
                if entity_sent_global_id is not None and all(evid_id == entity_sent_global_id for evid_id in evidence_sent_ids):
                     intra_sent_relations.append(relation)

        # Create sentence-level entries
        for sent_id in range(num_sentences):
            sent_data = {
                "title": f"{title}_sent{sent_id}",
                "part": part,
                "sent_id": sent_id, # Add local sentence ID
                "global_sent_id": sent_id_to_global_id.get(sent_id, -1), # Add global sentence ID if available
                "vertexSet": [vertex_set[ent_idx] for ent_idx in sent_id_to_entities[sent_id] if ent_idx < len(vertex_set)],
                "sents": [sents[sent_id]] if sent_id < len(sents) else [],
                # Filter labels for this sentence based on evidence
                "labels": [rel for rel in intra_sent_relations if
                           vertex_set[rel['h']][0]['sent_id'] == sent_id and
                           vertex_set[rel['t']][0]['sent_id'] == sent_id and
                           all(evid_id == sent_id_to_global_id.get(sent_id) for evid_id in rel.get('evidence', []))
                          ]
            }
            # Only add sentence if it contains entities or relations (optional filter)
            # if sent_data["vertexSet"] or sent_data["labels"]:
            converted_data.append(sent_data)

    # Write the converted data
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        print(f"Conversion successful. Output written to '{output_file}'.")
    except IOError as e:
        print(f"Error writing to output file '{output_file}': {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python doc2sentdata.py <input_file.json> <output_file.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_doc_to_sent_level(input_file, output_file)
