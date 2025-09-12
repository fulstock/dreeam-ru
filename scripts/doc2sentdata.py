import json
import sys
from collections import defaultdict

def convert_doc_to_sent_level(input_file, output_file):
    """
    Converts a document-level RE dataset to sentence-level by removing
    cross-sentence relations and correctly splitting vertexSets.

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
        orig_vertex_set = document.get("vertexSet", [])
        labels = document.get("labels", [])
        sents = document.get("sents", [])
        global_sents_ids = document.get("global_sents_ids", [])
        title = document.get("title", "")
        part = document.get("part", -1)

        num_sentences = len(sents)
        # Map original entity cluster index to {sentence_id: [mention_indices]}
        entity_sent_mapping = defaultdict(lambda: defaultdict(list))
        # Map sentence_id to set of original entity cluster indices present in that sentence
        sent_to_entity_indices = defaultdict(set)

        # 1. Analyze mentions to build mappings
        for ent_cluster_idx, entity_mentions in enumerate(orig_vertex_set):
            for mention_idx, mention in enumerate(entity_mentions):
                sent_id = mention.get("sent_id")
                if sent_id is not None and 0 <= sent_id < num_sentences:
                    entity_sent_mapping[ent_cluster_idx][sent_id].append(mention_idx)
                    sent_to_entity_indices[sent_id].add(ent_cluster_idx)

        # 2. Identify intra-sentence relations based on evidence and entity sentence
        intra_sent_relations_per_sentence = defaultdict(list)
        # Map (h_orig_idx, t_orig_idx) to set of sentences where this specific intra-sent relation occurs
        relation_to_sentences = defaultdict(set)

        for relation in labels:
            h_orig_idx = relation.get("h") # Head entity cluster index in original doc
            t_orig_idx = relation.get("t") # Tail entity cluster index in original doc
            evidence_global_ids = set(relation.get("evidence", []))

            if h_orig_idx is None or t_orig_idx is None or not evidence_global_ids:
                continue # Skip malformed or relations without evidence

            # Check if head and tail entities have mentions and get their primary sentence
            # (Using the first mention's sent_id as the primary sentence for the entity in this context)
            h_mentions = orig_vertex_set[h_orig_idx] if h_orig_idx < len(orig_vertex_set) else []
            t_mentions = orig_vertex_set[t_orig_idx] if t_orig_idx < len(orig_vertex_set) else []
            if not h_mentions or not t_mentions:
                continue

            h_primary_sent_id = h_mentions[0].get("sent_id")
            t_primary_sent_id = t_mentions[0].get("sent_id")

            # Check if head and tail are in the same sentence AND that sentence is valid
            if (h_primary_sent_id is not None and t_primary_sent_id is not None and
                0 <= h_primary_sent_id < num_sentences and
                0 <= t_primary_sent_id < num_sentences and
                h_primary_sent_id == t_primary_sent_id):
                # Get the global ID of the sentence where entities are located
                entity_sent_global_id = global_sents_ids[h_primary_sent_id] if h_primary_sent_id < len(global_sents_ids) else None

                # Check if ALL evidence sentences match the entity's sentence global ID
                if entity_sent_global_id is not None and evidence_global_ids == {entity_sent_global_id}:
                    # This is a valid intra-sentence relation for sentence h_primary_sent_id
                    intra_sent_relations_per_sentence[h_primary_sent_id].append(relation)
                    relation_to_sentences[(h_orig_idx, t_orig_idx)].add(h_primary_sent_id)


        # 3. Create sentence-level entries
        for sent_id in range(num_sentences):
            # Get global sentence ID if available
            global_sent_id = global_sents_ids[sent_id] if sent_id < len(global_sents_ids) else -1

            # Build the new vertexSet for this sentence
            # Only include entities that have mentions in this sentence
            new_vertex_set = []
            # Map original entity index to new entity index within this sentence's vertexSet
            orig_to_new_entity_index = {}

            for orig_ent_idx in sent_to_entity_indices.get(sent_id, []):
                mention_indices_in_this_sent = entity_sent_mapping[orig_ent_idx].get(sent_id, [])
                if mention_indices_in_this_sent:
                    # Create a new entity cluster containing only mentions from this sentence
                    new_entity_cluster = [orig_vertex_set[orig_ent_idx][i] for i in mention_indices_in_this_sent]
                    new_vertex_set.append(new_entity_cluster)
                    orig_to_new_entity_index[orig_ent_idx] = len(new_vertex_set) - 1

            # Build the new labels for this sentence
            # Only include relations that are intra-sent and belong to this sentence
            # AND remap 'h' and 't' indices to the new local indices
            new_labels = []
            for orig_relation in intra_sent_relations_per_sentence.get(sent_id, []):
                orig_h = orig_relation['h']
                orig_t = orig_relation['t']

                # Check if both head and tail entities are present in this sentence's new vertexSet
                if orig_h in orig_to_new_entity_index and orig_t in orig_to_new_entity_index:
                    # Create a copy of the relation and update 'h' and 't' indices
                    new_relation = orig_relation.copy()
                    new_relation['h'] = orig_to_new_entity_index[orig_h]
                    new_relation['t'] = orig_to_new_entity_index[orig_t]
                    # The 'evidence' field should still be valid (pointing to global sent ID of this sentence)
                    new_labels.append(new_relation)

            # Create the sentence-level document entry
            sent_data = {
                "title": f"{title}_sent{sent_id}",
                "part": part,
                "sent_id": sent_id,
                "global_sent_id": global_sent_id,
                # Include the sentence itself
                "sents": [sents[sent_id]] if sent_id < len(sents) else [],
                # Include the correctly split vertexSet
                "vertexSet": new_vertex_set,
                # Include the filtered and remapped labels
                "labels": new_labels
            }
            # Add the entry to the converted data list
            converted_data.append(sent_data)

    # 4. Write the converted data
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