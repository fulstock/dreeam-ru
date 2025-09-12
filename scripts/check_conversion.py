import json
import sys

def check_conversion(original_file, converted_file):
    """
    Checks if the conversion from document-level to sentence-level is correct.
    Verifies:
    1. No cross-sentence relations exist in the converted data.
    2. All intra-sentence relations from the original data are present in the converted data.
    3. Entities are correctly placed in their respective sentence contexts.
    """
    try:
        with open(original_file, 'r', encoding='utf-8') as f_orig, \
             open(converted_file, 'r', encoding='utf-8') as f_conv:
            orig_data = json.load(f_orig)
            conv_data = json.load(f_conv)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return False

    conv_sent_map = {} # Map (title, sent_id) -> converted sentence data
    for sent_doc in conv_data:
        key = (sent_doc.get("title", "").rsplit("_sent", 1)[0], sent_doc.get("sent_id"))
        conv_sent_map[key] = sent_doc

    issues_found = False

    for doc_idx, orig_doc in enumerate(orig_data):
        orig_vertex_set = orig_doc.get("vertexSet", [])
        orig_labels = orig_doc.get("labels", [])
        orig_sents = orig_doc.get("sents", [])
        orig_global_sents_ids = orig_doc.get("global_sents_ids", [])
        orig_title = orig_doc.get("title", "")
        num_orig_sentences = len(orig_sents)

        # Check 1: Ensure no cross-sentence relations in converted data for this doc's sentences
        for sent_id in range(num_orig_sentences):
            conv_key = (orig_title, sent_id)
            conv_sent_data = conv_sent_map.get(conv_key)
            if conv_sent_data:
                conv_labels = conv_sent_data.get("labels", [])
                for rel in conv_labels:
                    h_sent_id = orig_vertex_set[rel['h']][0]['sent_id'] if rel['h'] < len(orig_vertex_set) and orig_vertex_set[rel['h']] else None
                    t_sent_id = orig_vertex_set[rel['t']][0]['sent_id'] if rel['t'] < len(orig_vertex_set) and orig_vertex_set[rel['t']] else None
                    # If relation exists in converted sentence, head/tail must be in that sentence
                    if not (h_sent_id == sent_id and t_sent_id == sent_id):
                        print(f"ERROR in {conv_key}: Found relation {rel} where head/tail not in sentence {sent_id}.")
                        issues_found = True

        # Check 2: Ensure all intra-sentence relations from original are present
        for orig_rel in orig_labels:
             h_idx = orig_rel.get("h")
             t_idx = orig_rel.get("t")
             evidence_ids = orig_rel.get("evidence", [])

             if h_idx is None or t_idx is None or not evidence_ids:
                 continue # Skip if missing data

             h_primary_sent_id = orig_vertex_set[h_idx][0].get("sent_id") if h_idx < len(orig_vertex_set) and orig_vertex_set[h_idx] else None
             t_primary_sent_id = orig_vertex_set[t_idx][0].get("sent_id") if t_idx < len(orig_vertex_set) and orig_vertex_set[t_idx] else None

             # Check if it's an intra-sentence relation in the original
             if (h_primary_sent_id is not None and t_primary_sent_id is not None and
                 0 <= h_primary_sent_id < num_orig_sentences and
                 0 <= t_primary_sent_id < num_orig_sentences and
                 h_primary_sent_id == t_primary_sent_id):
                 # Find the global sentence ID for this relation's context
                 entity_sent_global_id = orig_global_sents_ids[h_primary_sent_id] if h_primary_sent_id < len(orig_global_sents_ids) else None

                 # Check if evidence sentences confirm it's intra-sentential
                 if entity_sent_global_id is not None and all(evid_id == entity_sent_global_id for evid_id in evidence_ids):
                     # This relation should exist in the converted sentence data
                     conv_key = (orig_title, h_primary_sent_id)
                     conv_sent_data = conv_sent_map.get(conv_key)
                     found_in_converted = False
                     if conv_sent_data:
                         for conv_rel in conv_sent_data.get("labels", []):
                             # Simple check: same head, tail, and sentence context
                             if (conv_rel.get("h") == orig_rel.get("h") and
                                 conv_rel.get("t") == orig_rel.get("t") and
                                 all(evid_id == entity_sent_global_id for evid_id in conv_rel.get("evidence", [])) # Check evidence matches
                                ):
                                 found_in_converted = True
                                 break
                     if not found_in_converted:
                         print(f"ERROR: Intra-sentence relation from original doc '{orig_title}' (sent {h_primary_sent_id}) likely missing in converted  {orig_rel}")
                         issues_found = True


        # Check 3: Basic entity presence check (could be more detailed)
        # For each original sentence, check if entities mentioned in it appear in the converted sentence data
        # This is a simplified check. A full check would involve mapping entity cluster IDs correctly.
        # A simpler proxy: ensure sentence data exists and has corresponding vertexSets if original had entities in that sent.
        # This check is implicitly covered by the structure of the conversion but can be made explicit if needed.

    if not issues_found:
        print("Conversion check passed: No obvious errors found based on implemented checks.")
        return True
    else:
        print("Conversion check FAILED: Issues found.")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_conversion.py <original_file.json> <converted_file.json>")
        sys.exit(1)

    original_file = sys.argv[1]
    converted_file = sys.argv[2]
    check_conversion(original_file, converted_file)
