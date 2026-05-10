#!/usr/bin/env python3
"""
Score predicted BRAT (.ann) files against gold BRAT (.ann) files.

Match entities by (start, end, type) — robust to T-id renumbering between
gold and prediction. For each file, compute relation-level TP/FP/FN where
a relation is the tuple (head_offsets, head_type, tail_offsets, tail_type, relation).

Skips ABBREVIATION/ALTERNATIVE_NAME by default (these are mention-merge relations
in BioNNE-R and not predicted by the doc-level RE model).
"""

import os
import sys
import argparse
from collections import defaultdict


def parse_ann(path, skip_relations=()):
    entities = {}  # T_id -> (type, start, end)
    relations = []  # list of (rel_type, head_T_id, tail_T_id)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("T"):
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                tid = parts[0]
                info = parts[1].split()
                if len(info) < 3:
                    continue
                etype = info[0]
                if ";" in info[1] or ";" in info[2]:
                    continue
                try:
                    start = int(info[1])
                    end = int(info[2])
                except ValueError:
                    continue
                entities[tid] = (etype, start, end)
            elif line.startswith("R"):
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                info = parts[1].split()
                if len(info) < 3:
                    continue
                rtype = info[0]
                if rtype in skip_relations:
                    continue
                arg1 = info[1].split(":")[1] if ":" in info[1] else info[1]
                arg2 = info[2].split(":")[1] if ":" in info[2] else info[2]
                relations.append((rtype, arg1, arg2))
    return entities, relations


def relations_to_tuples(entities, relations):
    """Convert (rtype, T_h, T_t) -> (rtype, h_type, h_start, h_end, t_type, t_start, t_end)."""
    out = set()
    for rtype, h, t in relations:
        if h not in entities or t not in entities:
            continue
        h_type, h_start, h_end = entities[h]
        t_type, t_start, t_end = entities[t]
        out.add((rtype, h_type, h_start, h_end, t_type, t_start, t_end))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--skip", default="ABBREVIATION,ALTERNATIVE_NAME",
                    help="Comma-separated relation types to skip in BOTH gold and pred. "
                         "Set to '' to keep all.")
    args = ap.parse_args()

    skip = tuple(s for s in args.skip.split(",") if s)
    print(f"Skipping relation types: {skip}")

    gold_files = sorted([f for f in os.listdir(args.gold_dir) if f.endswith(".ann")])
    print(f"Found {len(gold_files)} gold .ann files in {args.gold_dir}")

    overall = defaultdict(int)  # tp/fp/fn
    per_rel = defaultdict(lambda: defaultdict(int))  # rtype -> {tp, fp, fn}

    n_processed = 0
    n_missing_pred = 0

    for ann_file in gold_files:
        gold_path = os.path.join(args.gold_dir, ann_file)
        pred_path = os.path.join(args.pred_dir, ann_file)
        if not os.path.exists(pred_path):
            n_missing_pred += 1
            continue

        g_ents, g_rels = parse_ann(gold_path, skip_relations=skip)
        p_ents, p_rels = parse_ann(pred_path, skip_relations=skip)

        g_set = relations_to_tuples(g_ents, g_rels)
        p_set = relations_to_tuples(p_ents, p_rels)

        tp = g_set & p_set
        fp = p_set - g_set
        fn = g_set - p_set

        overall["tp"] += len(tp)
        overall["fp"] += len(fp)
        overall["fn"] += len(fn)

        for r in tp:
            per_rel[r[0]]["tp"] += 1
        for r in fp:
            per_rel[r[0]]["fp"] += 1
        for r in fn:
            per_rel[r[0]]["fn"] += 1

        n_processed += 1

    if n_missing_pred:
        print(f"Warning: {n_missing_pred} gold files had no matching prediction")

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    print(f"\n=== Results over {n_processed} files ===")
    p, r, f = prf(overall["tp"], overall["fp"], overall["fn"])
    print(f"\nMicro: P={p:.4f}  R={r:.4f}  F1={f:.4f}  (TP={overall['tp']}  FP={overall['fp']}  FN={overall['fn']})")

    print(f"\nPer-relation:")
    print(f"{'relation':<25} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 75)
    macro_f1 = 0.0
    n_rels = 0
    for rt in sorted(per_rel.keys()):
        s = per_rel[rt]
        p, r, f = prf(s["tp"], s["fp"], s["fn"])
        print(f"{rt:<25} {p:>8.4f} {r:>8.4f} {f:>8.4f} {s['tp']:>6} {s['fp']:>6} {s['fn']:>6}")
        macro_f1 += f
        n_rels += 1
    if n_rels:
        print("-" * 75)
        print(f"{'Macro F1':<25} {'':>8} {'':>8} {macro_f1/n_rels:>8.4f}")


if __name__ == "__main__":
    sys.exit(main())
