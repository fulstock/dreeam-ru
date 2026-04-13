import argparse
import os
import datetime

import numpy as np
import torch
import ujson as json
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from args import add_args
from model import DocREModel
from utils import set_seed, collate_fn, create_directory
from prepro import read_docred, negative_sampling, compute_relation_frequencies
from evaluation import to_official, official_evaluate, merge_results, optimize_per_class_thresholds, save_optimized_thresholds, load_optimized_thresholds, apply_per_class_thresholds
from losses import EnhancedAMTLoss, ATLoss, FocalLoss
import wandb
from tqdm import tqdm

import pandas as pd
import pickle

def load_input(batch, device, tag="dev"):

    input = {'input_ids': batch[0].to(device),
            'attention_mask': batch[1].to(device),
            'labels': batch[2].to(device),
            'entity_pos': batch[3],
            'hts': batch[4],
            'sent_pos': batch[5],
            'sent_labels': batch[6].to(device) if (not batch[6] is None) and (batch[7] is None) else None,
            'teacher_attns': batch[7].to(device) if not batch[7] is None else None,
            'tag': tag
            } 

    # print(input['sent_labels'])

    return input

def train(args, model, train_features, dev_features, id2rel):

    def finetune(features, optimizer, num_epoch, num_steps, id2rel):
        best_score = -1

        # Create training log file
        log_file = os.path.join(args.save_path, "training_log.txt")
        log_f = open(log_file, "w", buffering=1)  # Line buffering
        print(f"✓ Logging training to {log_file}")

        def log_print(msg):
            """Print to both console and log file"""
            print(msg)
            # Convert to string if dict or other non-string type
            if isinstance(msg, dict):
                msg = str(msg)
            log_f.write(str(msg) + "\n")
            log_f.flush()

        # Use smart batching if enabled (20-30% less padding)
        if args.use_smart_batching:
            from smart_batching import create_smart_dataloader
            train_dataloader = create_smart_dataloader(
                features, batch_size=args.train_batch_size,
                shuffle=True, drop_last=True,
                num_workers=4, pin_memory=True, prefetch_factor=2,
                num_buckets=args.num_buckets, collate_fn=collate_fn
            )
            print(f"✓ Smart batching enabled ({args.num_buckets} buckets, 20-30% less padding)")
        else:
            train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=4, pin_memory=True, prefetch_factor=2)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        scaler = GradScaler()
        log_print("Total steps: {}".format(total_steps))
        log_print("Warmup steps: {}".format(warmup_steps))

        # Create progress bar for steps (not epochs)
        pbar = tqdm(total=total_steps, desc='Training')

        for epoch in train_iterator:
            for step, batch in enumerate(train_dataloader):
                model.zero_grad()
                optimizer.zero_grad()
                model.train()

                inputs = load_input(batch, args.device, tag="train")
                outputs = model(**inputs)
                loss = [outputs["loss"]["rel_loss"]]

                if inputs["sent_labels"] != None:
                    loss.append(outputs["loss"]["evi_loss"] * args.evi_lambda)
                                
                if inputs["teacher_attns"] != None:
                    loss.append(outputs["loss"]["attn_loss"] * args.attn_lambda)
                
                loss = sum(loss) / args.gradient_accumulation_steps
                scaler.scale(loss).backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1

                    # Update progress bar with detailed info
                    loss_val = loss.item() * args.gradient_accumulation_steps
                    pbar.update(1)
                    pbar.set_postfix({
                        'epoch': f'{epoch+1}/{num_epoch}',
                        'loss': f'{loss_val:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })

                # wandb.log(outputs["loss"], step=num_steps)

                best_offi_results = None
                
                if (step + 1) == len(train_dataloader) or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and (step + 1) % args.gradient_accumulation_steps == 0):

                    pbar.set_description(f'Evaluating epoch {epoch+1}')
                    dev_scores, dev_output, official_results, results, _ = evaluate(args, model, dev_features, id2rel, tag="dev")
                    # wandb.log(dev_scores, step=num_steps)
                    pbar.set_description('Training')

                    log_print(f"\n=== Epoch {epoch+1}/{num_epoch} Evaluation ===")
                    log_print(dev_output)
                    if dev_scores["dev_F1_ign"] > best_score:
                        best_score = dev_scores["dev_F1_ign"]
                        best_offi_results = official_results
                        best_results = results
                        best_output = dev_output

                        ckpt_file = os.path.join(args.save_path, "best.ckpt")
                        log_print(f"saving model checkpoint into {ckpt_file} ...")
                        torch.save(model.state_dict(), ckpt_file)

                        # Save intermediate logs when best model is updated
                        pred_file = os.path.join(args.save_path, args.pred_file)
                        score_file = os.path.join(args.save_path, "scores.csv")
                        results_file = os.path.join(args.save_path, f"topk_{args.pred_file}")
                        dump_to_file(best_offi_results, pred_file, best_output, score_file, best_results, results_file)
                        log_print(f"✓ Saved intermediate results at epoch {epoch+1}, F1_ign: {best_score:.4f}")

                    # Save epoch checkpoint every 10 epochs
                    if (epoch + 1) % 10 == 0:
                        epoch_ckpt = os.path.join(args.save_path, f"epoch_{epoch+1}.ckpt")
                        log_print(f"saving epoch checkpoint into {epoch_ckpt} ...")
                        torch.save(model.state_dict(), epoch_ckpt)

                        # Save current epoch results
                        epoch_pred_file = os.path.join(args.save_path, f"epoch_{epoch+1}_{args.pred_file}")
                        epoch_score_file = os.path.join(args.save_path, f"epoch_{epoch+1}_scores.csv")
                        epoch_results_file = os.path.join(args.save_path, f"epoch_{epoch+1}_topk_{args.pred_file}")
                        dump_to_file(official_results, epoch_pred_file, dev_output, epoch_score_file, results, epoch_results_file)
                        log_print(f"✓ Saved epoch {epoch+1} results, current F1_ign: {dev_scores['dev_F1_ign']:.4f}")

                    if epoch == train_iterator[-1]: # last epoch

                        ckpt_file = os.path.join(args.save_path, "last.ckpt")
                        print(f"saving model checkpoint into {ckpt_file} ...")
                        torch.save(model.state_dict(), ckpt_file)

                        pred_file = os.path.join(args.save_path, args.pred_file)
                        score_file = os.path.join(args.save_path, "scores.csv")
                        results_file = os.path.join(args.save_path, f"topk_{args.pred_file}")

                        if not best_offi_results:
                            best_offi_results = official_results
                            best_output = dev_output
                            best_results = results
                        dump_to_file(best_offi_results, pred_file, best_output, score_file, best_results, results_file)
                        log_print(f"✓ Saved final results")

        pbar.close()
        log_print("\n=== Training Complete ===")
        log_f.close()

        return num_steps

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.lr_added},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr_transformer, eps=args.adam_epsilon)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps, id2rel)
    # Optimize per-class thresholds after training if enabled
    if args.use_per_class_thresholds:
        print("\n=== Optimizing Per-Class Thresholds ===")
        print("Loading best model checkpoint...")
        best_model_path = os.path.join(args.save_path, "best.ckpt")
        model.load_state_dict(torch.load(best_model_path))

        dev_dataloader = DataLoader(dev_features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False, num_workers=0, pin_memory=False)

        optimized_thresholds = optimize_per_class_thresholds(
            model=model,
            dev_dataloader=dev_dataloader,
            id2rel=id2rel,
            device=args.device,
            threshold_range=(args.threshold_range_min, args.threshold_range_max),
            threshold_step=args.threshold_step
        )

        # Save optimized thresholds
        threshold_path = os.path.join(args.save_path, "optimized_thresholds.json")
        save_optimized_thresholds(optimized_thresholds, threshold_path)
        print(f"✓ Optimized thresholds saved to {threshold_path}")

        # Calculate and print average F1
        if optimized_thresholds:
            avg_f1 = sum(info['f1'] for info in optimized_thresholds.values()) / len(optimized_thresholds)
            print(f"  Average F1 across all classes: {avg_f1*100:.2f}%")

def evaluate(args, model, features, id2rel, tag="dev", optimized_thresholds=None):
    # Use num_workers=0 for inference (faster for single-pass evaluation)
    # Use num_workers=4 during training (amortized over many epochs)
    num_workers = 0 if not args.do_train else 4
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False, num_workers=num_workers, pin_memory=True if num_workers > 0 else False)
    preds, evi_preds = [], []
    scores, topks = [], []
    attns = []
    logits_all = []  # Collect logits for per-class threshold application

    for batch in tqdm(dataloader, desc=f"Evaluating batches"):
        model.eval()
        if args.save_attn:
            tag = "infer"
        inputs = load_input(batch, args.device, tag)
        with torch.no_grad():
            outputs = model(**inputs)

            # If using optimized thresholds, collect logits instead of predictions
            if optimized_thresholds is not None:
                logits_all.append(outputs["logits"].cpu())  # Assuming model outputs logits

            pred = outputs["rel_pred"]
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            if "scores" in outputs:
                scores.append(outputs["scores"].cpu().numpy())  
                topks.append(outputs["topks"].cpu().numpy())   
            if "evi_pred" in outputs: # relation extraction and evidence extraction
                evi_pred = outputs["evi_pred"]
                evi_pred = evi_pred.cpu().numpy()
                evi_preds.append(evi_pred)   
            if "attns" in outputs: # attention recorded
                attn = outputs["attns"]
                attns.extend([a.cpu().numpy() for a in attn])

    # Apply per-class thresholds if provided
    if optimized_thresholds is not None and len(logits_all) > 0:
        print("\n=== Applying Per-Class Thresholds ===")
        logits_concat = torch.cat(logits_all, dim=0)
        preds_optimized = apply_per_class_thresholds(
            logits_concat,
            optimized_thresholds,
            num_labels=args.num_labels
        )
        preds = [preds_optimized.cpu().numpy()]
        print(f"✓ Applied optimized thresholds to {logits_concat.shape[0]} predictions")

    preds = np.concatenate(preds, axis=0)
    if len(scores) > 0:  # Fixed: use len() instead of comparing with []
        scores = np.concatenate(scores, axis=0)
        topks =  np.concatenate(topks, axis=0)
    if len(evi_preds) > 0:  # Fixed: use len() instead of comparing with []
        evi_preds = np.concatenate(evi_preds, axis=0)

    # Check if features are chunked (have entity_map field)
    if len(features) > 0 and 'entity_map' in features[0]:
        print(f"\n=== Merging {len(features)} chunk predictions ===")
        from evaluation import merge_chunk_predictions

        # Prepare scores for merging (convert to list of arrays per feature)
        scores_list = []
        if len(scores) > 0:
            # Assume scores are parallel to predictions
            for i in range(len(preds)):
                scores_list.append(scores[i] if i < len(scores) else None)
        else:
            scores_list = None

        # Merge chunks back to original documents
        merged_features, merged_preds, merged_scores = merge_chunk_predictions(
            features, preds, scores_list
        )

        print(f"✓ Merged to {len(merged_features)} original documents")

        # Replace with merged versions
        features = merged_features
        preds = np.concatenate(merged_preds, axis=0) if len(merged_preds) > 0 else preds
        if merged_scores is not None and len(merged_scores) > 0:
            scores = np.concatenate(merged_scores, axis=0)

    official_results, results = to_official(id2rel, preds, features, evi_preds = evi_preds, scores = scores, topks = topks)

    detailed_metrics = None  # Initialize variable for detailed metrics
    if len(official_results) > 0:
        if tag == "test":
            if args.train_file:  # Removed args.do_train check - evaluate test set during inference too
                best_re, best_evi, best_re_ign, detailed_metrics = official_evaluate(official_results, args.data_dir, args.train_file, args.test_file)
            else:
                best_re = best_evi = best_re_ign = [0, 0, 0]
        else:
            best_re, best_evi, best_re_ign, detailed_metrics = official_evaluate(official_results, args.data_dir, args.train_file, args.dev_file)
    else:
        best_re = best_evi = best_re_ign = [0, 0, 0]

    # Define 'output' BEFORE trying to use it or print detailed metrics
    output = {
        tag + "_rel": [i * 100 for i in best_re],
        tag + "_rel_ign": [i * 100 for i in best_re_ign], 
        tag + "_evi": [i * 100 for i in best_evi],
    }
    scores_dict = {"dev_F1": best_re[-1] * 100, "dev_evi_F1": best_evi[-1] * 100, "dev_F1_ign": best_re_ign[-1] * 100}

        # Print detailed metrics to console if available
    if detailed_metrics is not None:
        print("\n--- Detailed Per-Relation Metrics (Sorted by Relation Name) ---")
        # Sort relations alphabetically
        sorted_relations = sorted(detailed_metrics['per_relation'].keys())
        rel_data = []
        for rel in sorted_relations:
            metrics = detailed_metrics['per_relation'][rel]
            rel_data.append({
                'Relation': rel,
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1': f"{metrics['f1']:.4f}",
                'TP': metrics['tp'],
                'FP': metrics['fp'],
                'FN': metrics['fn']
            })

        # Create DataFrame
        rel_df = pd.DataFrame(rel_data)

        # Print DataFrame as CSV for easy copying
        print(rel_df.to_csv(index=False)) # This is the key change

        # Print Micro and Macro Averages
        # Format them clearly for copying, also as CSV lines
        print("--- Aggregate Metrics ---") # Keep header, but subsequent lines are pure CSV
        macro = detailed_metrics['macro_avg']
        micro = detailed_metrics['micro_avg']

        # Print aggregate metrics as CSV rows
        # Option 1: Simple labeled rows
        print("Metric Type,Precision,Recall,F1") # CSV Header for aggregate section
        print(f"Macro Avg,{macro['precision']:.4f},{macro['recall']:.4f},{macro['f1']:.4f}")
        print(f"Micro Avg,{micro['precision']:.4f},{micro['recall']:.4f},{micro['f1']:.4f}")

    if args.save_attn:
        attns_path = os.path.join(args.load_path, f"{os.path.splitext(args.test_file)[0]}.attns")        
        print(f"saving attentions into {attns_path} ...")
        with open(attns_path, "wb") as f:
            pickle.dump(attns, f)

    # Return the scores dictionary, output, official_results, results, and detailed_metrics
    return scores_dict, output, official_results, results, detailed_metrics

def dump_to_file(offi:list, offi_path: str, scores: list, score_path: str, results: list = [], res_path: str = "", thresh: float = None, detailed_metrics: dict = None):
    '''
    dump scores and (top-k) predictions to file.
    
    '''
    print(f"saving official predictions into {offi_path} ...")
    json.dump(offi, open(offi_path, "w"))
    
    print(f"saving evaluations into {score_path} ...")
    headers = ["precision", "recall", "F1"]
    scores_pd = pd.DataFrame.from_dict(scores, orient="index", columns = headers)
    print(scores_pd)
    scores_pd.to_csv(score_path, sep='\t')

        # Save detailed per-relation metrics if provided
    if detailed_metrics is not None:
        detailed_path = os.path.splitext(score_path)[0] + "_detailed.csv"
        print(f"saving detailed per-relation metrics into {detailed_path} ...")
        
        # Prepare data for CSV
        csv_data = []
        # Add per-relation data
        for rel, metrics in detailed_metrics['per_relation'].items():
            csv_data.append({
                'Type': 'Per-Relation',
                'Relation/Aggregate': rel,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'TP': metrics['tp'],
                'FP': metrics['fp'],
                'FN': metrics['fn']
            })
        # Add Macro Average
        macro = detailed_metrics['macro_avg']
        csv_data.append({
            'Type': 'Macro Average',
            'Relation/Aggregate': 'Macro',
            'Precision': macro['precision'],
            'Recall': macro['recall'],
            'F1': macro['f1'],
            'TP': '', 'FP': '', 'FN': ''
        })
        # Add Micro Average
        micro = detailed_metrics['micro_avg']
        csv_data.append({
            'Type': 'Micro Average',
            'Relation/Aggregate': 'Micro',
            'Precision': micro['precision'],
            'Recall': micro['recall'],
            'F1': micro['f1'],
            'TP': micro['tp'],
            'FP': micro['fp'],
            'FN': micro['fn']
        })
        
        # Create and save DataFrame
        detailed_df = pd.DataFrame(csv_data)
        detailed_df.to_csv(detailed_path, index=False)

    if len(results) != 0:
        assert res_path != ""
        print(f"saving topk results into {res_path} ...")
        json.dump(results, open(res_path, "w"))
    
    if thresh != None:
        thresh_path = os.path.join(os.path.dirname(offi_path), "thresh")
        if not os.path.exists(thresh_path):
            print(f"saving threshold into {thresh_path} ...")
            json.dump(thresh, open(thresh_path, "w"))        

    return


def main():
    


    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
        
    # wandb.init(project="DocRED", name=args.display_name)

    # create directory to save checkpoints and predicted files
    time = str(datetime.datetime.now()).replace(' ','_').replace(':','-')
    save_path_ = os.path.join(args.save_path, f"{time}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    print(device)
    print(torch.cuda.device_count())

    rel2id = json.load(open(os.path.join(args.data_dir, 'rel2id.json'), 'r'))
    id2rel = {value: key for key, value in rel2id.items()}

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    # Add entity marker '*' to tokenizer vocabulary
    # This ensures the marker is treated as a single token for entity boundary detection
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': ['*']})

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    # Resize model embeddings to accommodate new token
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"✓ Added {num_added} entity marker token(s) to vocabulary (total vocab size: {len(tokenizer)})")

    config.transformer_type = args.transformer_type

    set_seed(args)
    
    read = read_docred
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id

    # Initialize loss function based on args
    if args.use_focal_loss:
        print("\n=== Using Focal Loss ===")
        print(f"  gamma (focusing parameter): {args.focal_gamma}")
        print(f"  alpha (positive class weight): {args.focal_alpha}")
        print(f"  Expected: +1-3% F1 improvement over ATLoss")
        print(f"  Stability: Very safe (no gradient explosion)")

        loss_fnt = FocalLoss(
            gamma=args.focal_gamma,
            alpha=args.focal_alpha
        )
    elif args.use_amtl:
        print("\n=== Using Enhanced AMTL Loss ===")
        print(f"  num_segments: {args.num_segments}")
        print(f"  lambda_weight: {args.lambda_weight}")
        print(f"  use_effective_number: {args.use_effective_number}")
        print(f"  beta: {args.beta}")

        loss_fnt = EnhancedAMTLoss(
            num_classes=args.num_class,
            num_segments=args.num_segments,
            lambda_weight=args.lambda_weight,
            use_effective_number=args.use_effective_number,
            beta=args.beta
        )
    else:
        print("\n=== Using Standard ATLoss ===")
        loss_fnt = ATLoss()

    model = DocREModel(config, model, tokenizer,
                    emb_size=config.hidden_size,  # Explicit: use backbone's hidden size
                    num_labels=args.num_labels,
                    max_sent_num=args.max_sent_num,
                    evi_thresh=args.evi_thresh,
                    loss_fnt=loss_fnt)
    model.to(args.device)

    # Load checkpoint BEFORE compiling (if loading from checkpoint)
    if args.load_path != "": # load model from existing checkpoint

        model_path = os.path.join(args.load_path, "best.ckpt")
        print(f"Loading checkpoint from: {model_path}")

        # Load state dict
        state_dict = torch.load(model_path, map_location=args.device)

        # Handle compiled model checkpoints (strip _orig_mod. prefix if present)
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            print("✓ Detected compiled model checkpoint, stripping _orig_mod. prefix...")
            state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}

        model.load_state_dict(state_dict)
        print("✓ Checkpoint loaded successfully")

    # Enable torch.compile for faster inference (PyTorch 2.0+)
    # Note: Disabled by default due to large compilation overhead with small batches
    # Note: Must be done AFTER loading checkpoint to avoid key mismatch
    if hasattr(torch, 'compile') and not args.do_train and getattr(args, 'use_compile', False):
        print("✓ Compiling model with torch.compile for faster inference...")
        model = torch.compile(model)
        print("✓ Model compiled successfully")

    if args.do_train:  # Training

        # Enable gradient checkpointing for memory efficiency during training
        model.enable_gradient_checkpointing()

        create_directory(save_path_)
        args.save_path = save_path_

        train_file = os.path.join(args.data_dir, args.train_file)
        dev_file = os.path.join(args.data_dir, args.dev_file)

        train_features = read(args.data_dir, train_file, tokenizer, transformer_type=args.transformer_type, max_seq_length=args.max_seq_length, teacher_sig_path=args.teacher_sig_path, use_chunking=args.use_chunking, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, max_sent_num=args.max_sent_num)
        dev_features = read(args.data_dir, dev_file, tokenizer, transformer_type=args.transformer_type, max_seq_length=args.max_seq_length, use_chunking=args.use_chunking, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, max_sent_num=args.max_sent_num)

        # Apply negative sampling if enabled
        if args.use_negative_sampling:
            print(f"\n=== Applying Negative Sampling (ratio {args.neg_pos_ratio}:1) ===")
            train_features = negative_sampling(
                train_features,
                neg_pos_ratio=args.neg_pos_ratio,
                seed=args.seed
            )

        # Compute relation frequencies and initialize AMTL if enabled
        if args.use_amtl:
            print("\n=== Computing Relation Frequencies for AMTL ===")
            relation_frequencies = compute_relation_frequencies(train_features, num_classes=args.num_class)
            model.loss_fnt.set_relation_frequencies(relation_frequencies)
            print("✓ AMTL loss initialized with relation frequencies")

        train(args, model, train_features, dev_features, id2rel)

    else:  # Testing

        basename = os.path.splitext(args.test_file)[0]
        test_file = os.path.join(args.data_dir, args.test_file)

        test_features = read(args.data_dir, test_file, tokenizer, transformer_type=args.transformer_type, max_seq_length=args.max_seq_length, use_chunking=args.use_chunking, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, max_sent_num=args.max_sent_num)

        # Load optimized thresholds if they exist and per-class thresholds are enabled
        optimized_thresholds = None
        if args.use_per_class_thresholds:
            threshold_path = os.path.join(args.load_path, "optimized_thresholds.json")
            if os.path.exists(threshold_path):
                print(f"\n=== Loading Optimized Per-Class Thresholds ===")
                optimized_thresholds = load_optimized_thresholds(threshold_path)
                print(f"✓ Loaded optimized thresholds from {threshold_path}")
            else:
                print(f"⚠ Warning: --use_per_class_thresholds enabled but no optimized_thresholds.json found in {args.load_path}")
                print(f"  Will use standard threshold from loss function")

        # print(test_features[0]['sent_labels'])

        if args.eval_mode != "fushion":

            test_scores, test_output, official_results, results, detailed_metrics = evaluate(args, model, test_features, id2rel, tag="test", optimized_thresholds=optimized_thresholds)
            # wandb.log(test_scores)

            offi_path = os.path.join(args.load_path, args.pred_file)
            score_path = os.path.join(args.load_path, f"{basename}_scores.csv")
            res_path = os.path.join(args.load_path, f"topk_{args.pred_file}")

            dump_to_file(official_results, offi_path, test_output, score_path, results, res_path, detailed_metrics=detailed_metrics)          

        else: # inference stage fusion

            results = json.load(open(os.path.join(args.load_path, f"topk_{args.pred_file}")))

            # formulate pseudo documents from top-k (k=num_labels in arguments) predictions
            pseudo_test_features = read(args.data_dir, test_file, tokenizer, max_seq_length=args.max_seq_length, single_results = results)
            
        
            pseudo_test_scores, pseudo_output, pseudo_official_results, pseudo_results = evaluate(args, model, pseudo_test_features, id2rel, tag="test")

            if 'thresh' in os.listdir(args.load_path):
                with open(os.path.join(args.load_path, "thresh")) as f:
                    thresh = json.load(f)
                print(f"Threshold loaded from file: {thresh}")
            else:
                thresh = None
            
            merged_offi, thresh = merge_results(id2rel, results, pseudo_results, test_features, thresh)
            if args.do_train:
                merged_re, merged_evi, merged_re_ign, merged_detailed_metrics = official_evaluate(merged_offi, args.data_dir, args.train_file, args.test_file)
            else:
                merged_re = merged_evi = merged_re_ign = [0, 0, 0]
                merged_detailed_metrics = None

            tag = args.test_file.split('.')[0]
            merged_output = {
                tag + "_rel": [i * 100 for i in merged_re],
                tag + "_rel_ign": [i * 100 for i in merged_re_ign],
                tag + "_evi": [i * 100 for i in merged_evi],
            }
            wandb.log({"dev_F1": merged_re[-1] * 100, "dev_evi_F1": merged_evi[-1] * 100, "dev_F1_ign": merged_re_ign[-1] * 100})
            offi_path = os.path.join(args.load_path, f"fused_{args.pred_file}")
            score_path = os.path.join(args.load_path, f"{basename}_fused_scores.csv")
            dump_to_file(merged_offi, offi_path, merged_output, score_path, thresh=thresh, detailed_metrics=merged_detailed_metrics)


if __name__ == "__main__":
    main()
