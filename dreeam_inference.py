import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

from model import DocREModel
from utils import collate_fn
from evaluation import to_official, load_optimized_thresholds, apply_per_class_thresholds, merge_chunk_predictions
from prepro import chunk_document


class DreeamInference:
    """
    DREEAM inference class for bulk relation extraction from raw texts.
    
    This class provides a simplified interface for performing relation extraction
    on raw text documents with named entities, outputting simple tuples of
    entity spans and their relations.
    """
    
    def __init__(self, 
                 config_path: str,
                 device: str = "auto"):
        """
        Initialize the DREEAM inference model.
        
        Args:
            config_path: Path to configuration file
            device: Device to run inference on ("auto", "cuda", "cpu")
        """
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config_data = json.load(f)
        
        # Extract configuration parameters
        self.model_path = self.config_data.get("model_path", "./logs/nerel-ckpt")
        self.max_seq_length = self.config_data.get("max_seq_length", 1024)
        self.max_sent_num = self.config_data.get("max_sent_num", 25)
        self.evi_thresh = self.config_data.get("evi_thresh", 0.2)
        self.num_labels = self.config_data.get("num_labels", 4)
        self.batch_size = self.config_data.get("batch_size", 8)

        # Chunking parameters (for long documents)
        self.use_chunking = self.config_data.get("use_chunking", False)
        self.chunk_size = self.config_data.get("chunk_size", 512)
        self.chunk_overlap = self.config_data.get("chunk_overlap", 128)

        # Per-class thresholds (loaded later after rel2id is available)
        self.optimized_thresholds = None
        
        # Load relation mappings
        rel2id_path = self.config_data.get("rel2id", "rel2id.json")
        if not os.path.isabs(rel2id_path):
            rel2id_path = os.path.join(os.path.dirname(config_path), rel2id_path)
        
        # Try to find rel2id file locally
        if not os.path.exists(rel2id_path):
            # Try alternative locations
            alt_paths = [
                os.path.join(os.path.dirname(config_path), "rel2id.json"),
                "rel2id.json",
                os.path.join("rel2id", "nerel-rel2id.json")
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    rel2id_path = alt_path
                    break

        # Load rel2id from local file or download from HuggingFace
        if os.path.exists(rel2id_path):
            with open(rel2id_path, 'r', encoding='utf-8') as f:
                self.rel2id = json.load(f)
                self.id2rel = {v: k for k, v in self.rel2id.items()}
        else:
            # Try downloading from HuggingFace
            rel2id_path = self._download_rel2id(self.model_path)
            if rel2id_path and os.path.exists(rel2id_path):
                with open(rel2id_path, 'r', encoding='utf-8') as f:
                    self.rel2id = json.load(f)
                    self.id2rel = {v: k for k, v in self.rel2id.items()}
            else:
                raise FileNotFoundError(f"Relation mapping file not found: {rel2id_path}")
        
        # Initialize model
        self._load_model()
        
    def _load_model(self):
        """Load the trained model and tokenizer."""
        
        model_name = self.config_data.get("model_name_or_path", "DeepPavlov/rubert-base-cased")
        transformer_type = self.config_data.get("transformer_type", "bert")
        num_class = self.config_data.get("num_class", len(self.rel2id))
        
        # Load configuration and tokenizer
        config = AutoConfig.from_pretrained(model_name, num_labels=num_class)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add entity marker '*' to tokenizer vocabulary
        # This ensures the marker is treated as a single token for entity boundary detection
        num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': ['*']})

        # Load transformer model
        transformer_model = AutoModel.from_pretrained(model_name, config=config)

        # Resize model embeddings to accommodate new token
        if num_added > 0:
            transformer_model.resize_token_embeddings(len(self.tokenizer))
            print(f"✓ Added {num_added} entity marker token(s) to vocabulary (total vocab size: {len(self.tokenizer)})")
        
        # Set transformer type
        config.transformer_type = transformer_type
        config.cls_token_id = self.tokenizer.cls_token_id
        config.sep_token_id = self.tokenizer.sep_token_id
        
        # Create DocRE model
        self.model = DocREModel(
            config,
            transformer_model,
            self.tokenizer,
            emb_size=config.hidden_size,  # Explicit: use backbone's hidden size
            num_labels=self.num_labels,
            max_sent_num=self.max_sent_num,
            evi_thresh=self.evi_thresh
        )
        
        # Initialize position_ids if they don't exist (compatibility fix)
        if hasattr(self.model.model, 'embeddings') and not hasattr(self.model.model.embeddings, 'position_ids'):
            self.model.model.embeddings.register_buffer(
                "position_ids", 
                torch.arange(config.max_position_embeddings).expand((1, -1))
            )
        
        # Load trained weights
        checkpoint_path = self._get_checkpoint_path(self.model_path)

        if checkpoint_path:
            # Load state dict and inspect architecture
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # Detect checkpoint architecture from state_dict
            checkpoint_vocab_size = None
            checkpoint_emb_size = None

            if 'model.embeddings.word_embeddings.weight' in state_dict:
                checkpoint_vocab_size = state_dict['model.embeddings.word_embeddings.weight'].shape[0]

            if 'head_extractor.weight' in state_dict:
                checkpoint_emb_size = state_dict['head_extractor.weight'].shape[0]

            # Adjust tokenizer if vocab size mismatch
            if checkpoint_vocab_size and checkpoint_vocab_size != len(self.tokenizer):
                print(f"Note: Checkpoint vocab size ({checkpoint_vocab_size}) differs from current ({len(self.tokenizer)})")
                print(f"  Adjusting tokenizer to match checkpoint...")

                # Resize tokenizer and model to match checkpoint
                while len(self.tokenizer) < checkpoint_vocab_size:
                    self.tokenizer.add_special_tokens({'additional_special_tokens': [f'<unused{len(self.tokenizer)}>']})

                transformer_model.resize_token_embeddings(len(self.tokenizer))
                print(f"  ✓ Tokenizer adjusted to {len(self.tokenizer)} tokens")

            # Recreate model if emb_size mismatch
            if checkpoint_emb_size and checkpoint_emb_size != config.hidden_size:
                print(f"Note: Checkpoint emb_size ({checkpoint_emb_size}) differs from config hidden_size ({config.hidden_size})")
                print(f"  Recreating model with emb_size={checkpoint_emb_size}...")

                self.model = DocREModel(
                    config,
                    transformer_model,
                    self.tokenizer,
                    emb_size=checkpoint_emb_size,  # Use checkpoint's emb_size
                    num_labels=self.num_labels,
                    max_sent_num=self.max_sent_num,
                    evi_thresh=self.evi_thresh
                )
                print(f"  ✓ Model recreated with emb_size={checkpoint_emb_size}")

            # Handle version compatibility issues
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"Warning: Missing keys in state_dict: {missing_keys}")
                # Handle specific missing keys that are common in version mismatches
                if any("position_ids" in key for key in missing_keys):
                    print("Note: position_ids missing - this is normal for newer transformers versions")

            if unexpected_keys:
                print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")

            print(f"✓ Loaded model from: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found in: {self.model_path}")
        
        self.model.to(self.device)
        self.model.eval()

        # Enable torch.compile for 30-50% faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            print("✓ Compiling model with torch.compile for faster inference...")
            self.model = torch.compile(self.model)
            print("✓ Model compiled successfully")

        # Load per-class thresholds if available
        self._load_optimized_thresholds()

    def _download_rel2id(self, repo_id: str) -> Optional[str]:
        """
        Download rel2id.json from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID

        Returns:
            Path to downloaded file, or None if failed
        """
        try:
            print(f"Attempting to download rel2id.json from HuggingFace: {repo_id}")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename="rel2id.json",
                cache_dir=os.path.expanduser("~/.cache/huggingface/dreeam")
            )
            print(f"✓ Successfully downloaded rel2id.json from HuggingFace Hub")
            return downloaded_path
        except Exception as e:
            print(f"Warning: Could not download rel2id.json from HuggingFace: {e}")
            return None

    def _get_checkpoint_path(self, model_path: str) -> Optional[str]:
        """
        Get checkpoint path from local filesystem or download from HuggingFace Hub.

        Args:
            model_path: Path to model checkpoint (local or HuggingFace repo ID)

        Returns:
            Path to checkpoint file, or None if not found
        """
        # Try local path first
        if os.path.exists(model_path):
            # If it's a file, use it directly
            if os.path.isfile(model_path):
                return model_path
            # If it's a directory, look for common checkpoint names
            else:
                for ckpt_name in ["best.ckpt", "last.ckpt", "model.ckpt", "pytorch_model.bin"]:
                    ckpt_path = os.path.join(model_path, ckpt_name)
                    if os.path.exists(ckpt_path):
                        return ckpt_path

        # Not found locally, try to download from HuggingFace Hub
        print(f"Local checkpoint not found, attempting to download from HuggingFace: {model_path}")

        try:
            # Try downloading common checkpoint filenames
            for filename in ["model.ckpt", "best.ckpt", "last.ckpt", "pytorch_model.bin", "model.safetensors"]:
                try:
                    print(f"  Attempting to download: {filename}...")
                    downloaded_path = hf_hub_download(
                        repo_id=model_path,
                        filename=filename,
                        cache_dir=os.path.expanduser("~/.cache/huggingface/dreeam")
                    )
                    print(f"✓ Successfully downloaded {filename} from HuggingFace Hub")
                    return downloaded_path
                except Exception as e:
                    # Try next filename
                    continue

            # If none of the common names worked, raise error
            raise FileNotFoundError(
                f"Could not find checkpoint in HuggingFace repo '{model_path}'. "
                f"Tried: model.ckpt, best.ckpt, last.ckpt, pytorch_model.bin, model.safetensors"
            )

        except Exception as e:
            print(f"Error downloading from HuggingFace: {e}")
            return None

    def _load_optimized_thresholds(self):
        """Load per-class optimized thresholds if available."""
        # Try to load from config first
        threshold_path = self.config_data.get("optimized_thresholds")

        # If not in config, try default location in model directory
        if not threshold_path:
            threshold_path = os.path.join(self.model_path, "optimized_thresholds.json")

        # Make path absolute if relative
        if threshold_path and not os.path.isabs(threshold_path):
            threshold_path = os.path.join(os.path.dirname(self.model_path), threshold_path)

        # Load thresholds if file exists
        if threshold_path and os.path.exists(threshold_path):
            self.optimized_thresholds = load_optimized_thresholds(threshold_path)
            print(f"✓ Loaded optimized thresholds from: {threshold_path}")
            print(f"  Per-class thresholds available for {len(self.optimized_thresholds)} relation types")
        else:
            self.optimized_thresholds = None
            if self.config_data.get("optimized_thresholds"):
                print(f"Warning: Optimized thresholds not found at: {threshold_path}")

    def _convert_to_docred_format(self, 
                                  text: str, 
                                  entities: List[Dict],
                                  title: str = "document") -> Dict:
        """
        Convert raw text and entities to DocRED format.
        
        Args:
            text: Raw input text
            entities: List of entity dictionaries with keys: 'text', 'start', 'end', 'type'
            title: Document title
            
        Returns:
            Dictionary in DocRED format
        """
        
        # Split text into sentences by periods
        sentences = []
        current_pos = 0
        for i, char in enumerate(text):
            if char == '.' and i < len(text) - 1:
                sentence = text[current_pos:i+1].strip()
                if sentence:
                    sentences.append(sentence)
                current_pos = i + 1
        
        # Add remaining text as last sentence if exists
        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if remaining:
                sentences.append(remaining)
        
        # Tokenize sentences
        tokenized_sents = []
        char_to_sent_token = {}  # Map char positions to (sent_id, token_id)
        
        current_char = 0
        for sent_id, sentence in enumerate(sentences):
            # Find sentence start in original text
            sent_start = text.find(sentence, current_char)
            if sent_start == -1:
                sent_start = current_char
            
            # Tokenize sentence by splitting on whitespace
            words = sentence.split()
            tokenized_sents.append(words)
            
            # Map character positions to token positions
            word_start = sent_start
            for token_id, word in enumerate(words):
                word_pos = text.find(word, word_start)
                if word_pos != -1:
                    for char_idx in range(word_pos, word_pos + len(word)):
                        char_to_sent_token[char_idx] = (sent_id, token_id)
                    word_start = word_pos + len(word)
            
            current_char = sent_start + len(sentence)
        
                # Group entities by their text (handle coreference)
        entity_groups = {}
        for entity in entities:
            entity_text = entity['text']
            if entity_text not in entity_groups:
                entity_groups[entity_text] = []
            
            # Find token positions for this entity
            start_char = entity['start']
            end_char = entity['end']
            
            # Validate character positions
            if start_char >= end_char or start_char < 0 or end_char > len(text):
                continue
            
            # Find the sentence and token positions
            sent_id = None
            start_token = None
            end_token = None
            
            # Look for the sentence containing this entity
            found_positions = False
            for char_pos in range(start_char, min(end_char, len(text))):
                if char_pos in char_to_sent_token:
                    s_id, t_id = char_to_sent_token[char_pos]
                    if sent_id is None:
                        sent_id = s_id
                        start_token = t_id
                    if s_id == sent_id:
                        end_token = t_id + 1  # Exclusive end
                        found_positions = True
            
            # Fallback: if we can't find exact positions, try to find the entity in sentences
            if not found_positions:
                for s_id, sent_tokens in enumerate(tokenized_sents):
                    for t_id, token in enumerate(sent_tokens):
                        if entity_text.lower() in token.lower() or token.lower() in entity_text.lower():
                            sent_id = s_id
                            start_token = t_id
                            end_token = t_id + 1
                            found_positions = True
                            break
                    if found_positions:
                        break
                    
            if sent_id is not None and start_token is not None and end_token is not None:
                # Ensure valid token positions
                if (sent_id < len(tokenized_sents) and 
                    start_token < len(tokenized_sents[sent_id]) and 
                    end_token <= len(tokenized_sents[sent_id]) and
                    start_token < end_token):
                    
                    entity_groups[entity_text].append({
                        'name': entity['text'],
                        'sent_id': sent_id,
                        'pos': [start_token, end_token],
                        'type': entity.get('type', 'ENTITY')
                    })
        
        # Create vertex set from entity groups
        # Filter out empty entity groups (entities that couldn't be mapped to tokens)
        vertex_set = [group for group in entity_groups.values() if len(group) > 0]

        # Debug: warn if entities were skipped
        num_skipped = len(entity_groups) - len(vertex_set)
        if num_skipped > 0:
            print(f"Warning: Skipped {num_skipped} entities that couldn't be mapped to tokens in '{title}'")

        return {
            'title': title,
            'sents': tokenized_sents,
            'vertexSet': vertex_set
        }
    
    def _add_entity_markers(self, sample: Dict) -> Tuple[List[str], List[Dict], List[Tuple]]:
        """
        Add entity markers (*) to the beginning and end of entities.
        
        Args:
            sample: Document in DocRED format
            
        Returns:
            Tuple of (tokenized_text, sentence_map, sentence_positions)
        """
        entity_start, entity_end = [], []
        
        # Record entity positions
        for entity in sample['vertexSet']:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0]))
                entity_end.append((sent_id, pos[1] - 1))
        
        sents = []
        sent_map = []
        sent_pos = []
        
        sent_start = 0
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            
            for i_t, token in enumerate(sent):
                tokens_wordpiece = self.tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            
            sent_end = len(sents)
            sent_pos.append((sent_start, sent_end))
            sent_start = sent_end
            
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)
        
        return sents, sent_map, sent_pos
    
    def _create_features(self, samples: List[Dict]) -> List[Dict]:
        """
        Create features for inference from DocRED format samples.
        
        Args:
            samples: List of documents in DocRED format
            
        Returns:
            List of feature dictionaries for model input
        """
        features = []
        
        for sample_idx, sample in enumerate(samples):
            # Add entity markers
            sents, sent_map, sent_pos = self._add_entity_markers(sample)

            # Check if document needs chunking
            if self.use_chunking and len(sents) > self.chunk_size:
                print(f'Chunking document "{sample["title"][:50]}..." ({len(sents)} tokens) into chunks of {self.chunk_size} with overlap {self.chunk_overlap}')

                # First, build entity_pos and train_triple for chunk_document
                entity_pos_temp = []
                for entity in sample['vertexSet']:
                    current_entity_pos = []
                    for mention in entity:
                        sent_id = mention["sent_id"]
                        pos = mention["pos"]
                        if (sent_id < len(sent_map) and
                            pos[0] in sent_map[sent_id] and
                            pos[1] in sent_map[sent_id]):
                            start = sent_map[sent_id][pos[0]]
                            end = sent_map[sent_id][pos[1]]
                            if start < end:
                                current_entity_pos.append((start, end))
                    if current_entity_pos:
                        entity_pos_temp.append(current_entity_pos)

                # Create dummy train_triple (not used in inference but needed for chunk_document)
                train_triple = {}

                # Call chunk_document from prepro.py
                chunks = chunk_document(
                    sents=sents,
                    entity_pos=entity_pos_temp,
                    train_triple=train_triple,
                    sent_pos=sent_pos,
                    sample=sample,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    docred_rel2id=self.rel2id,
                    max_sent_num=self.max_sent_num
                )

                if chunks is not None:
                    # Convert chunks to features
                    for chunk in chunks:
                        chunk_input_ids = self.tokenizer.convert_tokens_to_ids(chunk['chunk_tokens'])
                        chunk_input_ids = self.tokenizer.build_inputs_with_special_tokens(chunk_input_ids)

                        # Create dummy labels
                        labels = [[1] + [0] * (len(self.rel2id) - 1) for _ in chunk['chunk_hts']]

                        chunk_feature = {
                            'input_ids': chunk_input_ids,
                            'entity_pos': chunk['chunk_entity_pos'],
                            'labels': labels,
                            'hts': chunk['chunk_hts'],
                            'sent_pos': chunk['chunk_sent_pos'],
                            'sent_labels': None,
                            'title': sample['title'],
                            'entity_map': chunk['entity_map'],  # Important: mark as chunked
                            'original_title': sample['title']
                        }
                        features.append(chunk_feature)
                    continue  # Skip normal processing for this sample

            # Normal processing (no chunking)
            # Truncate if too long
            if len(sents) > self.max_seq_length - 2:
                sents = sents[:self.max_seq_length - 2]

            # Convert to input IDs
            input_ids = self.tokenizer.convert_tokens_to_ids(sents)
            input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)

            # Create entity positions
            entity_pos = []
            valid_entities = []

            for entity_idx, entity in enumerate(sample['vertexSet']):
                current_entity_pos = []
                for mention in entity:
                    sent_id = mention["sent_id"]
                    pos = mention["pos"]
                    # Check bounds more carefully
                    if (sent_id < len(sent_map) and
                        pos[0] in sent_map[sent_id] and
                        pos[1] in sent_map[sent_id] and
                        len(pos) >= 2):
                        start = sent_map[sent_id][pos[0]]
                        end = sent_map[sent_id][pos[1]]
                        # Ensure start < end and within bounds
                        if start < end and start < len(input_ids) and end <= len(input_ids):
                            current_entity_pos.append((start, end))

                # Only add entities that have valid positions
                if current_entity_pos:
                    entity_pos.append(current_entity_pos)
                    valid_entities.append(entity_idx)

            # Create all possible entity pairs for inference (only between valid entities)
            hts = []
            num_valid_entities = len(entity_pos)

            if num_valid_entities > 1:  # Need at least 2 entities for relations
                for h in range(num_valid_entities):
                    for t in range(num_valid_entities):
                        if h != t:
                            hts.append([h, t])

            # Skip this sample if no valid entity pairs
            if not hts:
                # Create dummy data to avoid empty batch issues
                entity_pos = [[(0, 1)]]  # Dummy entity
                hts = [[0, 0]]  # Dummy pair (will be filtered out later)

            # Create dummy labels (not used during inference)
            labels = [[1] + [0] * (len(self.rel2id) - 1) for _ in hts]

            feature = {
                'input_ids': input_ids,
                'entity_pos': entity_pos,
                'labels': labels,
                'hts': hts,
                'sent_pos': sent_pos,
                'sent_labels': None,
                'title': sample['title']
            }

            features.append(feature)
        
        return features
    
    def _load_input(self, batch, tag="test"):
        """Load batch input for model inference."""
        return {
            'input_ids': batch[0].to(self.device),
            'attention_mask': batch[1].to(self.device),
            'labels': batch[2].to(self.device),
            'entity_pos': batch[3],
            'hts': batch[4],
            'sent_pos': batch[5],
            'sent_labels': batch[6].to(self.device) if batch[6] is not None else None,
            'teacher_attns': batch[7].to(self.device) if batch[7] is not None else None,
            'tag': tag
        }
    
    def predict_relations(self, 
                         texts: List[str], 
                         entities_list: List[List[Dict]],
                         titles: Optional[List[str]] = None) -> List[List[Tuple]]:
        """
        Predict relations for a batch of texts.
        
        Args:
            texts: List of input text strings
            entities_list: List of entity lists, where each entity list contains
                          dictionaries with keys: 'text', 'start', 'end', 'type'
            titles: Optional list of document titles
            
        Returns:
            List of relation predictions for each document, where each prediction
            is a tuple of (head_entity_text, tail_entity_text, relation_type)
        """
        
        if titles is None:
            titles = [f"document_{i}" for i in range(len(texts))]
        
        # Convert to DocRED format
        samples = []
        for i, (text, entities) in enumerate(zip(texts, entities_list)):
            sample = self._convert_to_docred_format(text, entities, titles[i])
            samples.append(sample)
        
        # Create features
        features = self._create_features(samples)
        
        # Run inference
        dataloader = DataLoader(features, batch_size=self.batch_size, shuffle=False,
                               collate_fn=collate_fn, drop_last=False,
                               num_workers=4, pin_memory=True, prefetch_factor=2)

        all_preds = []
        all_scores = []
        all_topks = []
        all_logits = []  # Collect logits for per-class thresholds

        with torch.no_grad():
            for batch in dataloader:
                inputs = self._load_input(batch, tag="test")

                outputs = self.model(**inputs)

                # Collect logits if we have per-class thresholds
                if self.optimized_thresholds is not None and "logits" in outputs:
                    all_logits.append(outputs["logits"].cpu())

                pred = outputs["rel_pred"].cpu().numpy()
                pred[np.isnan(pred)] = 0
                all_preds.append(pred)

                if "scores" in outputs:
                    all_scores.append(outputs["scores"].cpu().numpy())
                    all_topks.append(outputs["topks"].cpu().numpy())

        # Concatenate predictions
        preds = np.concatenate(all_preds, axis=0)
        scores = np.concatenate(all_scores, axis=0) if len(all_scores) > 0 else []
        topks = np.concatenate(all_topks, axis=0) if len(all_topks) > 0 else []

        # Apply per-class thresholds if available
        if self.optimized_thresholds is not None and len(all_logits) > 0:
            logits_concat = torch.cat(all_logits, dim=0)
            preds_optimized = apply_per_class_thresholds(
                logits_concat,
                self.optimized_thresholds,
                num_labels=self.num_labels
            )
            preds = preds_optimized.cpu().numpy()
            print(f"✓ Applied optimized thresholds to {logits_concat.shape[0]} predictions")

        # Check if features are chunked (have entity_map field)
        if len(features) > 0 and 'entity_map' in features[0]:
            print(f"\n=== Merging {len(features)} chunk predictions ===")

            # Prepare scores for merging (convert to list of arrays per feature)
            scores_list = []
            if len(scores) > 0:
                idx = 0
                for feat in features:
                    num_pairs = len(feat['hts'])
                    scores_list.append(scores[idx:idx + num_pairs])
                    idx += num_pairs

            # Merge chunk predictions
            merged_features, merged_preds, merged_scores = merge_chunk_predictions(
                features=features,
                preds=preds,
                scores=scores_list if len(scores_list) > 0 else None,
                id2rel=self.id2rel,
                merge_strategy='union'
            )

            # Update features and predictions with merged results
            features = merged_features
            preds = merged_preds

            if merged_scores is not None and len(merged_scores) > 0:
                scores = np.concatenate(merged_scores, axis=0)

        # Convert to official format
        official_results, results = to_official(self.id2rel, preds, features,
                                              scores=scores, topks=topks)

        # Convert to simple tuple format
        return self._convert_to_simple_format(official_results, samples)
    
    def _convert_to_simple_format(self, 
                                 official_results: List[Dict], 
                                 samples: List[Dict]) -> List[List[Tuple]]:
        """
        Convert official results to simple tuple format.
        
        Args:
            official_results: Results in official format
            samples: Original samples for entity text lookup
            
        Returns:
            List of relation tuples for each document
        """
        # Group results by document title
        title_to_results = {}
        for result in official_results:
            title = result['title']
            if title not in title_to_results:
                title_to_results[title] = []
            title_to_results[title].append(result)
        
        # Convert to simple format
        simple_results = []
        for sample in samples:
            title = sample['title']
            doc_relations = []
            
            if title in title_to_results:
                for result in title_to_results[title]:
                    h_idx = result['h_idx']
                    t_idx = result['t_idx']
                    relation = result['r']
                    
                    # Get entity texts
                    if (h_idx < len(sample['vertexSet']) and t_idx < len(sample['vertexSet']) and
                        len(sample['vertexSet'][h_idx]) > 0 and len(sample['vertexSet'][t_idx]) > 0):
                        head_entity = sample['vertexSet'][h_idx][0]['name']  # First mention
                        tail_entity = sample['vertexSet'][t_idx][0]['name']  # First mention

                        # Skip "Na" (no relation) predictions
                        if relation != "Na":
                            doc_relations.append((head_entity, tail_entity, relation))
            
            simple_results.append(doc_relations)
        
        return simple_results
    
    def predict_single(self, 
                      text: str, 
                      entities: List[Dict],
                      title: str = "document") -> List[Tuple]:
        """
        Predict relations for a single document.
        
        Args:
            text: Input text string
            entities: List of entity dictionaries with keys: 'text', 'start', 'end', 'type'
            title: Document title
            
        Returns:
            List of relation tuples (head_entity_text, tail_entity_text, relation_type)
        """
        results = self.predict_relations([text], [entities], [title])
        return results[0] if results else []


# Example usage
def parse_brat_file(txt_path: str, ann_path: str) -> Tuple[str, List[Dict], str]:
    """
    Parse BRAT format files (.txt and .ann).

    Args:
        txt_path: Path to text file
        ann_path: Path to annotation file

    Returns:
        Tuple of (text, entities, file_id)
    """
    # Read text file
    try:
        with open(txt_path, 'r', encoding='utf-8', newline='') as f:
            text = f.read()
    except UnicodeDecodeError:
        # Try different encodings
        for encoding in ['cp1251', 'latin-1', 'utf-8-sig']:
            try:
                with open(txt_path, 'r', encoding=encoding, newline='') as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue

    # Parse entities from .ann file
    entities = []
    if os.path.exists(ann_path):
        with open(ann_path, 'r', encoding='utf-8', newline='') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith('T'):
                    continue

                parts = line.split('\t')
                if len(parts) < 3:
                    continue

                entity_info = parts[1].split()
                entity_text = parts[2]

                if len(entity_info) >= 3:
                    entity_type = entity_info[0]
                    start_pos = int(entity_info[1])
                    end_pos = int(entity_info[2])

                    entities.append({
                        'text': entity_text,
                        'start': start_pos,
                        'end': end_pos,
                        'type': entity_type
                    })

    file_id = os.path.splitext(os.path.basename(txt_path))[0]
    return text, entities, file_id


def format_relations_readable(relations: List[Tuple], sort_by_entity: bool = True) -> str:
    """
    Format relations in readable format: [head] -- relation -> [tail]

    Args:
        relations: List of (head, tail, relation) tuples
        sort_by_entity: If True, sort by entities first, then by relation

    Returns:
        Formatted string
    """
    if not relations:
        return "No relations predicted.\n"

    # Sort relations
    if sort_by_entity:
        # Sort by head entity, then tail entity, then relation type
        sorted_relations = sorted(relations, key=lambda x: (x[0], x[1], x[2]))
    else:
        # Sort by relation type, then entities
        sorted_relations = sorted(relations, key=lambda x: (x[2], x[0], x[1]))

    # Format output
    lines = []
    lines.append(f"Total relations: {len(sorted_relations)}\n")
    lines.append("-" * 80)

    for head, tail, relation in sorted_relations:
        lines.append(f"[{head}] -- {relation} -> [{tail}]")

    return "\n".join(lines)


def process_brat_files(test_dir: str,
                       output_dir: str,
                       inference_model: 'DreeamInference',
                       max_files: Optional[int] = None,
                       verbose: bool = False) -> None:
    """
    Process BRAT format files and generate relation predictions.

    Args:
        test_dir: Directory containing .txt and .ann files
        output_dir: Directory to write output files
        inference_model: Initialized DreeamInference instance
        max_files: Maximum number of files to process (for testing)
        verbose: Print verbose output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all .txt files
    txt_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.txt')])

    if max_files:
        txt_files = txt_files[:max_files]

    print(f"Found {len(txt_files)} files to process")

    # Process each file
    for txt_file in txt_files:
        txt_path = os.path.join(test_dir, txt_file)
        ann_path = os.path.join(test_dir, txt_file.replace('.txt', '.ann'))

        # Parse BRAT files
        text, entities, file_id = parse_brat_file(txt_path, ann_path)

        if verbose:
            print(f"\nProcessing: {file_id}")
            print(f"  Text length: {len(text)} chars")
            print(f"  Entities: {len(entities)}")

        # Skip if insufficient entities
        if len(entities) < 2:
            if verbose:
                print(f"  Skipping (insufficient entities)")
            continue

        # Predict relations
        try:
            relations = inference_model.predict_single(text, entities, file_id)

            if verbose:
                print(f"  Predicted relations: {len(relations)}")

            # Write output file
            output_path = os.path.join(output_dir, f"{file_id}_relations.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"File: {file_id}\n")
                f.write(f"Text: {text[:100]}...\n\n") if len(text) > 100 else f.write(f"Text: {text}\n\n")
                f.write(f"Entities ({len(entities)}):\n")

                # Sort entities by position
                sorted_entities = sorted(entities, key=lambda e: (e['start'], e['type'], e['end']))
                for i, entity in enumerate(sorted_entities):
                    f.write(f"  {i+1}. [{entity['type']}] {entity['text']} ({entity['start']}-{entity['end']})\n")

                f.write("\n")
                f.write(format_relations_readable(relations, sort_by_entity=True))

            if verbose:
                print(f"  ✓ Written to: {output_path}")

        except Exception as e:
            print(f"  Error processing {file_id}: {e}")
            continue

    print(f"\n✓ Processing complete. Output written to: {output_dir}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='DREEAM Relation Extraction Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process BRAT files from directory
  python dreeam_inference.py --test_dir ./data/test --output_dir ./output --config_path ./nerel-config.json

  # Process with verbose output
  python dreeam_inference.py --test_dir ./data/test --output_dir ./output --config_path ./nerel-config.json --verbose

  # Process only first 10 files (for testing)
  python dreeam_inference.py --test_dir ./data/test --output_dir ./output --config_path ./nerel-config.json --max_files 10
        """
    )

    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to directory containing .txt and .ann files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for predictions')
    parser.add_argument('--config_path', type=str, default='./nerel-config.json',
                        help='Path to model configuration file (default: ./nerel-config.json)')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to process (for testing)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for inference (default: auto)')

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory not found: {args.test_dir}")
        return 1

    # Validate config file
    if not os.path.exists(args.config_path):
        print(f"Error: Config file not found: {args.config_path}")
        return 1

    # Initialize inference model
    print("=" * 80)
    print("DREEAM Relation Extraction Inference")
    print("=" * 80)
    print(f"Config: {args.config_path}")
    print(f"Test directory: {args.test_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 80)
    print("\nInitializing model...")

    try:
        inference = DreeamInference(
            config_path=args.config_path,
            device=args.device
        )
        print("✓ Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Process files
    try:
        process_brat_files(
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            inference_model=inference,
            max_files=args.max_files,
            verbose=args.verbose
        )
        return 0
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())    