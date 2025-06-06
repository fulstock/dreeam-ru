import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.utils.data import DataLoader

from model import DocREModel
from utils import collate_fn
from evaluation import to_official


class DreeamInference:
    """
    DREEAM inference class for bulk relation extraction from raw texts.
    
    This class provides a simplified interface for performing relation extraction
    on raw text documents with named entities, outputting simple tuples of
    entity spans and their relations.
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = None,
                 device: str = "auto",
                 max_seq_length: int = 1024,
                 max_sent_num: int = 25,
                 evi_thresh: float = 0.2,
                 num_labels: int = 4,
                 batch_size: int = 8):
        """
        Initialize the DREEAM inference model.
        
        Args:
            model_path: Path to the trained model checkpoint directory
            config_path: Path to configuration file (optional, will look for dreeam-config.json)
            device: Device to run inference on ("auto", "cuda", "cpu")
            max_seq_length: Maximum sequence length for input
            max_sent_num: Maximum number of sentences per document
            evi_thresh: Evidence threshold for sentence selection
            num_labels: Maximum number of relation labels per entity pair
            batch_size: Batch size for inference
        """
        
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.max_sent_num = max_sent_num
        self.evi_thresh = evi_thresh
        self.num_labels = num_labels
        self.batch_size = batch_size
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load configuration
        if config_path is None:
            config_path = os.path.join(model_path, "dreeam-config.json")
            if not os.path.exists(config_path):
                config_path = "dreeam-config.json"
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load relation mappings
        rel2id_path = os.path.join(os.path.dirname(config_path), "rel2id.json")
        if not os.path.exists(rel2id_path):
            rel2id_path = os.path.join(self.config_data.get("data_dir", "."), "rel2id.json")
        
        if os.path.exists(rel2id_path):
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
        
        # Load transformer model
        transformer_model = AutoModel.from_pretrained(model_name, config=config)
        
        # Set transformer type
        config.transformer_type = transformer_type
        config.cls_token_id = self.tokenizer.cls_token_id
        config.sep_token_id = self.tokenizer.sep_token_id
        
        # Create DocRE model
        self.model = DocREModel(
            config, 
            transformer_model, 
            self.tokenizer,
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
        checkpoint_path = os.path.join(self.model_path, "best.ckpt")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(self.model_path, "last.ckpt")
        
        if os.path.exists(checkpoint_path):
            # Load state dict and handle missing keys gracefully
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle version compatibility issues
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in state_dict: {missing_keys}")
                # Handle specific missing keys that are common in version mismatches
                if any("position_ids" in key for key in missing_keys):
                    print("Note: position_ids missing - this is normal for newer transformers versions")
            
            if unexpected_keys:
                print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
            
            print(f"Loaded model from: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found in: {self.model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
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
        vertex_set = list(entity_groups.values())
        
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
                               collate_fn=collate_fn, drop_last=False)
        
        all_preds = []
        all_scores = []
        all_topks = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = self._load_input(batch, tag="test")
                
                outputs = self.model(**inputs)
                pred = outputs["rel_pred"].cpu().numpy()
                pred[np.isnan(pred)] = 0
                all_preds.append(pred)
                
                if "scores" in outputs:
                    all_scores.append(outputs["scores"].cpu().numpy())
                    all_topks.append(outputs["topks"].cpu().numpy())
        
        # Concatenate predictions
        preds = np.concatenate(all_preds, axis=0)
        scores = np.concatenate(all_scores, axis=0) if all_scores else []
        topks = np.concatenate(all_topks, axis=0) if all_topks else []
        
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
                    if h_idx < len(sample['vertexSet']) and t_idx < len(sample['vertexSet']):
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
if __name__ == "__main__":
    # Example usage of the DreeamInference class
    
    # Initialize the inference model
    inference = DreeamInference(
        model_path="./logs/your_experiment",
        config_path="./dreeam-config.json",
        device="auto",
        batch_size=4
    )
    
    # Example text and entities
    text = "Иван Петров работает в компании Газпром. Он является директором отдела продаж."
    entities = [
        {"text": "Иван Петров", "start": 0, "end": 11, "type": "PERSON"},
        {"text": "Газпром", "start": 33, "end": 40, "type": "ORGANIZATION"}
    ]
    
    # Predict relations
    relations = inference.predict_single(text, entities)
    
    print("Predicted relations:")
    for head, tail, relation in relations:
        print(f"{head} -> {relation} -> {tail}")    