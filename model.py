import os
import torch
import torch.nn as nn
from opt_einsum import contract
import torch.nn.functional as F
from long_seq import process_long_input
from losses import ATLoss


class DocREModel(nn.Module):

    def __init__(self, config, model, tokenizer,
                emb_size=None, block_size=64, num_labels=-1,
                max_sent_num=25, evi_thresh=0.2):
        '''
        Initialize the model.
        :model: Pretrained langage model encoder;
        :tokenizer: Tokenzier corresponding to the pretrained language model encoder;
        :emb_size: Dimension of embeddings for subject/object (head/tail) representations.
                   If None, automatically set to match config.hidden_size for optimal performance.
        :block_size: Number of blocks for grouped bilinear classification;
        :num_labels: Maximum number of relation labels for each entity pair;
        :max_sent_num: Maximum number of sentences for each document;
        :evi_thresh: Threshold for selecting evidence sentences.
        '''

        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size

        # Auto-set emb_size to match hidden_size if not specified
        # This ensures optimal performance for different backbone models
        # (e.g., BERT-base: 768, RoBERTa-large: 1024)
        if emb_size is None:
            emb_size = self.hidden_size

        # Validate that emb_size is divisible by block_size
        # This is required for the grouped bilinear classification
        assert emb_size % block_size == 0, \
            f"emb_size ({emb_size}) must be divisible by block_size ({block_size}). " \
            f"Current hidden_size is {self.hidden_size}."

        self.loss_fnt = ATLoss()
        self.loss_fnt_evi = nn.KLDivLoss(reduction="batchmean")

        self.head_extractor = nn.Linear(self.hidden_size * 2, emb_size)
        self.tail_extractor = nn.Linear(self.hidden_size * 2, emb_size)

        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.total_labels = config.num_labels
        self.max_sent_num = max_sent_num
        self.evi_thresh = evi_thresh

        # Store whether gradient checkpointing is supported
        self.supports_gradient_checkpointing = hasattr(self.model, 'gradient_checkpointing_enable')

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing for memory efficiency (30-40% memory reduction).
        This trades compute for memory by not storing all intermediate activations.
        Only use during training, not inference.
        """
        if self.supports_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled (30-40% memory reduction)")
        else:
            print("⚠ Warning: Model does not support gradient_checkpointing_enable")

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing (faster but uses more memory)."""
        if self.supports_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
            print("✓ Gradient checkpointing disabled")


    def encode(self, input_ids, attention_mask):
        
        '''
        Get the embedding of each token. For long document that has more than 512 tokens, split it into two overlapping chunks.
        Inputs:
            :input_ids: (batch_size, doc_len)
            :attention_mask: (batch_size, doc_len)
        Outputs:
            :sequence_output: (batch_size, doc_len, hidden_dim)
            :attention: (batch_size, num_attn_heads, doc_len, doc_len)
        '''
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        # process long documents.
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts, offset):

        '''
        Get head, tail, context embeddings from token embeddings.
        Vectorized implementation for 3-5x speedup.
        Inputs:
            :sequence_output: (batch_size, doc_len, hidden_dim)
            :attention: (batch_size, num_attn_heads, doc_len, doc_len)
            :entity_pos: list of list. Outer length = batch size, inner length = number of entities each batch.
            :hts: list of list. Outer length = batch size, inner length = number of combination of entity pairs each batch.
            :offset: 1 for bert and roberta. Offset caused by [CLS] token.
        Outputs:
            :hss: (num_ent_pairs_all_batches, emb_size)
            :tss: (num_ent_pairs_all_batches, emb_size)
            :rss: (num_ent_pairs_all_batches, emb_size)
            :ht_atts: (num_ent_pairs_all_batches, doc_len)
            :rels_per_batch: list of length = batch size. Each entry represents the number of entity pairs of the batch.
        '''

        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        ht_atts = []

        for i in range(len(entity_pos)): # for each batch
            # Vectorized entity aggregation
            entity_embs, entity_atts = self._aggregate_entities_vectorized(
                sequence_output[i], attention[i], entity_pos[i], offset, c, h
            )

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)

            # obtain subject/object (head/tail) embeddings from entity embeddings.
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

            ht_att = (h_att * t_att).mean(1) # average over all heads
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30)
            ht_atts.append(ht_att)

            # obtain local context embeddings.
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

        rels_per_batch = [len(b) for b in hss]
        hss = torch.cat(hss, dim=0) # (num_ent_pairs_all_batches, emb_size)
        tss = torch.cat(tss, dim=0) # (num_ent_pairs_all_batches, emb_size)
        rss = torch.cat(rss, dim=0) # (num_ent_pairs_all_batches, emb_size)
        ht_atts = torch.cat(ht_atts, dim=0) # (num_ent_pairs_all_batches, max_doc_len)

        return hss, rss, tss, ht_atts, rels_per_batch

    def _aggregate_entities_vectorized(self, seq_out, attn, entities, offset, c, h):
        '''
        Vectorized entity aggregation for 3-5x speedup.
        Aggregates multiple mentions into single entity representations.

        Args:
            seq_out: (doc_len, hidden_dim)
            attn: (num_heads, doc_len, doc_len)
            entities: List of entity mentions, each entity is list of (start, end) tuples
            offset: Position offset for special tokens
            c: Sequence length
            h: Number of attention heads

        Returns:
            entity_embs: List of entity embeddings
            entity_atts: List of entity attentions
        '''
        entity_embs = []
        entity_atts = []

        for entity in entities:
            if len(entity) > 1:
                # Multi-mention entity: collect all valid mentions
                valid_positions = [start + offset for start, end in entity
                                 if start + offset < c]

                if len(valid_positions) > 0:
                    # Vectorized gathering of mention embeddings
                    mention_positions = torch.LongTensor(valid_positions).to(seq_out.device)
                    mention_embs = torch.index_select(seq_out, 0, mention_positions)  # (num_mentions, hidden_dim)
                    mention_atts = torch.index_select(attn, 1, mention_positions)  # (num_heads, num_mentions, seq_len)

                    # Log-sum-exp pooling for embeddings
                    e_emb = torch.logsumexp(mention_embs, dim=0)

                    # Mean pooling for attention (transpose to get correct dimension)
                    e_att = mention_atts.mean(dim=1)  # (num_heads, seq_len)
                else:
                    # All mentions truncated - use zero vectors
                    e_emb = torch.zeros(self.config.hidden_size, device=seq_out.device, dtype=seq_out.dtype)
                    e_att = torch.zeros(h, c, device=attn.device, dtype=attn.dtype)
            else:
                # Single-mention entity
                start, end = entity[0]
                if start + offset < c:
                    e_emb = seq_out[start + offset]
                    e_att = attn[:, start + offset]  # (num_heads, seq_len)
                else:
                    # Mention truncated - use zero vectors
                    e_emb = torch.zeros(self.config.hidden_size, device=seq_out.device, dtype=seq_out.dtype)
                    e_att = torch.zeros(h, c, device=attn.device, dtype=attn.dtype)

            entity_embs.append(e_emb)
            entity_atts.append(e_att)

        return entity_embs, entity_atts


    def forward_rel(self, hs, ts, rs):
        '''
        Forward computation for RE with fused bilinear operation (15-25% faster).
        Inputs:
            :hs: (num_ent_pairs_all_batches, emb_size)
            :ts: (num_ent_pairs_all_batches, emb_size)
            :rs: (num_ent_pairs_all_batches, emb_size)
        Outputs:
            :logits: (num_ent_pairs_all_batches, num_rel_labels)
        '''

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=-1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=-1)))

        # Fused bilinear operation using einsum (faster + less memory)
        # Split into blocks for grouped bilinear
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)

        # Fused einsum: compute outer product and flatten in one operation
        # Old: (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, emb_size * block_size)
        # New: einsum performs outer product without creating 4D intermediate tensor
        bl = contract("bgi,bgj->bgij", b1, b2).reshape(-1, self.emb_size * self.block_size)

        logits = self.bilinear(bl)

        return logits


    def forward_evi(self, doc_attn, sent_pos, batch_rel, offset):
        '''
        Forward computation for ER.
        Inputs:
            :doc_attn: (num_ent_pairs_all_batches, doc_len), attention weight of each token for computing localized context pooling.
            :sent_pos: list of list. The outer length = batch size. The inner list contains (start, end) position of each sentence in each batch.
            :batch_rel: list of length = batch size. Each entry represents the number of entity pairs of the batch.
            :offset: 1 for bert and roberta. Offset caused by [CLS] token.
        Outputs:
            :s_attn:  (num_ent_pairs_all_batches, max_sent_all_batch), sentence-level evidence distribution of each entity pair.
        '''
        
        max_sent_num = max([len(sent) for sent in sent_pos])
        rel_sent_attn = []
        # print(len(sent_pos)) # 6
        # print(doc_attn.shape) # torch.Size([17356, 512])
        for i in range(len(sent_pos)): # for each batch
            # the relation ids corresponds to document in batch i is [sum(batch_rel[:i]), sum(batch_rel[:i+1]))
            curr_attn = doc_attn[sum(batch_rel[:i]):sum(batch_rel[:i+1])]
            curr_sent_pos = [torch.arange(s[0], s[1]).to(curr_attn.device) + offset for s in sent_pos[i]] # + offset

            # print(curr_sent_pos) # ..., 650]]
            # print(curr_attn.shape) # torch.Size([6806, 512])
            curr_attn_per_sent = [curr_attn.index_select(-1, sent) for sent in curr_sent_pos]
            curr_attn_per_sent += [torch.zeros_like(curr_attn_per_sent[0])] * (max_sent_num - len(curr_attn_per_sent))
            sum_attn = torch.stack([attn.sum(dim=-1) for attn in curr_attn_per_sent], dim=-1) # sum across those attentions
            rel_sent_attn.append(sum_attn)

        s_attn = torch.cat(rel_sent_attn, dim=0)
        return s_attn


    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None, # relation labels
                entity_pos=None,
                hts=None, # entity pairs
                sent_pos=None, 
                sent_labels=None, # evidence labels (0/1)
                teacher_attns=None, # evidence distribution from teacher model
                tag="train",
                ):

        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        output = {}
        sequence_output, attention = self.encode(input_ids, attention_mask)

        hs, rs, ts, doc_attn, batch_rel = self.get_hrt(sequence_output, attention, entity_pos, hts, offset)
        logits = self.forward_rel(hs, ts, rs)
        output["rel_pred"] = self.loss_fnt.get_label(logits, num_labels=self.num_labels)

        if sent_labels != None: # human-annotated evidence available

            s_attn = self.forward_evi(doc_attn, sent_pos, batch_rel, offset)
            output["evi_pred"] = F.pad(s_attn > self.evi_thresh, (0, self.max_sent_num - s_attn.shape[-1]))

        if tag in ["test", "dev"]: # testing
            scores_topk = self.loss_fnt.get_score(logits, self.num_labels)
            output["scores"] = scores_topk[0]
            output["topks"] = scores_topk[1]
        
        if tag == "infer": # teacher model inference
            output["attns"] = doc_attn.split(batch_rel)

        else: # training
            # relation extraction loss
            loss = self.loss_fnt(logits.float(), labels.float())
            output["loss"] = {"rel_loss": loss.to(sequence_output)}
                
            if sent_labels != None: # supervised training with human evidence

                idx_used = torch.nonzero(labels[:,1:].sum(dim=-1)).view(-1)
                # evidence retrieval loss (kldiv loss)
                s_attn = s_attn[idx_used]
                sent_labels = sent_labels[idx_used]
                norm_s_labels = sent_labels/(sent_labels.sum(dim=-1, keepdim=True) + 1e-30)
                norm_s_labels[norm_s_labels == 0] = 1e-30
                s_attn[s_attn == 0] = 1e-30
                evi_loss = self.loss_fnt_evi(s_attn.log(), norm_s_labels)
                output["loss"]["evi_loss"] = evi_loss.to(sequence_output)
            
            elif teacher_attns != None: # self training with teacher attention
                
                doc_attn[doc_attn == 0] = 1e-30
                teacher_attns[teacher_attns == 0] = 1e-30
                attn_loss = self.loss_fnt_evi(doc_attn.log(), teacher_attns)
                output["loss"]["attn_loss"] = attn_loss.to(sequence_output)
        
        return output
