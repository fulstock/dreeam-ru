import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from collections import Counter


class EnhancedAMTLoss(nn.Module):
    """
    Enhanced Adaptive Multi-Threshold Loss (AMTL) with:
    1. Multi-threshold segmentation (Xu et al., 2025)
    2. Effective number weighting (Ayaou, 2025)

    References:
    - Xu et al. (2025): "An Adaptive Multi-Threshold Loss and a General Framework
      for Collaborating Losses in Document-Level Relation Extraction", ACL 2025
    - Ayaou (2025): "Tackling Class Imbalance in Relation Extraction for french text"
    """

    def __init__(
        self,
        num_classes=48,
        num_segments=4,
        lambda_weight=3.5,
        use_effective_number=True,
        beta=0.9999,
        relation_freq_path=None,
    ):
        """
        Args:
            num_classes: Total number of relation classes (e.g., 48 for NEREL)
            num_segments: Number of sub-label space segments (default: 4)
            lambda_weight: Coefficient for weighted threshold averaging (default: 3.5)
            use_effective_number: Whether to use effective number weighting (default: True)
            beta: Beta parameter for effective number calculation (default: 0.9999)
            relation_freq_path: Path to JSON file with relation frequencies
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_segments = num_segments
        self.lambda_weight = lambda_weight
        self.use_effective_number = use_effective_number
        self.beta = beta

        # Initialize segments and weights
        self.segments = None
        self.class_weights = None
        self.relation_freq_path = relation_freq_path

        # Will be set during training
        self.samples_per_class = None

    def set_relation_frequencies(self, relation_frequencies):
        """
        Set relation frequencies and create segments

        Args:
            relation_frequencies: dict {relation_id: count} or list of counts
        """
        if isinstance(relation_frequencies, dict):
            # Convert dict to list
            freq_list = [relation_frequencies.get(i, 0) for i in range(self.num_classes)]
        else:
            freq_list = relation_frequencies

        self.samples_per_class = torch.FloatTensor(freq_list)

        # Create segments based on frequency
        self.segments = self._create_segments(freq_list)

        # Compute effective number weights
        if self.use_effective_number:
            self.class_weights = self._compute_effective_number_weights(freq_list)
        else:
            self.class_weights = torch.ones(self.num_classes)

    def _create_segments(self, relation_frequencies):
        """
        Partition label space into segments based on relation frequency

        Args:
            relation_frequencies: list of counts for each relation class

        Returns:
            List of segments, where each segment is a list of relation indices
        """
        # Sort relations by frequency (descending)
        sorted_indices = np.argsort(relation_frequencies)[::-1].tolist()

        # Partition into equal-sized segments
        segment_size = len(sorted_indices) // self.num_segments
        segments = []

        for i in range(self.num_segments):
            start_idx = i * segment_size
            if i == self.num_segments - 1:
                # Last segment takes remaining relations
                segment = sorted_indices[start_idx:]
            else:
                end_idx = (i + 1) * segment_size
                segment = sorted_indices[start_idx:end_idx]
            segments.append(segment)

        return segments

    def _compute_effective_number_weights(self, samples_per_class):
        """
        Compute effective number of samples for each class

        Formula: w_i = (1 - β) / (1 - β^n_i)
        where n_i is the number of samples for class i

        Args:
            samples_per_class: list of sample counts for each class

        Returns:
            Tensor of shape (num_classes,) with class weights
        """
        samples_per_class = np.array(samples_per_class, dtype=np.float64)

        # Avoid division by zero
        samples_per_class = np.maximum(samples_per_class, 1.0)

        # Compute effective number
        effective_num = 1.0 - np.power(self.beta, samples_per_class)
        weights = (1.0 - self.beta) / (effective_num + 1e-8)

        # Normalize weights
        weights = weights / weights.sum() * len(weights)

        return torch.FloatTensor(weights)

    def _compute_threshold_logit(self, logits, segment_idx, all_th_logits):
        """
        Compute weighted average threshold for a segment (Equation 2)

        logit^i_TH = (logit^i_TH + Σ(j≠i) logit^j_TH) / λ

        Args:
            logits: Tensor of shape (batch_size, num_classes)
            segment_idx: Index of current segment
            all_th_logits: List of threshold logits for all segments

        Returns:
            Tensor of shape (batch_size, 1) with threshold logits for this segment
        """
        batch_size = logits.size(0)

        # Initialize threshold logit (learnable parameter for this segment)
        # We use the mean of logits for relations in this segment as initial threshold
        segment_relations = self.segments[segment_idx]
        th_logit = logits[:, segment_relations].mean(dim=1, keepdim=True)

        # Weighted average with other thresholds
        if len(all_th_logits) > 1:
            other_th_sum = sum([th for i, th in enumerate(all_th_logits) if i != segment_idx])
            th_logit = (th_logit + other_th_sum) / self.lambda_weight

        return th_logit

    def _compute_segment_loss(self, logits, labels, segment_idx, th_logit):
        """
        Compute loss for a single segment (Equation 3)

        L^i_3 = -Σ(r ∈ P^i_T) log(exp(logit_r) / (Σ(r' ∈ P^i_T) exp(logit_r') + exp(logit^i_TH)))
        L^i_4 = -log(exp(logit^i_TH) / (Σ(r' ∈ N^i_T) exp(logit_r') + exp(logit^i_TH)))

        Args:
            logits: Tensor of shape (batch_size, num_classes)
            labels: Tensor of shape (batch_size, num_classes)
            segment_idx: Index of current segment
            th_logit: Threshold logit for this segment, shape (batch_size, 1)

        Returns:
            Scalar loss for this segment
        """
        batch_size = logits.size(0)
        segment_relations = self.segments[segment_idx]

        # Extract logits for this segment
        segment_logits = logits[:, segment_relations]  # (batch_size, segment_size)
        segment_labels = labels[:, segment_relations]  # (batch_size, segment_size)

        # Create masks for positive and negative classes in this segment
        pos_mask = segment_labels  # 1 for positive, 0 otherwise
        neg_mask = 1 - segment_labels  # 1 for negative, 0 otherwise

        # L^i_3: Positive class loss
        # Combine segment logits with threshold
        logits_with_th = torch.cat([segment_logits, th_logit], dim=1)  # (batch_size, segment_size + 1)

        # Mask for positive classes + threshold
        pos_th_mask = torch.cat([pos_mask, torch.ones(batch_size, 1).to(pos_mask)], dim=1)

        # Rank positive classes to threshold
        logit1 = logits_with_th - (1 - pos_th_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1)[:, :-1] * pos_mask).sum(1)

        # L^i_4: Negative class loss
        # Mask for negative classes + threshold
        neg_th_mask = torch.cat([neg_mask, torch.ones(batch_size, 1).to(neg_mask)], dim=1)

        # Rank threshold to negative classes
        logit2 = logits_with_th - (1 - neg_th_mask) * 1e30
        th_label = torch.zeros(batch_size, segment_logits.size(1) + 1).to(labels)
        th_label[:, -1] = 1.0  # Threshold is the last position
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Combine losses
        segment_loss = (loss1 + loss2).mean()

        return segment_loss

    def forward(self, logits, labels):
        """
        Forward pass (Equation 4)

        L_AMTL = (1/n) Σ^n_{i=1} (L^i_3 + L^i_4)

        Args:
            logits: Tensor of shape (batch_size, num_classes)
            labels: Tensor of shape (batch_size, num_classes)

        Returns:
            Scalar loss
        """
        if self.segments is None:
            raise ValueError(
                "Segments not initialized. Call set_relation_frequencies() first."
            )

        # Move class weights to same device as logits
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)

        # Compute threshold logits for all segments first
        # (needed for weighted averaging)
        all_th_logits = []
        for i in range(self.num_segments):
            segment_relations = self.segments[i]
            th_logit = logits[:, segment_relations].mean(dim=1, keepdim=True)
            all_th_logits.append(th_logit)

        # Apply weighted averaging to thresholds
        for i in range(self.num_segments):
            if self.num_segments > 1:
                other_th_sum = sum([th for j, th in enumerate(all_th_logits) if j != i])
                all_th_logits[i] = (all_th_logits[i] + other_th_sum) / self.lambda_weight

        # Compute loss for each segment
        total_loss = 0
        for i in range(self.num_segments):
            segment_loss = self._compute_segment_loss(
                logits, labels, i, all_th_logits[i]
            )

            # Apply effective number weighting if enabled
            if self.use_effective_number and self.class_weights is not None:
                segment_relations = self.segments[i]
                segment_weights = self.class_weights[segment_relations].mean()
                segment_loss = segment_loss * segment_weights

            total_loss += segment_loss

        # Average across segments
        amtl_loss = total_loss / self.num_segments

        return amtl_loss

    def get_label(self, logits, num_labels=-1):
        """
        Get predicted labels using adaptive thresholds

        For multi-threshold, we use the threshold from the segment each relation belongs to

        Args:
            logits: Tensor of shape (batch_size, num_classes)
            num_labels: Maximum number of labels to predict per example

        Returns:
            Tensor of shape (batch_size, num_classes) with binary predictions
        """
        if self.segments is None:
            # Fallback to standard threshold (use class 0 as threshold)
            th_logit = logits[:, 0].unsqueeze(1)
            output = torch.zeros_like(logits).to(logits)
            mask = (logits > th_logit)
            if num_labels > 0:
                top_v, _ = torch.topk(logits, num_labels, dim=1)
                top_v = top_v[:, -1]
                mask = (logits >= top_v.unsqueeze(1)) & mask
            output[mask] = 1.0
            output[:, 0] = (output.sum(1) == 0.).to(logits)
            return output

        # Compute threshold for each segment
        all_th_logits = []
        for i in range(self.num_segments):
            segment_relations = self.segments[i]
            th_logit = logits[:, segment_relations].mean(dim=1, keepdim=True)
            all_th_logits.append(th_logit)

        # Apply weighted averaging
        for i in range(self.num_segments):
            if self.num_segments > 1:
                other_th_sum = sum([th for j, th in enumerate(all_th_logits) if j != i])
                all_th_logits[i] = (all_th_logits[i] + other_th_sum) / self.lambda_weight

        # Create output tensor
        output = torch.zeros_like(logits).to(logits)

        # For each segment, compare logits with segment threshold
        for i in range(self.num_segments):
            segment_relations = self.segments[i]
            th_logit = all_th_logits[i]

            for rel_idx in segment_relations:
                mask = logits[:, rel_idx] > th_logit.squeeze()
                output[:, rel_idx][mask] = 1.0

        # Apply top-k constraint if specified
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = logits >= top_v.unsqueeze(1)
            output = output * mask.float()

        # Set "no relation" if no other relation is predicted
        output[:, 0] = (output.sum(1) == 0.).to(logits)

        return output

    def get_score(self, logits, num_labels=-1):
        """
        Get top-k relation scores

        Args:
            logits: Tensor of shape (batch_size, num_classes)
            num_labels: Number of top labels to return

        Returns:
            If num_labels > 0: tuple of (values, indices) from torch.topk
            Otherwise: (score, 0) where score is difference from threshold
        """
        if num_labels > 0:
            return torch.topk(logits, num_labels, dim=1)
        else:
            return logits[:, 1] - logits[:, 0], 0


class ATLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)
        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):

        th_logit = logits[:, 0].unsqueeze(1)  # theshold is norelation
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1] # smallest logits among the num_labels
            # predictions are those logits > thresh and logits >= smallest
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        # if no such relation label exist: set its label to 'Nolabel'
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

    def get_score(self, logits, num_labels=-1):

        if num_labels > 0:
            return torch.topk(logits, num_labels, dim=1)
        else:
            return logits[:,1] - logits[:,0], 0
