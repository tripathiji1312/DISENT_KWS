from __future__ import annotations
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax (ArcFace) for speaker identification.
    
    References:
      ArcFace: Additive Angular Margin Loss for Deep Face Recognition (Deng et al.)
    """
    def __init__(
        self,
        in_dim: int = config.EMBED_DIM,                         # 192
        n_classes: int = config.NUM_SPEAKERS_VOXCELEB,          # 7205
        scale: float = config.AAM_SCALE,                        # 30
        margin: float = config.AAM_MARGIN,                      # 0.2
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s = scale
        self.m = margin
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize weights and input features
        w = F.normalize(self.weight, dim=1)
        x = F.normalize(x, dim=1)
        
        # Calculate cosine similarity
        cosine = F.linear(x, w)
        
        # Compute theta (angular margin)
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        # Compute additive margin
        one_hot = F.one_hot(labels, self.weight.size(0)).float()
        logits = self.s * torch.cos(theta + self.m * one_hot)
        
        return self.ce(logits, labels)


class PrototypicalLoss(nn.Module):
    """Prototypical metric loss for few-shot keyword enrollment verification."""
    def __init__(
        self,
        scale: float = config.PROTO_SCALE,                       # 32
        margin: float = config.PROTO_MARGIN,                     # 0.25
    ):
        super().__init__()
        self.s = scale
        self.m = margin

    def forward(
        self,
        anchor: torch.Tensor,                                   # (B, D)
        positive: torch.Tensor,                                 # (B, D)
        negatives: torch.Tensor,                                # (B, N, D)
    ) -> torch.Tensor:
        # Normalize to compute cosine similarity
        anchor_n = F.normalize(anchor, dim=-1)
        positive_n = F.normalize(positive, dim=-1)
        negatives_n = F.normalize(negatives, dim=-1)

        sim_pos = F.cosine_similarity(anchor_n, positive_n, dim=-1) # (B,)
        
        # Cosine similarity with negative candidates
        # anchor_n: (B, 1, D) and negatives_n: (B, N, D) -> output (B, N)
        sim_neg = F.cosine_similarity(anchor_n.unsqueeze(1), negatives_n, dim=-1)

        # Apply angular scaling and margin
        logits = torch.cat([
            (self.s * (sim_pos - self.m)).unsqueeze(1),
            self.s * sim_neg
        ], dim=1) # (B, 1 + N)

        # Positive class index is 0
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


def rejection_loss(
    anchor: torch.Tensor,                                       # (B, D)
    positive: torch.Tensor,                                     # (B, D)
    confuser: torch.Tensor,                                     # (B, D)
    margin: float = config.REJECTION_MARGIN,                    # 0.4
) -> torch.Tensor:
    """Rejection loss to maximize margin between target keyword and phonetically similar confusers."""
    anchor_n = F.normalize(anchor, dim=-1)
    positive_n = F.normalize(positive, dim=-1)
    confuser_n = F.normalize(confuser, dim=-1)

    sim_pos = F.cosine_similarity(anchor_n, positive_n, dim=-1)
    sim_neg = F.cosine_similarity(anchor_n, confuser_n, dim=-1)

    return F.relu(sim_neg - sim_pos + margin).mean()


class KDLoss(nn.Module):
    """Soft Kullback-Leibler Divergence Loss for Knowledge Distillation."""
    def __init__(self, temperature: float = config.KD_TEMPERATURE):     # 4
        super().__init__()
        self.T = temperature

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        # Apply temperature scaling
        s = F.log_softmax(student_logits / self.T, dim=1)
        t = F.softmax(teacher_logits / self.T, dim=1)
        
        # KL-Div scaled by T^2 as per standard distillation formulation
        return (self.T * self.T) * F.kl_div(s, t, reduction="batchmean")
