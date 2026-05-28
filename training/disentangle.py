from __future__ import annotations
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class GradientReversal(torch.autograd.Function):
    """GRL layer for adversarial feature learning."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Reverse the gradient direction by multiplying by -lambda
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Helper method to apply gradient reversal."""
    return GradientReversal.apply(x, lambda_)


class AdversarialHead(nn.Module):
    """Classifier head attached via GRL to make representations invariant."""
    def __init__(self, embed_dim: int = config.EMBED_DIM, n_classes: int = 35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(96, n_classes),
        )

    def forward(self, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        x_rev = grad_reverse(x, lambda_)
        return self.net(x_rev)


class CLUB(nn.Module):
    """Contrastive Log-ratio Upper Bound on Mutual Information (MI).
    
    Used to minimize mutual information between speaker and phonetic embeddings.
    """
    def __init__(self, dim: int = config.EMBED_DIM):
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

    def forward(self, z_spk: torch.Tensor, z_phn: torch.Tensor) -> torch.Tensor:
        mu = self.mu_net(z_spk)
        logvar = self.logvar_net(z_spk)
        
        # Clip logvar to prevent division by zero or large exponents
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        # positive sample log ratio
        pos = -((mu - z_phn) ** 2) / (2.0 * torch.exp(logvar) + 1e-6) - 0.5 * logvar
        
        # negative sample log ratio (shuffle z_phn over batch dimension)
        z_phn_shuffle = z_phn[torch.randperm(z_phn.size(0), device=z_phn.device)]
        neg = -((mu - z_phn_shuffle) ** 2) / (2.0 * torch.exp(logvar) + 1e-6) - 0.5 * logvar
        
        # Contrastive upper bound estimation
        mi_upper = pos.sum(dim=-1).mean() - neg.sum(dim=-1).mean()
        return mi_upper


class DisentanglementLoss(nn.Module):
    """Combined GRL Adversarial Loss + CLUB MI Minimization Loss."""
    def __init__(
        self,
        embed_dim: int = config.EMBED_DIM,                          # 192
        n_spk: int = config.NUM_SPEAKERS_VOXCELEB,                  # 7205
        n_phn: int = config.NUM_KEYWORDS_GSC,                       # 35
        club_weight: float = config.CLUB_WEIGHT,                    # 0.1
    ):
        super().__init__()
        # Predict speakers from phonetic representations (to reverse speaker features out)
        self.adv_spk = AdversarialHead(embed_dim, n_spk)
        # Predict keywords/phonemes from speaker representations (to reverse phonetic features out)
        self.adv_phn = AdversarialHead(embed_dim, n_phn)
        
        self.club = CLUB(embed_dim)
        self.ce = nn.CrossEntropyLoss()
        self.club_weight = club_weight

    def forward(
        self,
        z_phn: torch.Tensor,
        z_spk: torch.Tensor,
        spk_labels: torch.Tensor,
        phn_labels: torch.Tensor,
        lambda_: float,
    ) -> torch.Tensor:
        # 1. Adversarial: make z_phn speaker-invariant
        spk_pred = self.adv_spk(z_phn, lambda_)
        loss_adv_spk = self.ce(spk_pred, spk_labels)

        # 2. Adversarial: make z_spk phonetic-invariant
        phn_pred = self.adv_phn(z_spk, lambda_)
        loss_adv_phn = self.ce(phn_pred, phn_labels)

        # 3. MI Minimization: CLUB bound (detaching one stream helps optimization stability)
        loss_mi = self.club(z_spk.detach(), z_phn)

        # Total combined loss
        return loss_adv_spk + loss_adv_phn + self.club_weight * loss_mi
