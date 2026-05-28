"""
Unit tests for training losses, disentanglement layers, and learning rate schedules.
Run: pytest tests/test_training.py -v
"""

import sys
import os
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from training.losses import AAMSoftmax, PrototypicalLoss, rejection_loss, KDLoss
from training.disentangle import CLUB, DisentanglementLoss, grad_reverse, AdversarialHead
from training.scheduler import grl_lambda_schedule


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


class TestLosses:
    """Test standard loss functions in training/losses.py."""

    def test_aam_softmax_shapes_and_backward(self):
        """Verify AAMSoftmax forward and backward passes."""
        B, D = 8, config.EMBED_DIM
        loss_fn = AAMSoftmax(in_dim=D, n_classes=10, scale=30.0, margin=0.2)
        
        x = torch.randn(B, D, requires_grad=True)
        labels = torch.randint(0, 10, (B,))
        
        loss = loss_fn(x, labels)
        
        # Check forward pass
        assert loss.dim() == 0  # scalar
        assert not torch.isnan(loss)
        assert loss.item() > 0.0

        # Check backward pass
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_prototypical_loss_shapes_and_backward(self):
        """Verify PrototypicalLoss forward and backward passes."""
        B, D = 8, config.EMBED_DIM
        loss_fn = PrototypicalLoss(scale=32.0, margin=0.25)
        
        anchor = torch.randn(B, D, requires_grad=True)
        positive = torch.randn(B, D, requires_grad=True)
        negatives = torch.randn(B, 4, D, requires_grad=True)  # 4 negatives per anchor
        
        loss = loss_fn(anchor, positive, negatives)
        
        # Check forward pass
        assert loss.dim() == 0  # scalar
        assert not torch.isnan(loss)
        
        # Check backward pass
        loss.backward()
        assert anchor.grad is not None
        assert positive.grad is not None
        assert negatives.grad is not None

    def test_rejection_loss(self):
        """Verify rejection_loss margin computation."""
        B, D = 8, config.EMBED_DIM
        anchor = torch.randn(B, D)
        positive = torch.randn(B, D)
        confuser = torch.randn(B, D)
        
        loss = rejection_loss(anchor, positive, confuser, margin=0.4)
        assert loss.dim() == 0  # scalar
        assert not torch.isnan(loss)
        assert loss.item() >= 0.0

    def test_kd_loss(self):
        """Verify Kullback-Leibler KD loss."""
        B = 8
        loss_fn = KDLoss(temperature=4.0)
        
        student = torch.randn(B, 35, requires_grad=True)
        teacher = torch.randn(B, 35)  # target distribution
        
        loss = loss_fn(student, teacher)
        assert loss.dim() == 0  # scalar
        assert not torch.isnan(loss)
        
        loss.backward()
        assert student.grad is not None


class TestDisentanglement:
    """Test adversarial and mutual information layers in training/disentangle.py."""

    def test_grad_reverse_autograd(self):
        """Verify that GradientReversal correctly negates and scales gradients during backward pass."""
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        # Apply GRL with lambda = 0.5
        x_rev = grad_reverse(x, lambda_=0.5)
        
        # Forward pass: identity
        assert torch.allclose(x_rev, x)
        
        # Backward pass: d(out)/dx = -0.5
        loss = x_rev.sum()
        loss.backward()
        
        expected_grad = torch.tensor([-0.5, -0.5])
        assert torch.allclose(x.grad, expected_grad)

    def test_adversarial_head_flow(self):
        """Verify AdversarialHead classifier shape and gradient flow."""
        B, D = 8, config.EMBED_DIM
        head = AdversarialHead(embed_dim=D, n_classes=5)
        x = torch.randn(B, D, requires_grad=True)
        
        out = head(x, lambda_=0.8)
        assert out.shape == (B, 5)
        
        loss = out.sum()
        loss.backward()
        
        # Gradients should flow backward to x
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_club_mi_bound(self):
        """Verify CLUB mutual information bound calculation and shapes."""
        B, D = 8, config.EMBED_DIM
        club_estimator = CLUB(dim=D)
        
        z_spk = torch.randn(B, D, requires_grad=True)
        z_phn = torch.randn(B, D, requires_grad=True)
        
        mi_bound = club_estimator(z_spk, z_phn)
        assert mi_bound.dim() == 0  # scalar
        assert not torch.isnan(mi_bound)
        
        mi_bound.backward()
        assert z_spk.grad is not None
        assert z_phn.grad is not None

    def test_disentanglement_loss_backward(self):
        """Verify the combined DisentanglementLoss backward flow."""
        B, D = 8, config.EMBED_DIM
        loss_fn = DisentanglementLoss(embed_dim=D, n_spk=50, n_phn=10, club_weight=0.1)
        
        z_phn = torch.randn(B, D, requires_grad=True)
        z_spk = torch.randn(B, D, requires_grad=True)
        
        spk_labels = torch.randint(0, 50, (B,))
        phn_labels = torch.randint(0, 10, (B,))
        
        loss = loss_fn(z_phn, z_spk, spk_labels, phn_labels, lambda_=0.5)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        
        loss.backward()
        assert z_phn.grad is not None
        assert z_spk.grad is not None


class TestScheduler:
    """Test training scheduler logic."""

    def test_grl_lambda_schedule_bounds(self):
        """Verify schedule values are correct at boundaries."""
        # 1. At epoch 0, lambda must be exactly 0.0
        val_start = grl_lambda_schedule(0, 100)
        assert pytest.approx(val_start, abs=1e-5) == 0.0

        # 2. At maximum epoch, lambda should be close to 1.0 (specifically: 2 / (1 + e^-10) - 1 ≈ 0.9999)
        val_end = grl_lambda_schedule(100, 100)
        assert pytest.approx(val_end, abs=1e-4) == 0.9999

        # 3. Schedule should be strictly increasing
        v1 = grl_lambda_schedule(10, 100)
        v2 = grl_lambda_schedule(20, 100)
        v3 = grl_lambda_schedule(50, 100)
        assert v1 < v2 < v3
