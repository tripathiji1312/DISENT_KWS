"""
models/__init__.py
==================
Public API for the models package.
Import the unified model and individual components from here.
"""

from models.bc_resnet  import BCResNet2, BCResBlock
from models.temporal   import get_temporal_block, DilatedConvTemporalBlock, USE_MAMBA
from models.film       import FiLM
from models.heads      import PhoneticHead, SpeakerHead, AttentiveStatsPool, CausalConformerBlock, SEDWRes2NetBlock
from models.scorer     import DualGateScorer
from models.disent_v2  import DISENT_KWS_v2

__all__ = [
    "DISENT_KWS_v2",
    "BCResNet2",
    "BCResBlock",
    "get_temporal_block",
    "DilatedConvTemporalBlock",
    "USE_MAMBA",
    "FiLM",
    "PhoneticHead",
    "SpeakerHead",
    "AttentiveStatsPool",
    "CausalConformerBlock",
    "SEDWRes2NetBlock",
    "DualGateScorer",
]
