# -*- coding: utf-8 -*-
"""Minimal MORSE-Net model package.

This package intentionally exposes only the proposed MORSE-Net architecture
and its required internal modules. Baseline/comparison model definitions are
not included.
"""

from .morse_net import (
    MORSENet,
    MyNet7ViewTimeFreq,      # backward-compatible alias
    RhythmFrequencyView,
    MorphologyGradientView,
    create_morse_net,
    create_mynet7view_timefreq,
)
from .temporal_backbone import (
    AdaptiveViewWeight,
    AdaptiveWeight,          # backward-compatible alias
    LeadRegionViewNet,
    MyNet,                   # backward-compatible alias
    MyNet6View,              # backward-compatible alias
    MyNet12Lead,             # backward-compatible alias
    TemporalRes2NetBackbone,
    TwelveLeadBaselineNet,
)
from .rhythm_frequency_branch import (
    LearnableSTFT,
    SpectrogramCNN,
    TimeFreqFusionNet,
    create_timefreq_model,
)
from .diagnostic_label_graph import LabelGraphRefiner, FeatureLabelGCN
from .attention_layers import CoordAtt, CrossModalAttention, LeadAttention

__all__ = [
    "MORSENet",
    "MyNet7ViewTimeFreq",
    "RhythmFrequencyView",
    "MorphologyGradientView",
    "create_morse_net",
    "create_mynet7view_timefreq",
    "AdaptiveViewWeight",
    "AdaptiveWeight",
    "LeadRegionViewNet",
    "MyNet",
    "MyNet6View",
    "MyNet12Lead",
    "TemporalRes2NetBackbone",
    "TwelveLeadBaselineNet",
    "LearnableSTFT",
    "SpectrogramCNN",
    "TimeFreqFusionNet",
    "create_timefreq_model",
    "LabelGraphRefiner",
    "FeatureLabelGCN",
    "CoordAtt",
    "CrossModalAttention",
    "LeadAttention",
]
