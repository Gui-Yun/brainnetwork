"""
BrainNetwork - Brain Network Analysis Toolkit

This package provides tools for analyzing neural activity data, including:
- Data loading and preprocessing
- Network construction and analysis
- Decoding and classification
"""

__version__ = "0.1.0"

# ========== Data Loading Module ==========
from .loaddata import (
    # Core loading functions
    load_data,
    preprocess_data,
    preprocess_spike_data,

    # Data processing functions
    process_trigger,
    segment_neuron_data,
    rr_selection,

    # Parameters
    t_stimulus,
    l_stimulus,
    l_trials,
    ipd,
    isi,
    trials_num,
    reliability_threshold,
    snr_threshold,
)

# ========== Network Analysis Module ==========
from .network import (
    # Core network construction functions
    construct_correlation_network,
    compute_correlation_matrix,
    build_correlation_graph,
    correlation_network_summary,

    # Network metrics
    compute_network_metrics,
    compute_betweenness_stats,
    compute_network_metrics_by_class,

    # Data processing
    reshape_segments,
    create_supernodes,
)

# ========== Decoding Analysis Module ==========
from .decoding import (
    # Fisher information
    Fisher_information,
    FI_by_timepoints,
    FI_by_timepoints_v2,
    FI_by_neuron_count,

    # Classification
    classify_by_timepoints,
)

from .visualization import *

# Define public API
__all__ = [
    # Version
    "__version__",

    # Data loading
    "load_data",
    "preprocess_data",
    "preprocess_spike_data",
    "process_trigger",
    "segment_neuron_data",
    "rr_selection",

    # Network analysis
    "construct_correlation_network",
    "compute_correlation_matrix",
    "build_correlation_graph",
    "correlation_network_summary",
    "compute_network_metrics",
    "compute_betweenness_stats",
    "compute_network_metrics_by_class",
    "reshape_segments",
    "create_supernodes",

    # Decoding analysis
    "Fisher_information",
    "FI_by_timepoints",
    "FI_by_timepoints_v2",
    "FI_by_neuron_count",
    "classify_by_timepoints",

    # Parameters
    "t_stimulus",
    "l_stimulus",
    "l_trials",
    "ipd",
    "isi",
    "trials_num",
    "reliability_threshold",
    "snr_threshold",
]
