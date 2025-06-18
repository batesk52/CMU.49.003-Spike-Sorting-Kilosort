# Example Usage of the New Flexible Cluster Filtering System
# ============================================================

import numpy as np
from automations import RM1
from pathlib import Path

# Example 1: Simple label filtering
# ==================================

# Initialize rat and wrappers (assuming you have these set up)
# rat = RM1.Rat(DATA_DIRECTORY, PROBE_DIRECTORY, "RAT_ID")
# ks_wrapper = RM1.Kilosort_wrapper(SAVE_DIRECTORY, PROBE_DIRECTORY)

# Get clusters by specific labels
trial_name = "VF_DRG_1_trial"

# Single label (replaces get_good_clusters_from_group)
good_clusters = ks_wrapper.get_clusters_by_labels(trial_name, 'good')

# Multiple labels (replaces the old cluster_types='both' approach)
good_and_mua = ks_wrapper.get_clusters_by_labels(trial_name, ['good', 'mua'])

# Any custom labels
custom_labels = ks_wrapper.get_clusters_by_labels(trial_name, ['custom_label1', 'custom_label2'])

print(f"Good clusters: {good_clusters}")
print(f"Good + MUA clusters: {good_and_mua}")
print(f"Custom labels: {custom_labels}")


# Example 2: Analysis with different cluster configurations
# =========================================================

# Method 1: Simple string (same as before, but more flexible)
results1 = rat.get_von_frey_analysis(
    si_wrapper, ks_wrapper, 
    cluster_labels='good'  # Only good clusters
)

# Method 2: Multiple labels
results2 = rat.get_von_frey_analysis(
    si_wrapper, ks_wrapper, 
    cluster_labels=['good', 'mua']  # Both good and MUA clusters
)

# Method 3: Custom labels
results3 = rat.get_von_frey_analysis(
    si_wrapper, ks_wrapper, 
    cluster_labels=['high_quality', 'responsive']  # Custom criteria
)

# Method 4: Per-trial customization
trial_specific_labels = {
    'VF_DRG_1_trial': 'good',
    'VF_DRG_2_trial': ['good', 'mua'],
    'VF_DRG_3_trial': ['good', 'responsive']
}

results4 = rat.get_von_frey_analysis(
    si_wrapper, ks_wrapper, 
    cluster_labels=trial_specific_labels
)


# Example 3: Multi-rat analysis with flexible labeling
# ====================================================

# Create rat group and wrappers
rats = [
    RM1.Rat(DATA_DIRECTORY, PROBE_DIRECTORY, "DW322"),
    RM1.Rat(DATA_DIRECTORY, PROBE_DIRECTORY, "DW323"),
    RM1.Rat(DATA_DIRECTORY, PROBE_DIRECTORY, "DW327")
]
group = RM1.RatGroup(rats)
si_wrappers = group.create_spikeinterface_wrappers(SAVE_DIRECTORY)
ks_wrappers = group.create_kilosort_wrappers(SAVE_DIRECTORY, PROBE_DIRECTORY)

# Method 1: Same labels for all rats
from automations.analysis_functions import MultiRatVonFreyAnalysis

analyzer = MultiRatVonFreyAnalysis(group, si_wrappers, ks_wrappers)
results = analyzer.analyze_all_trials(
    excel_parent_folder=SAVE_DIRECTORY,
    cluster_labels=['good', 'mua'],  # Same for all rats
    fast_mode=True,
    skip_correlations=False
)

# Method 2: Different labels per rat
rat_specific_labels = {
    'DW322': 'good',  # Only good clusters for DW322
    'DW323': ['good', 'mua'],  # Good and MUA for DW323
    'DW327': ['good', 'responsive']  # Good and responsive for DW327
}

results = analyzer.analyze_all_trials(
    excel_parent_folder=SAVE_DIRECTORY,
    cluster_labels=rat_specific_labels,
    fast_mode=True
)

# Method 3: Per-trial customization (most flexible)
nested_labels = {
    'DW322': {
        'VF_DRG_1_trial': 'good',
        'VF_DRG_2_trial': ['good', 'mua']
    },
    'DW323': ['good', 'mua'],  # Same for all trials in this rat
    'DW327': 'good'
}

results = analyzer.analyze_all_trials(
    excel_parent_folder=SAVE_DIRECTORY,
    cluster_labels=nested_labels
)


# Example 4: Performance optimizations
# ====================================

# For large datasets, use performance optimizations
results_optimized = analyzer.analyze_all_trials(
    excel_parent_folder=SAVE_DIRECTORY,
    cluster_labels='good',
    fast_mode=True,  # Enables faster processing for large cluster counts
    skip_correlations=True,  # Skip correlation computation for speed
    correlation_window_size=5000,  # Smaller window for faster correlation
    subwindow_width=0.5  # Analysis parameter
)


# Example 5: Backward compatibility
# =================================

# The old way still works (for backward compatibility)
good_clusters_by_rat_and_trial = {}
for rat_id, ks_wrapper in ks_wrappers.items():
    good_clusters_by_trial = {}
    for trial_name in ks_wrapper.kilosort_results.keys():
        # Use new flexible method
        good_clusters = ks_wrapper.get_clusters_by_labels(trial_name, 'good')
        good_clusters_by_trial[trial_name] = good_clusters
    good_clusters_by_rat_and_trial[rat_id] = good_clusters_by_trial

# This still works with the new system
results_legacy = analyzer.analyze_all_trials(
    excel_parent_folder=SAVE_DIRECTORY,
    good_clusters_by_trial=good_clusters_by_rat_and_trial  # Legacy parameter
)

print("Analysis complete! The new system provides:")
print("1. Flexible cluster label filtering")
print("2. Performance optimizations for large datasets")
print("3. Backward compatibility with existing code")
print("4. Per-trial and per-rat customization options") 