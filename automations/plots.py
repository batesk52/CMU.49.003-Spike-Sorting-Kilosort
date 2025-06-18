def vf_all_trials_combined_plot(combined_results, combined_qst_notes, ks_wrappers, corr_threshold=0.1, cluster_labels=None, cluster_labels_xlsx=None):
    """
    Plot pre vs post DRGS reactivity for all trials, using clusters filtered by an external cluster label mapping or by Kilosort wrapper labels.
    
    Parameters
    ----------
    combined_results : dict
        Results dictionary from MultiRatVonFreyAnalysis
    combined_qst_notes : pd.DataFrame
        Combined QST notes from MultiRatVonFreyAnalysis
    ks_wrappers : dict
        Dictionary mapping rat_id to Kilosort_wrapper
    corr_threshold : float, optional
        Correlation threshold for filtering clusters, by default 0.1
    cluster_labels : dict, list, or None, optional
        If a dict, should map trial_name to allowed cluster IDs. If a list, used as labels for ks_wrapper.get_clusters_by_labels. If None, defaults to ['good', 'mua', 'noise'].
    cluster_labels_xlsx : str or Path, optional
        If provided, use this spreadsheet to map cluster_labels to cluster IDs for each trial.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    print(f"\nPlotting with correlation threshold: {corr_threshold}")
    print(f"Number of trials in combined_results: {len(combined_results)}")
    
    # If using spreadsheet, build mapping for all trials
    cluster_dict_from_xlsx = None
    if cluster_labels_xlsx is not None:
        cluster_dict_from_xlsx = build_cluster_label_dict_from_xlsx(cluster_labels_xlsx, cluster_labels, combined_results)

    # Create a dictionary mapping Trial_ID to Freq. (Hz)
    freq_dict = {}
    for _, row in combined_qst_notes.iterrows():
        if pd.notna(row['Trial_ID']):
            freq_dict[row['Trial_ID']] = row['Freq. (Hz)']
    
    print(f"[DEBUG] Created frequency dictionary with {len(freq_dict)} entries")
    print(f"[DEBUG] Available frequencies: {list(freq_dict.values())}")

    all_points = []
    total_clusters = 0
    filtered_clusters = 0
    
    for combined_key, res in combined_results.items():
        print(f"\nProcessing trial: {combined_key}")
        
        # Extract rat_id and trial_name from combined_key
        rat_id = combined_key.split('_')[0]
        trial_name = '_'.join(combined_key.split('_')[1:])
        
        # Get the Kilosort wrapper for this rat
        ks_wrapper = ks_wrappers.get(rat_id, None)
        if ks_wrapper is None:
            print(f"No Kilosort wrapper found for rat {rat_id}. Skipping...")
            continue
        
        # Determine clusters to plot for this trial
        if cluster_labels_xlsx is not None:
            clusters_to_plot = cluster_dict_from_xlsx.get(combined_key, [])
            print(f"[DEBUG] Clusters to plot for {combined_key} (from spreadsheet mapping): {clusters_to_plot} (types: {[type(x) for x in clusters_to_plot]})")
        elif isinstance(cluster_labels, dict):
            clusters_to_plot = cluster_labels.get(combined_key, [])
            print(f"[DEBUG] Clusters to plot for {combined_key} (from external mapping): {clusters_to_plot}")
        else:
            labels_to_use = cluster_labels if cluster_labels is not None else ['good', 'mua', 'noise']
            clusters_to_plot = ks_wrapper.get_clusters_by_labels(trial_name, labels_to_use)
            print(f"[DEBUG] Clusters to plot for {combined_key} (labels {labels_to_use}): {clusters_to_plot}")
        
        # Retrieve the stimulation frequency from the frequency dictionary using the trial key.
        if combined_key not in freq_dict:
            print(f"Frequency not found for trial {combined_key} in combined_qst_notes. Skipping...")
            continue
        freq_hz = freq_dict[combined_key]

        if "voltage" not in res:
            print(f"'voltage' not found in result for {combined_key}. Skipping...")
            continue
        if "firing_rates" not in res:
            print(f"'firing_rates' not found in result for {combined_key}. Skipping...")
            continue
        if "correlations" not in res:
            print(f"'correlations' not found in result for {combined_key}. Skipping...")
            continue

        avg_voltage_df = res["voltage"]
        firing_rates_df = res["firing_rates"]
        correlations = res["correlations"]

        print(f"[DEBUG] Correlation keys for {combined_key}: {list(correlations.keys())}")
        print(f"[DEBUG] Correlation values for {combined_key}: {list(correlations.values())}")
        print(f"[DEBUG] Using correlation threshold: {corr_threshold}")

        if len(firing_rates_df) < 2:
            print(f"firing_rates for {combined_key} does not contain enough rows for correlation data. Skipping...")
            continue

        # Get the firing data (excluding correlation rows)
        firing_data = firing_rates_df.iloc[:-2]

        # Ensure essential columns exist
        if "group" not in avg_voltage_df.columns or "avg_voltage" not in avg_voltage_df.columns:
            print(f"'group' or 'avg_voltage' column missing in avg_voltage_df for {combined_key}. Skipping...")
            continue
        if "group" not in firing_data.columns:
            print(f"'group' column missing in firing_rates for {combined_key}. Skipping...")
            continue

        # Get cluster columns (excluding 'group' and any other non-cluster columns)
        non_cluster_cols = ["group"]
        cluster_cols = [c for c in firing_data.columns if c not in non_cluster_cols]
        print(f"[DEBUG] Cluster columns in firing_data: {cluster_cols}")
        print(f"[DEBUG] Cluster columns types: {[type(c) for c in cluster_cols]}")
        if not cluster_cols:
            print(f"No neuron columns found in firing_rates for {combined_key}. Skipping...")
            continue

        print(f"Total clusters in analysis: {len(cluster_cols)}")
        total_clusters += len(cluster_cols)

        # Filter clusters based on correlation threshold and labels
        cluster_cols_filtered = []
        for clus in cluster_cols:
            # Try to extract integer cluster ID from column name
            clus_int = None
            if isinstance(clus, int):
                clus_int = clus
            elif isinstance(clus, str):
                if clus.startswith('cluster_'):
                    try:
                        clus_int = int(clus.replace('cluster_', ''))
                    except Exception:
                        clus_int = None
                else:
                    try:
                        clus_int = int(clus)
                    except Exception:
                        clus_int = None
            # Print debug info for matching
            print(f"[DEBUG] Checking cluster column: {clus} (int: {clus_int})")
            # Matching logic: allow int or string match
            match = False
            # If clusters_to_plot is a list of ints, match on clus_int
            if any(isinstance(x, int) for x in clusters_to_plot):
                if clus_int is not None and clus_int in clusters_to_plot:
                    match = True
            # If clusters_to_plot is a list of strings, match on clus
            if not match and any(isinstance(x, str) for x in clusters_to_plot):
                if clus in clusters_to_plot or (clus_int is not None and f'cluster_{clus_int}' in clusters_to_plot):
                    match = True
            print(f"[DEBUG]   - Match with clusters_to_plot: {match}")
            # Try both string and int keys for correlations
            corr_key_str = f'cluster_{clus_int}' if clus_int is not None else clus
            corr_key_int = clus_int
            corr_val = None
            if corr_key_str in correlations:
                corr_val = correlations[corr_key_str]
            elif corr_key_int in correlations:
                corr_val = correlations[corr_key_int]
            elif clus in correlations:
                corr_val = correlations[clus]
            print(f"[DEBUG]   - Correlation value: {corr_val}")
            if not match:
                print(f"[DEBUG]   - Skipping cluster {clus} (no match in clusters_to_plot)")
                continue
            if corr_val is None:
                print(f"[DEBUG]   - Correlation for cluster {clus} not found in {combined_key}. Skipping...")
                continue
            try:
                if abs(corr_val) >= corr_threshold:
                    cluster_cols_filtered.append(clus)
                    print(f"[DEBUG]   - Cluster {clus} passed correlation threshold with value {corr_val:.3f}")
                else:
                    print(f"[DEBUG]   - Cluster {clus} did NOT pass correlation threshold (|{corr_val:.3f}| < {corr_threshold})")
            except Exception as e:
                print(f"Error processing correlation for cluster {clus} in {combined_key}: {e}. Skipping...")
                continue

        print(f"Clusters after correlation filtering: {len(cluster_cols_filtered)}")
        filtered_clusters += len(cluster_cols_filtered)

        if not cluster_cols_filtered:
            print(f"No clusters in {combined_key} meet corr_threshold={corr_threshold}. Skipping...")
            continue

        # Compute ratios: divide firing rate by avg_voltage
        ratio_df = firing_data[["group"] + cluster_cols_filtered].copy()
        for cluster in cluster_cols_filtered:
            ratio_df[cluster] = ratio_df[cluster] / avg_voltage_df["avg_voltage"]
        ratio_long = ratio_df.melt(id_vars="group", value_vars=cluster_cols_filtered,
                                   var_name="cluster", value_name="ratio")
        ratio_summary = ratio_long.groupby(["cluster", "group"])["ratio"].mean().unstack("group")
        for grp in ["pre-stim", "post-stim"]:
            if grp not in ratio_summary.columns:
                ratio_summary[grp] = np.nan
        print(f"[DEBUG] ratio_summary for {combined_key}:\n{ratio_summary}")
        for cluster_id, row in ratio_summary.iterrows():
            print(f"[DEBUG] Cluster {cluster_id}: pre-stim={{row.get('pre-stim')}}, post-stim={{row.get('post-stim')}}")
            if pd.notna(row["pre-stim"]) and pd.notna(row["post-stim"]):
                all_points.append((row["pre-stim"], row["post-stim"], freq_hz))

    print(f"\nSummary:")
    print(f"Total clusters across all trials: {total_clusters}")
    print(f"Clusters after correlation filtering: {filtered_clusters}")
    print(f"Number of data points to plot: {len(all_points)}")

    if not all_points:
        print("No valid data points to plot after correlation filtering.")
        return

    all_points_df = pd.DataFrame(all_points, columns=['pre_stim', 'post_stim', 'freq_hz'])
    all_points_df['freq_hz'] = pd.to_numeric(all_points_df['freq_hz'], errors='coerce')
    all_points_df = all_points_df.dropna(subset=['freq_hz'])
    unique_freqs = np.unique(all_points_df['freq_hz'])
    print(f"Unique frequencies in plot: {unique_freqs}")
    
    cmap = plt.get_cmap('tab10')
    freq_to_color = {f: cmap(i % 10) for i, f in enumerate(np.sort(unique_freqs))}
    plt.figure(figsize=(8, 6))
    for f in unique_freqs:
        freq_points = all_points_df[all_points_df['freq_hz'] == f]
        plt.scatter(freq_points['pre_stim'], freq_points['post_stim'],
                    color=freq_to_color[f], alpha=0.7, label=f'{f} Hz')
        if len(freq_points) > 1:
            slope = np.sum(freq_points['pre_stim'] * freq_points['post_stim']) / np.sum(freq_points['pre_stim'] ** 2)
            xvals = np.linspace(freq_points['pre_stim'].min(), freq_points['pre_stim'].max(), 100)
            plt.plot(xvals, slope * xvals, color=freq_to_color[f], linestyle='-', linewidth=2, alpha=0.7)
    all_vals = np.concatenate([all_points_df['pre_stim'].dropna(), all_points_df['post_stim'].dropna()])
    min_val, max_val = np.nanmin(all_vals), np.nanmax(all_vals)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x (no change)')
    plt.xlabel('Pre-DRGS Reactivity (Hz / μV)', fontsize=14, labelpad=8)
    plt.ylabel('Post-DRGS Reactivity (Hz / μV)',fontsize=14, labelpad=8)
    plt.title(f'Neuron Reactivity, Pre vs. Post DRGS',fontsize=16, fontweight='bold', pad=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Stim Frequency')
    plt.tight_layout()
    plt.show()

def build_cluster_label_dict_from_xlsx(cluster_labels_xlsx, cluster_labels, combined_results):
    """
    Build a mapping from trial name to allowed cluster IDs using the spreadsheet and the user's filter.
    cluster_labels: can be a string, list, per-rat dict, or nested dict (same as before).
    """
    import pandas as pd
    df = pd.read_excel(cluster_labels_xlsx)
    df.columns = [col.strip() for col in df.columns]
    # Build a mapping: {rat: {trial: {label: [clusters]}}}
    label_map = {}
    for _, row in df.iterrows():
        rat = str(row['Rat'])
        trial = str(row['Trial'])
        label = str(row['Label']).strip().lower()
        cluster = row['Cluster']
        label_map.setdefault(rat, {}).setdefault(trial, {}).setdefault(label, []).append(cluster)
    # Now, for each trial in combined_results, get the allowed clusters
    cluster_dict = {}
    for trial_key in combined_results.keys():
        rat_id = trial_key.split('_')[0]
        trial_name = '_'.join(trial_key.split('_')[1:])
        # Determine which labels to use for this trial
        if isinstance(cluster_labels, dict):
            # Per-rat or per-trial
            if rat_id in cluster_labels:
                if isinstance(cluster_labels[rat_id], dict):
                    # Per-trial
                    labels = cluster_labels[rat_id].get(trial_name, cluster_labels[rat_id].get('default', []))
                else:
                    labels = cluster_labels[rat_id]
            else:
                labels = cluster_labels.get('default', [])
        else:
            labels = cluster_labels
        # Normalize to list
        if isinstance(labels, str):
            labels = [labels]
        labels = [str(l).strip().lower() for l in labels]
        # Get clusters for these labels
        clusters = []
        for label in labels:
            clusters += label_map.get(rat_id, {}).get(trial_name, {}).get(label, [])
        cluster_dict[trial_key] = sorted(set(clusters))
    return cluster_dict
