"""
• Consolidated repeated steps (e.g. retrieving channel index and properties) into helper functions.
• Adopted pathlib consistently for file paths.
• Cleaned up repeated code in the trial-info extraction for the Von Frey plots.
• Added inline comments to clarify sections and changes.

"""

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from math import ceil
from pathlib import Path
from collections import OrderedDict, Counter

# =============================================================================
# Helper Functions
# =============================================================================

def _get_channel_info(recording, channel_id):
    """
    Retrieve the index, gain, and offset for a given channel_id from a recording.
    """
    channel_ids = recording.get_channel_ids()
    try:
        channel_idx = list(channel_ids).index(channel_id)
    except ValueError:
        raise ValueError(f"Channel '{channel_id}' not found in the recording.")
    gain = recording.get_property('gain_to_uV')[channel_idx]
    offset = recording.get_property('offset_to_uV')[channel_idx]
    return channel_idx, gain, offset

def _extract_trial_info(vfa_instance, trial_name):
    """
    Extract trial number and frequency from the VonFreyAnalysis instance.
    Assumes trial name is of the form "VF_x_..." and that the qst_trial_notes index
    contains the trial number.
    Returns (trial_num, freq_hz) or (None, None) if extraction fails.
    """
    trial_parts = trial_name.split('_')
    if len(trial_parts) < 2:
        print(f"Could not extract trial number from trial_name: {trial_name}")
        return None, None
    try:
        trial_num = int(trial_parts[1])
    except ValueError:
        print(f"Unable to convert trial number to int from: {trial_parts[1]}")
        return None, None

    if trial_num not in vfa_instance.rat.qst_trial_notes.index:
        print(f"Trial number {trial_num} not found in qst_trial_notes.")
        return trial_num, None

    freq_hz = vfa_instance.rat.qst_trial_notes.loc[trial_num, 'Freq. (Hz)']
    return trial_num, freq_hz

# =============================================================================
# Interactive and Static Trace Plots
# =============================================================================

def interactive_trace(recording, channel_id='A-000', downsample_factor=10, title=None, save_csv=None):
    """
    Plot an interactive voltage trace using Plotly.
    """
    # Retrieve channel information and trace
    _, gain, offset = _get_channel_info(recording, channel_id)
    trace = recording.get_traces(channel_ids=[channel_id])
    voltage = trace * gain + offset

    sampling_rate = recording.get_sampling_frequency()
    n_samples = len(voltage)
    time = np.arange(n_samples) / sampling_rate

    # Clean and downsample
    mask = ~np.isnan(time) & ~np.isnan(voltage) & ~np.isinf(time) & ~np.isinf(voltage)
    time, voltage = time[mask], voltage[mask]
    time, voltage = time[::downsample_factor], voltage[::downsample_factor]

    if save_csv is not None:
        df = pd.DataFrame({'time_s': time, 'voltage_uV': voltage})
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        print(f"Exported time-voltage data to CSV: {save_csv}")

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=time, y=voltage, mode='lines', name='Voltage Trace', line=dict(color='blue')
    ))
    plot_title = title if title is not None else f'Voltage vs Time for Channel {channel_id}'
    fig.update_layout(
        title=plot_title,
        xaxis_title='Time (s)',
        yaxis_title='Voltage (µV)',
        showlegend=True
    )
    fig.show()

def static_trace(recording, channel_id='A-000', start_time=0, end_time=10, title=None):
    """
    Plot a static voltage trace using Matplotlib.
    """
    _, gain, offset = _get_channel_info(recording, channel_id)
    # Assuming return_scaled=True gives voltage directly
    trace = recording.get_traces(channel_ids=[channel_id], return_scaled=True)
    voltage = trace
    sampling_rate = recording.get_sampling_frequency()
    n_samples = len(voltage)
    time = np.arange(n_samples) / sampling_rate

    time, voltage = time.flatten(), voltage.flatten()
    start_idx, end_idx = int(start_time * sampling_rate), int(end_time * sampling_rate)
    time_window, voltage_window = time[start_idx:end_idx], voltage[start_idx:end_idx]

    plt.figure(figsize=(15, 5))
    plt.plot(time_window, voltage_window, label=f'Channel {channel_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (µV)')
    plot_title = title if title is not None else f'Voltage Trace for Channel {channel_id} ({start_time}-{end_time} seconds)'
    plt.title(plot_title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================================================================
# Von Frey Pre/Post Stimulus Plots per Trial
# =============================================================================

def vf_pre_post_stim_per_trial(von_frey_analysis_instance):
    """
    For each trial in the VonFreyAnalysis.windowed_results, produce a scatter plot comparing
    pre-stimulus vs. post-stimulus ratios (firing_rate/avg_voltage) per cluster.
    """
    for trial_name, data_dict in von_frey_analysis_instance.windowed_results.items():
        avg_voltage_df = data_dict['avg_voltage_df']
        firing_rates_df = data_dict['firing_rates_df']

        trial_num, freq_hz = _extract_trial_info(von_frey_analysis_instance, trial_name)
        if trial_num is None or freq_hz is None:
            continue

        # Ensure required columns exist
        if 'group' not in avg_voltage_df.columns or 'avg_voltage' not in avg_voltage_df.columns:
            print(f"Missing columns in {trial_name}'s average voltage DataFrame.")
            continue
        if 'group' not in firing_rates_df.columns:
            print(f"Missing 'group' in firing_rates_df for {trial_name}.")
            continue

        non_cluster_cols = ['group']
        cluster_cols = [c for c in firing_rates_df.columns if c not in non_cluster_cols]
        if not cluster_cols:
            print(f"No cluster columns found in firing_rates_df for {trial_name}.")
            continue

        # Compute ratio per cluster and sub-window
        ratio_df = firing_rates_df.copy()
        for cluster in cluster_cols:
            ratio_df[cluster] = ratio_df[cluster] / avg_voltage_df['avg_voltage']

        ratio_long = ratio_df.melt(id_vars='group', value_vars=cluster_cols,
                                   var_name='cluster', value_name='ratio')
        ratio_summary = ratio_long.groupby(['cluster', 'group'])['ratio'].mean().unstack('group')
        for grp in ['pre-stim', 'post-stim']:
            if grp not in ratio_summary.columns:
                ratio_summary[grp] = np.nan

        plt.figure(figsize=(8, 6))
        above_line = ratio_summary['post-stim'] > ratio_summary['pre-stim']
        below_line = ratio_summary['post-stim'] < ratio_summary['pre-stim']
        on_line = (ratio_summary['post-stim'] == ratio_summary['pre-stim']) & ~ratio_summary['post-stim'].isna()

        plt.scatter(ratio_summary.loc[above_line, 'pre-stim'],
                    ratio_summary.loc[above_line, 'post-stim'],
                    color='green', alpha=0.7, label='Increased firing (post-stim)')
        plt.scatter(ratio_summary.loc[below_line, 'pre-stim'],
                    ratio_summary.loc[below_line, 'post-stim'],
                    color='red', alpha=0.7, label='Decreased firing (post-stim)')
        if on_line.any():
            plt.scatter(ratio_summary.loc[on_line, 'pre-stim'],
                        ratio_summary.loc[on_line, 'post-stim'],
                        color='blue', alpha=0.7, label='No change')

        all_vals = np.concatenate([ratio_summary['pre-stim'].dropna(), ratio_summary['post-stim'].dropna()])
        if all_vals.size:
            min_val, max_val = np.nanmin(all_vals), np.nanmax(all_vals)
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x (no change)')
        plt.xlabel('Pre-Stim Avg Ratio (firing_rate/voltage)')
        plt.ylabel('Post-Stim Avg Ratio (firing_rate/voltage)')
        plt.title(f'{trial_name} | Stim Freq: {freq_hz} Hz')
        plt.legend()
        plt.tight_layout()
        plt.show()

# =============================================================================
# Aggregated Trial Plots
# =============================================================================

def vf_pre_post_stim_all_trials(von_frey_analysis_instance):
    """
    Combine data from all trials and create a single scatter plot.
    """
    all_points = []
    for trial_name, data_dict in von_frey_analysis_instance.windowed_results.items():
        avg_voltage_df = data_dict['avg_voltage_df']
        firing_rates_df = data_dict['firing_rates_df']
        _, freq_hz = _extract_trial_info(von_frey_analysis_instance, trial_name)
        if freq_hz is None:
            continue

        if 'group' not in avg_voltage_df.columns or 'avg_voltage' not in avg_voltage_df.columns:
            print(f"Missing columns in {trial_name}'s DataFrames.")
            continue
        if 'group' not in firing_rates_df.columns:
            print(f"Missing 'group' in firing_rates_df for {trial_name}.")
            continue

        non_cluster_cols = ['group']
        cluster_cols = [c for c in firing_rates_df.columns if c not in non_cluster_cols]
        if not cluster_cols:
            print(f"No cluster columns found in firing_rates_df for {trial_name}.")
            continue

        ratio_df = firing_rates_df.copy()
        for cluster in cluster_cols:
            ratio_df[cluster] = ratio_df[cluster] / avg_voltage_df['avg_voltage']
        ratio_long = ratio_df.melt(id_vars='group', value_vars=cluster_cols,
                                   var_name='cluster', value_name='ratio')
        ratio_summary = ratio_long.groupby(['cluster', 'group'])['ratio'].mean().unstack('group')
        for grp in ['pre-stim', 'post-stim']:
            if grp not in ratio_summary.columns:
                ratio_summary[grp] = np.nan
        for cluster_id, row in ratio_summary.iterrows():
            pre_val, post_val = row['pre-stim'], row['post-stim']
            if pd.notna(pre_val) and pd.notna(post_val):
                all_points.append((pre_val, post_val, freq_hz))

    if not all_points:
        print("No valid data points to plot.")
        return

    all_points_df = pd.DataFrame(all_points, columns=['pre_stim', 'post_stim', 'freq_hz'])
    all_points_df['freq_hz'] = pd.to_numeric(all_points_df['freq_hz'], errors='coerce')
    all_points_df = all_points_df.dropna(subset=['freq_hz'])
    unique_freqs = np.unique(all_points_df['freq_hz'])
    cmap = plt.get_cmap('tab10')
    freq_to_color = {f: cmap(i % 10) for i, f in enumerate(unique_freqs)}

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
    if all_vals.size:
        min_val, max_val = np.nanmin(all_vals), np.nanmax(all_vals)
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x (no change)')
    plt.xlabel('Pre-Stim Avg Ratio (firing_rate/voltage)')
    plt.ylabel('Post-Stim Avg Ratio (firing_rate/voltage)')
    plt.title('Pre vs. Post Stim Ratio: All Trials')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Stim Frequency')
    plt.tight_layout()
    plt.show()

def vf_pre_post_stim_all_trials_correlated(von_frey_analysis_instance, corr_threshold=0.1):
    """
    Similar to vf_pre_post_stim_all_trials but only includes clusters
    whose inverse-ISI correlation meets or exceeds the specified threshold.
    """
    all_points = []
    for trial_name, data_dict in von_frey_analysis_instance.windowed_results.items():
        avg_voltage_df = data_dict['avg_voltage_df']
        firing_rates_df = data_dict['firing_rates_df']
        _, freq_hz = _extract_trial_info(von_frey_analysis_instance, trial_name)
        if freq_hz is None:
            continue
        if trial_name not in von_frey_analysis_instance.inv_isi_correlations:
            print(f"No inverse-ISI correlation data for {trial_name}. Skipping.")
            continue
        correlations_dict = von_frey_analysis_instance.inv_isi_correlations[trial_name]
        if 'group' not in avg_voltage_df.columns or 'avg_voltage' not in avg_voltage_df.columns:
            print(f"Missing 'group' or 'avg_voltage' in {trial_name}'s DataFrames.")
            continue
        if 'group' not in firing_rates_df.columns:
            print(f"Missing 'group' in firing_rates_df for {trial_name}.")
            continue

        non_cluster_cols = ['group']
        cluster_cols = [c for c in firing_rates_df.columns if c not in non_cluster_cols]
        cluster_cols_filtered = []
        for clus in cluster_cols:
            try:
                cluster_id = int(clus)
            except ValueError:
                cluster_id = clus
            if abs(correlations_dict.get(cluster_id, 0.0)) >= corr_threshold:
                cluster_cols_filtered.append(clus)
        if not cluster_cols_filtered:
            print(f"No clusters in trial '{trial_name}' meet corr_threshold={corr_threshold}. Skipping.")
            continue

        ratio_df = firing_rates_df[['group'] + cluster_cols_filtered].copy()
        for cluster in cluster_cols_filtered:
            ratio_df[cluster] = ratio_df[cluster] / avg_voltage_df['avg_voltage']
        ratio_long = ratio_df.melt(id_vars='group', value_vars=cluster_cols_filtered,
                                   var_name='cluster', value_name='ratio')
        ratio_summary = ratio_long.groupby(['cluster', 'group'])['ratio'].mean().unstack('group')
        for grp in ['pre-stim', 'post-stim']:
            if grp not in ratio_summary.columns:
                ratio_summary[grp] = np.nan
        for cluster_id, row in ratio_summary.iterrows():
            if pd.notna(row['pre-stim']) and pd.notna(row['post-stim']):
                all_points.append((row['pre-stim'], row['post-stim'], freq_hz))
    if not all_points:
        print("No valid data points to plot after correlation filtering.")
        return
    all_points_df = pd.DataFrame(all_points, columns=['pre_stim', 'post_stim', 'freq_hz'])
    all_points_df['freq_hz'] = pd.to_numeric(all_points_df['freq_hz'], errors='coerce')
    all_points_df = all_points_df.dropna(subset=['freq_hz'])
    unique_freqs = np.unique(all_points_df['freq_hz'])
    cmap = plt.get_cmap('tab10')
    freq_to_color = {f: cmap(i % 10) for i, f in enumerate(unique_freqs)}
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
    if all_vals.size:
        min_val, max_val = np.nanmin(all_vals), np.nanmax(all_vals)
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x (no change)')
    plt.xlabel('Pre-Stim Avg Ratio (firing_rate/voltage)')
    plt.ylabel('Post-Stim Avg Ratio (firing_rate/voltage)')
    plt.title(f'Pre vs. Post Stim Ratio: All Trials | corr_threshold={corr_threshold}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Stim Frequency')
    plt.tight_layout()
    plt.show()

# =============================================================================
# Aggregated Animal Plots
# =============================================================================


def vf_all_trials_combined_plot(combined_results, combined_qst_notes, corr_threshold=0.1):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Create a dictionary mapping Trial_ID to Freq. (Hz)
    freq_dict = dict(zip(combined_qst_notes['Trial_ID'], combined_qst_notes['Freq. (Hz)']))
    # Create a dictionary mapping current amplitude to amp
    amp_dict = dict(zip(combined_qst_notes['Trial_ID'], combined_qst_notes['amp']))


    all_points = []
    # Loop over each trial result in the combined_results dictionary.
    for combined_key, res in combined_results.items():
        # Retrieve the stimulation frequency from the frequency dictionary using the trial key.
        if combined_key not in freq_dict:
            raise ValueError(f"Frequency not found for trial {combined_key} in combined_qst_notes.")
        freq_hz = freq_dict[combined_key]

        if "voltage_df" not in res:
            raise KeyError(f"'voltage_df' not found in result for {combined_key}.")
        if "firing_df" not in res:
            raise KeyError(f"'firing_df' not found in result for {combined_key}.")

        avg_voltage_df = res["voltage_df"]
        firing_rates_df = res["firing_df"]

        if len(firing_rates_df) < 2:
            raise ValueError(f"firing_rates_df for {combined_key} does not contain enough rows for correlation data.")

        # Assume the correlation row is the second-to-last row.
        firing_data = firing_rates_df.iloc[:-2]
        correlation_data = firing_rates_df.iloc[-2]

        # Remove unwanted keys (like "group" or "Unnamed: 0") and convert remaining keys to integers
        corr_dict = {int(k): v for k, v in correlation_data.to_dict().items() if str(k).isdigit()}
        correlations_dict = corr_dict

        # Ensure essential columns exist.
        if "group" not in avg_voltage_df.columns or "avg_voltage" not in avg_voltage_df.columns:
            raise KeyError(f"'group' or 'avg_voltage' column missing in avg_voltage_df for {combined_key}.")
        if "group" not in firing_data.columns:
            raise KeyError(f"'group' column missing in firing_rates_df for {combined_key}.")

        non_cluster_cols = ["group"]
        cluster_cols = [c for c in firing_data.columns if c not in non_cluster_cols]
        if not cluster_cols:
            raise ValueError(f"No neuron columns found in firing_rates_df for {combined_key}.")

        cluster_cols_filtered = []
        for clus in cluster_cols:
            try:
                cluster_id = int(clus)
            except ValueError:
                cluster_id = clus
            if cluster_id not in correlations_dict:
                # raise KeyError(f"Correlation for cluster {cluster_id} not found in {combined_key}.")
                print(f"Correlation for cluster {cluster_id} not found in {combined_key}. Skipping...")
                pass
            try:
                if abs(correlations_dict[cluster_id]) >= corr_threshold:
                    cluster_cols_filtered.append(clus)
            except :
                print(f"Correlation for cluster {cluster_id} not found in {combined_key}. Skipping...")
                pass
        if not cluster_cols_filtered:
            ## changed the code to not raise an error, and continue with warning
            # raise ValueError(f"No clusters in {combined_key} meet corr_threshold={corr_threshold}.")
            print(f"No clusters in {combined_key} meet corr_threshold={corr_threshold}. Skipping...")
            continue

        # Compute ratios: divide firing rate by avg_voltage.
        ratio_df = firing_data[["group"] + cluster_cols_filtered].copy()
        for cluster in cluster_cols_filtered:
            ratio_df[cluster] = ratio_df[cluster] / avg_voltage_df["avg_voltage"]
        ratio_long = ratio_df.melt(id_vars="group", value_vars=cluster_cols_filtered,
                                   var_name="cluster", value_name="ratio")
        ratio_summary = ratio_long.groupby(["cluster", "group"])["ratio"].mean().unstack("group")
        for grp in ["pre-stim", "post-stim"]:
            if grp not in ratio_summary.columns:
                ratio_summary[grp] = np.nan
        for cluster_id, row in ratio_summary.iterrows():
            if pd.notna(row["pre-stim"]) and pd.notna(row["post-stim"]):
                all_points.append((row["pre-stim"], row["post-stim"], freq_hz))

    if not all_points:
        raise ValueError("No valid data points to plot after correlation filtering.")

    all_points_df = pd.DataFrame(all_points, columns=['pre_stim', 'post_stim', 'freq_hz'])
    all_points_df['freq_hz'] = pd.to_numeric(all_points_df['freq_hz'], errors='coerce')
    all_points_df = all_points_df.dropna(subset=['freq_hz'])
    unique_freqs = np.unique(all_points_df['freq_hz'])
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
    ## not using this title with the correlation threshold directly listed inside
    # plt.title(f'Neuron Reactivity, Pre vs. Post DRGS | Correlation >= {corr_threshold}',fontsize=16, fontweight='bold', pad=10)
    plt.title(f'Neuron Reactivity, Pre vs. Post DRGS',fontsize=16, fontweight='bold', pad=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Stim Frequency')
    plt.tight_layout()
    plt.show()


# =============================================================================
# Other Plot Functions
# =============================================================================

def plot_von_frey_raster_and_trace(von_frey_analysis_instance, trial_name, title=None):
    """
    Plot the Von Frey voltage trace (top) and a raster plot (bottom) for a given trial.
    """
    if trial_name not in von_frey_analysis_instance.spikes.kilosort_results:
        print(f"No kilosort results found for trial: {trial_name}.")
        return
    ks_output = von_frey_analysis_instance.spikes.kilosort_results[trial_name]
    st, clu, fs = ks_output['spike_times'], ks_output['spike_clusters'], ks_output['ops']['fs']
    unique_clusters = np.unique(clu)
    cluster_to_y = {c: i for i, c in enumerate(unique_clusters)}
    if trial_name not in von_frey_analysis_instance.signals.data.analog_data:
        print(f"Trial '{trial_name}' not found in analog_data.")
        return
    recording = von_frey_analysis_instance.signals.data.analog_data[trial_name]
    sampling_rate = recording.get_sampling_frequency()
    if 'ANALOG-IN-2' not in recording.get_channel_ids():
        print(f"ANALOG-IN-2 not found in trial '{trial_name}'.")
        return
    vf_data = recording.get_traces(channel_ids=['ANALOG-IN-2'], return_scaled=True).flatten()
    time_vector = np.arange(len(vf_data)) / sampling_rate
    spike_times_sec = st / fs
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])
    ax_top = fig.add_subplot(gs[0])
    ax_top.plot(time_vector, vf_data, color='blue', linewidth=0.5)
    ax_top.set_ylabel('Voltage (uV)')
    ax_top.set_title(title if title is not None else f"Von Frey Voltage Trace: {trial_name}")
    ax_top.set_xlim([0, time_vector[-1]])
    ax_bottom = fig.add_subplot(gs[1])
    mask = np.isin(clu, list(cluster_to_y.keys()))
    ax_bottom.scatter(spike_times_sec[mask], [cluster_to_y[c] for c in clu[mask]], s=1, color='k', alpha=0.5)
    ax_bottom.set_xlabel('Time (sec)')
    ax_bottom.set_ylabel('Channel (derived from cluster)')
    ax_bottom.set_title("Raster Plot of Spikes")
    ax_bottom.set_xlim([time_vector[0], time_vector[-1]])
    ax_bottom.set_ylim([max(cluster_to_y.values()), min(cluster_to_y.values())])
    plt.tight_layout()
    plt.show()

def plot_vf_spike_raster_filtered_units(von_frey_analysis_instance, trial_name, corr_threshold=0.1, title=None):
    """
    Plot a raster of spikes for filtered units (clusters) based on correlation threshold.
    """
    if trial_name not in von_frey_analysis_instance.spikes.kilosort_results:
        print(f"No kilosort results for trial '{trial_name}'.")
        return
    ks_output = von_frey_analysis_instance.spikes.kilosort_results[trial_name]
    st, clu, fs = ks_output['spike_times'], ks_output['spike_clusters'], ks_output['ops']['fs']
    spike_times_sec = st / fs
    if trial_name not in von_frey_analysis_instance.signals.data.analog_data:
        print(f"Trial '{trial_name}' not found in analog_data.")
        return
    recording = von_frey_analysis_instance.signals.data.analog_data[trial_name]
    sampling_rate_vf = recording.get_sampling_frequency()
    if 'ANALOG-IN-2' not in recording.get_channel_ids():
        print(f"ANALOG-IN-2 not found in trial '{trial_name}'.")
        return
    vf_data = recording.get_traces(channel_ids=['ANALOG-IN-2'], return_scaled=True).flatten()
    time_vector = np.arange(len(vf_data)) / sampling_rate_vf
    unique_clusters = np.unique(clu)
    # Compute correlation per cluster via a simple approach (see your inverse-ISI method)
    cluster_correlations = {}
    for cluster in unique_clusters:
        spikes_c = np.sort(spike_times_sec[clu == cluster])
        if len(spikes_c) < 2:
            cluster_correlations[cluster] = 0
            continue
        time_since_last = np.zeros(len(vf_data))
        last_idx = 0
        spike_idx = 0
        for i, t in enumerate(time_vector):
            while spike_idx < len(spikes_c) and spikes_c[spike_idx] <= t:
                last_idx = spike_idx
                spike_idx += 1
            time_since_last[i] = t - spikes_c[last_idx]
        corr = 0 if np.std(time_since_last)==0 or np.std(vf_data)==0 else np.corrcoef(time_since_last, vf_data)[0,1]
        cluster_correlations[cluster] = corr
    filtered_clusters = [c for c in unique_clusters if abs(cluster_correlations.get(c, 0)) >= corr_threshold]
    if not filtered_clusters:
        print(f"No clusters meet the correlation threshold for trial '{trial_name}'.")
        return
    filtered_clusters_sorted = np.sort(filtered_clusters)
    cluster_to_y = {c: i for i, c in enumerate(filtered_clusters_sorted)}
    mask = np.isin(clu, filtered_clusters)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    ax[0].plot(time_vector, vf_data, color='blue')
    ax[0].set_ylabel('Voltage (uV)')
    ax[0].set_title(title if title is not None else f'{trial_name} - Von Frey Trace & Filtered Raster')
    ax[1].scatter(spike_times_sec[mask], [cluster_to_y[c] for c in clu[mask]], s=0.5, color='k', alpha=0.5)
    ax[1].set_ylabel('Filtered Units')
    ax[1].set_xlabel('Time (sec)')
    ax[1].set_yticks(range(len(filtered_clusters_sorted)))
    ax[1].set_yticklabels(filtered_clusters_sorted)
    ax[1].set_xlim([time_vector[0], time_vector[-1]])
    plt.tight_layout()
    plt.show()

def plot_inv_isi_vs_von_frey_all_clusters(von_frey_data: np.ndarray, 
                                          inv_isi_traces: dict, 
                                          correlations: dict, 
                                          trial_name: str, 
                                          title: str = None):
    """
    Plot inverse-ISI traces and Von Frey data for each cluster in a trial.
    Each cluster is shown in its own subplot with the correlation coefficient.
    """
    num_clusters = len(inv_isi_traces)
    if num_clusters == 0:
        print(f"No clusters to plot for trial '{trial_name}'.")
        return
    ncols = 2
    nrows = ceil(num_clusters / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows), sharex=True)
    if num_clusters == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    for idx, (cluster_id, inv_trace) in enumerate(inv_isi_traces.items()):
        row, col = idx // ncols, idx % ncols
        ax_vf = axes[row][col]
        ax_vf.plot(von_frey_data, label='Von Frey', color='blue')
        ax_vf.set_title(f'Cluster {cluster_id} | Corr: {correlations.get(cluster_id, 0):.3f}')
        ax_vf.legend(loc='upper left')
        ax_vf.set_ylabel('Von Frey (uV)')
        ax_inv = ax_vf.twinx()
        ax_inv.plot(inv_trace, color='orange', label='Inv-ISI')
        ax_inv.set_ylabel('Inv-ISI')
        ax_inv.legend(loc='upper right')
    total_subplots = nrows * ncols
    if num_clusters < total_subplots:
        for empty_idx in range(num_clusters, total_subplots):
            row, col = empty_idx // ncols, empty_idx % ncols
            fig.delaxes(axes[row][col])
    plt.xlabel('Samples')
    fig.suptitle(title if title else f'Inverse-ISI vs. Von Frey Data for Trial {trial_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

# =============================================================================
# Modular Spike Data Plotting Functions
# =============================================================================

def plot_raster(spike_times, clusters, cluster_ids=None, fs=30000, ax=None, title=None):
    """
    Plot a raster of spike times for given clusters. spike_times and clusters should be 1D arrays of same length.
    cluster_ids: list/array of clusters to plot (default: all unique in clusters)
    fs: sampling rate (Hz) for converting spike_times if needed
    ax: matplotlib axis to plot on (optional)
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    if cluster_ids is None:
        cluster_ids = np.unique(clusters)
    cluster_to_y = {c: i for i, c in enumerate(cluster_ids)}
    
    # Plot spikes
    for c in cluster_ids:
        idx = clusters == c
        ax.scatter(spike_times[idx], np.full(np.sum(idx), cluster_to_y[c]), 
                  marker='|', s=100, label=f'Cluster {c}')
    
    # Customize axis labels and ticks
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cluster')
    ax.set_yticks(list(cluster_to_y.values()))
    ax.set_yticklabels(list(cluster_to_y.keys()))
    
    # Set title
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Spike Raster')
        
    # Customize legend
    if len(cluster_ids) > 10:
        # For many clusters, place legend outside
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 borderaxespad=0., ncol=max(1, len(cluster_ids)//20))
    else:
        # For fewer clusters, keep legend inside
        ax.legend(loc='upper right')
        
    plt.tight_layout()
    plt.show()

def plot_waveform(waveform, t=None, ax=None, title=None):
    """
    Plot a single mean waveform (1D or 2D array). t is time axis (optional).
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    if waveform is None:
        print('No waveform to plot.')
        return
    if waveform.ndim == 1:
        ax.plot(t if t is not None else np.arange(len(waveform)), waveform, label='Mean Waveform')
    else:
        for i in range(waveform.shape[1]):
            ax.plot(t if t is not None else np.arange(waveform.shape[0]), waveform[:, i], label=f'Ch {i}')
    ax.set_xlabel('Time (samples)' if t is None else 'Time (ms)')
    ax.set_ylabel('Amplitude (a.u.)')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Mean Waveform')
    ax.legend()
    plt.tight_layout()
    plt.show()
