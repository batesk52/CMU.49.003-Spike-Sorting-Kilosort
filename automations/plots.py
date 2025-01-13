# plots.py

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

def interactive_trace(recording, channel_id='A-000', downsample_factor=10, title=None):
    """
    Plots the voltage trace using Plotly.

    Parameters:
    - recording: the recording object from which to extract data
    - channel_id: string, the ID of the channel to plot
    - downsample_factor: int, factor by which to downsample the data
    - title: string, optional custom title for the plot
    """
    # Get channel IDs and find the index of the specified channel
    channel_ids = recording.get_channel_ids()
    try:
        channel_idx = list(channel_ids).index(channel_id)
    except ValueError:
        raise ValueError(f"Channel '{channel_id}' not found in the recording.")

    # Retrieve gain and offset for the specified channel
    gain_to_uV_array = recording.get_property('gain_to_uV')
    offset_to_uV_array = recording.get_property('offset_to_uV')
    gain_to_uV = gain_to_uV_array[channel_idx]
    offset_to_uV = offset_to_uV_array[channel_idx]
    print(f"gain_to_uV: {gain_to_uV}, offset_to_uV: {offset_to_uV}")

    # Retrieve the trace for the specified channel
    trace = recording.get_traces(channel_ids=[channel_id])

    # Convert raw data to voltage
    voltage = trace * gain_to_uV + offset_to_uV

    # Get sampling frequency and calculate time axis
    sampling_rate = recording.get_sampling_frequency()
    n_samples = len(voltage)
    time = np.arange(n_samples) / sampling_rate  # Time in seconds

    # Flatten the arrays
    time = time.flatten()
    voltage = voltage.flatten()

    # Optional: Clean data to remove NaNs or Infs
    mask = ~np.isnan(time) & ~np.isnan(voltage) & ~np.isinf(time) & ~np.isinf(voltage)
    time = time[mask]
    voltage = voltage[mask]

    # Optional: Downsample data
    time = time[::downsample_factor]
    voltage = voltage[::downsample_factor]

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=time,
        y=voltage,
        mode='lines',
        name='Voltage Trace',
        line=dict(color='blue')
    ))

    # Determine the title
    if title is None:
        plot_title = f'Voltage vs Time for Channel {channel_id}'
    else:
        plot_title = title

    # Set plot layout
    fig.update_layout(
        title=plot_title,
        xaxis_title='Time (s)',
        yaxis_title='Voltage (µV)',
        showlegend=True,
        # width=1900,
        # height=800
    )

    # Show the interactive plot
    fig.show()

def static_trace(recording, channel_id='A-000', start_time=0, end_time=10, title=None):
    """
    Plots the voltage trace using Matplotlib.

    Parameters:
    - recording: the recording object from which to extract data
    - channel_id: string, the ID of the channel to plot
    - start_time: float, start time in seconds for the plot
    - end_time: float, end time in seconds for the plot
    - title: string, optional custom title for the plot
    """
    # Get channel IDs and find the index of the specified channel
    channel_ids = recording.get_channel_ids()
    try:
        channel_idx = list(channel_ids).index(channel_id)
    except ValueError:
        raise ValueError(f"Channel '{channel_id}' not found in the recording.")

    # Retrieve gain and offset for the specified channel
    gain_to_uV_array = recording.get_property('gain_to_uV')
    offset_to_uV_array = recording.get_property('offset_to_uV')
    gain_to_uV = gain_to_uV_array[channel_idx]
    offset_to_uV = offset_to_uV_array[channel_idx]
    print(f"gain_to_uV: {gain_to_uV}, offset_to_uV: {offset_to_uV}")

    # Retrieve the trace for the specified channel
    trace = recording.get_traces(channel_ids=[channel_id], return_scaled=True)

    # Convert raw data to voltage (if needed)
    voltage = trace  # Assuming return_scaled=True already gives voltage

    # Get sampling frequency and calculate time axis
    sampling_rate = recording.get_sampling_frequency()
    n_samples = len(voltage)
    time = np.arange(n_samples) / sampling_rate  # Time in seconds

    # Flatten the arrays    
    time = time.flatten()
    voltage = voltage.flatten()  

    # Plotting a time window
    start_idx = int(start_time * sampling_rate)
    end_idx = int(end_time * sampling_rate)

    time_window = time[start_idx:end_idx]
    voltage_window = voltage[start_idx:end_idx]

    plt.figure(figsize=(15, 5))
    plt.plot(time_window, voltage_window, label=f'Channel {channel_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (µV)')
    
    # Determine the title
    if title is None:
        plot_title = f'Voltage Trace for Channel {channel_id} ({start_time}-{end_time} seconds)'
    else:
        plot_title = title
    
    plt.title(plot_title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def vf_pre_post_stim_per_trial(von_frey_analysis_instance):
    """
    Parameters:
    - results: dict returned by analyze_subwindows
    - von_frey_analysis_instance: An instance of VonFreyAnalysis, so we can access self.rat.qst_trial_notes

    This function will produce a scatter plot for each trial.
    """
    for trial_name, data_dict in von_frey_analysis_instance.windowed_results.items():
        avg_voltage_df = data_dict['avg_voltage_df']
        firing_rates_df = data_dict['firing_rates_df']

        # Extract the trial number from the trial name
        # Assuming trial names look like: "VF_10_241125_162725"
        # and you want the second element after splitting by '_', which is "10"
        trial_parts = trial_name.split('_')
        # Make sure to handle any irregularities in naming convention
        if len(trial_parts) < 2:
            print(f"Could not extract trial number from trial_name: {trial_name}")
            continue

        try:
            trial_num = int(trial_parts[1])
        except ValueError:
            print(f"Unable to convert trial number to int from: {trial_parts[1]}")
            continue

        # Retrieve the frequency from qst_trial_notes using the trial number
        # This assumes that qst_trial_notes has a row for each trial number
        # and has a column named 'Freq. (Hz)'
        if trial_num not in von_frey_analysis_instance.rat.qst_trial_notes.index:
            print(f"Trial number {trial_num} not found in qst_trial_notes.")
            continue

        freq_hz = von_frey_analysis_instance.rat.qst_trial_notes.loc[trial_num, 'Freq. (Hz)']

        # Ensure we have 'group' and 'avg_voltage' columns
        if 'group' not in avg_voltage_df.columns or 'avg_voltage' not in avg_voltage_df.columns:
            print(f"Missing 'group' or 'avg_voltage' in {trial_name}'s DataFrames.")
            continue
        if 'group' not in firing_rates_df.columns:
            print(f"Missing 'group' in firing_rates_df for {trial_name}.")
            continue

        non_cluster_cols = ['group']
        cluster_cols = [c for c in firing_rates_df.columns if c not in non_cluster_cols]

        if len(cluster_cols) == 0:
            print(f"No cluster columns found in firing_rates_df for {trial_name}.")
            continue

        # Compute ratio = firing_rate/avg_voltage per cluster & sub-window
        ratio_df = firing_rates_df.copy()
        for cluster in cluster_cols:
            ratio_df[cluster] = ratio_df[cluster] / avg_voltage_df['avg_voltage']

        # Convert to long format
        ratio_long = ratio_df.melt(id_vars='group', value_vars=cluster_cols, var_name='cluster', value_name='ratio')

        # Compute average ratio per (cluster, group)
        ratio_summary = ratio_long.groupby(['cluster', 'group'])['ratio'].mean().unstack('group')

        # Ensure both groups exist, fill missing with NaN
        for grp in ['pre-stim', 'post-stim']:
            if grp not in ratio_summary.columns:
                ratio_summary[grp] = np.nan

        plt.figure(figsize=(8, 6))

        # Determine which clusters are above or below the line y=x
        above_line = ratio_summary['post-stim'] > ratio_summary['pre-stim']
        below_line = ratio_summary['post-stim'] < ratio_summary['pre-stim']
        on_line = (ratio_summary['post-stim'] == ratio_summary['pre-stim']) & ~ratio_summary['post-stim'].isna()

        # Scatter plot for clusters that have increased firing (above line)
        plt.scatter(ratio_summary.loc[above_line, 'pre-stim'],
                    ratio_summary.loc[above_line, 'post-stim'],
                    color='green', alpha=0.7, label='Increased firing (post-stim)')

        # Scatter plot for clusters that have decreased firing (below line)
        plt.scatter(ratio_summary.loc[below_line, 'pre-stim'],
                    ratio_summary.loc[below_line, 'post-stim'],
                    color='red', alpha=0.7, label='Decreased firing (post-stim)')

        # If any cluster is exactly on the line
        if on_line.any():
            plt.scatter(ratio_summary.loc[on_line, 'pre-stim'],
                        ratio_summary.loc[on_line, 'post-stim'],
                        color='blue', alpha=0.7, label='No change')

        # Add a y=x line (trendline)
        all_vals = np.concatenate([ratio_summary['pre-stim'].dropna(), ratio_summary['post-stim'].dropna()])
        if len(all_vals) > 0:
            min_val, max_val = np.nanmin(all_vals), np.nanmax(all_vals)
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x (no change)')

        # annotate points with cluster IDs (if desired)
        for cluster_id, row in ratio_summary.iterrows():
            if pd.notna(row['pre-stim']) and pd.notna(row['post-stim']):
                plt.text(row['pre-stim'], row['post-stim'], str(cluster_id), fontsize=8, alpha=0.7)

        plt.xlabel('Pre-Stim Avg Ratio (firing_rate/voltage)')
        plt.ylabel('Post-Stim Avg Ratio (firing_rate/voltage)')
        # Include the trial name and frequency in the title
        plt.title(f'{trial_name} | Stim Freq: {freq_hz} Hz')

        # plt.grid(True, linestyle='--', alpha=0.5) # uncomment to add grid
        plt.legend()

        plt.tight_layout()
        # commenting out the save because I don't want to bother with making a smart functionion for saving directory
        # plt.savefig(f"{trial_name}_ratio_scatter.png", dpi=300)
        plt.show()


def vf_pre_post_stim_all_trials(von_frey_analysis_instance):

    """
    Given the results dictionary from analyze_subwindows, this function:
    - Combines data from all trials into one plot.
    - For each trial, computes firing_rate/voltage ratio per cluster (pre-stim and post-stim).
    - Plots all trials on a single scatter plot with x = pre-stim ratio and y = post-stim ratio.
    - Points are color-coded by the trial's frequency (Hz).
    - Adds a linear trendline for each frequency group with intercept fixed at 0.
    - The legend shows different frequencies and their corresponding colors.

    Parameters:
    - results: dict returned by analyze_subwindows
    - von_frey_analysis_instance: An instance of VonFreyAnalysis, so we can access self.rat.qst_trial_notes

    This function produces a single scatter plot combining all trials.
    """

    all_points = []

    for trial_name, data_dict in von_frey_analysis_instance.windowed_results.items():
        avg_voltage_df = data_dict['avg_voltage_df']
        firing_rates_df = data_dict['firing_rates_df']

        # Extract the trial number from the trial name
        trial_parts = trial_name.split('_')
        if len(trial_parts) < 2:
            print(f"Could not extract trial number from trial_name: {trial_name}")
            continue

        try:
            trial_num = int(trial_parts[1])
        except ValueError:
            print(f"Unable to convert trial number to int from: {trial_parts[1]}")
            continue

        if trial_num not in von_frey_analysis_instance.rat.qst_trial_notes.index:
            print(f"Trial number {trial_num} not found in qst_trial_notes.")
            continue

        freq_hz = von_frey_analysis_instance.rat.qst_trial_notes.loc[trial_num, 'Freq. (Hz)']

        if 'group' not in avg_voltage_df.columns or 'avg_voltage' not in avg_voltage_df.columns:
            print(f"Missing 'group' or 'avg_voltage' in {trial_name}'s DataFrames.")
            continue
        if 'group' not in firing_rates_df.columns:
            print(f"Missing 'group' in firing_rates_df for {trial_name}.")
            continue

        non_cluster_cols = ['group']
        cluster_cols = [c for c in firing_rates_df.columns if c not in non_cluster_cols]

        if len(cluster_cols) == 0:
            print(f"No cluster columns found in firing_rates_df for {trial_name}.")
            continue

        # Compute ratio = firing_rate/avg_voltage per cluster & sub-window
        ratio_df = firing_rates_df.copy()
        for cluster in cluster_cols:
            ratio_df[cluster] = ratio_df[cluster] / avg_voltage_df['avg_voltage']

        # Convert to long format
        ratio_long = ratio_df.melt(id_vars='group', value_vars=cluster_cols, var_name='cluster', value_name='ratio')

        # Compute average ratio per (cluster, group)
        ratio_summary = ratio_long.groupby(['cluster', 'group'])['ratio'].mean().unstack('group')

        # Ensure both groups exist, fill missing with NaN
        for grp in ['pre-stim', 'post-stim']:
            if grp not in ratio_summary.columns:
                ratio_summary[grp] = np.nan

        # Add the data points (pre-stim, post-stim) for each cluster
        for cluster_id, row in ratio_summary.iterrows():
            pre_stim_val = row['pre-stim']
            post_stim_val = row['post-stim']
            # Only add if both are not NaN
            if pd.notna(pre_stim_val) and pd.notna(post_stim_val):
                all_points.append((pre_stim_val, post_stim_val, freq_hz))

    if len(all_points) == 0:
        print("No valid data points to plot.")
        return

    # Convert to DataFrame for easier processing
    all_points_df = pd.DataFrame(all_points, columns=['pre_stim', 'post_stim', 'freq_hz'])

    # Identify unique frequencies
    unique_freqs = np.unique(all_points_df['freq_hz'])

    # Create a colormap or color cycle for frequencies
    cmap = plt.get_cmap('tab10')  # tab10 has distinct colors
    freq_to_color = {}
    for i, f in enumerate(unique_freqs):
        freq_to_color[f] = cmap(i % 10)

    # Now create a single figure
    plt.figure(figsize=(8, 6))

    # Plot all points by frequency
    for f in unique_freqs:
        freq_points = all_points_df[all_points_df['freq_hz'] == f]

        # Scatter plot for these points
        plt.scatter(freq_points['pre_stim'], freq_points['post_stim'], color=freq_to_color[f], alpha=0.7, label=f'{f} Hz')

        # Compute trendline for this frequency group with intercept fixed at 0
        if len(freq_points) > 1:
            # Perform linear regression with intercept fixed at 0
            slope = np.sum(freq_points['pre_stim'] * freq_points['post_stim']) / np.sum(freq_points['pre_stim'] ** 2)
            xvals = np.linspace(freq_points['pre_stim'].min(), freq_points['pre_stim'].max(), 100)
            yvals = slope * xvals
            plt.plot(xvals, yvals, color=freq_to_color[f], linestyle='-', linewidth=2, alpha=0.7)

    # Add a y=x line (trendline for no change)
    all_vals = np.concatenate([all_points_df['pre_stim'].dropna(), all_points_df['post_stim'].dropna()])
    if len(all_vals) > 0:
        min_val, max_val = np.nanmin(all_vals), np.nanmax(all_vals)
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x (no change)')

    plt.xlabel('Pre-Stim Avg Ratio (firing_rate/voltage)')
    plt.ylabel('Post-Stim Avg Ratio (firing_rate/voltage)')
    plt.title('Pre vs. Post Stim Ratio: All Trials')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Stim Frequency')
    plt.tight_layout()
    plt.show()


# can I get rid of this?
def plot_von_frey_raster_and_trace(von_frey_analysis_instance, trial_name, title=None):
    # get spike data
    if trial_name not in von_frey_analysis_instance.spikes.kilosort_results:
        print(f"No kilosort results found for trial: {trial_name}.")
        return
    kilosort_output = von_frey_analysis_instance.spikes.kilosort_results[trial_name]
    st = kilosort_output['spike_times']    # Spike times in samples
    clu = kilosort_output['spike_clusters']# Corresponding cluster IDs
    fs = kilosort_output['ops']['fs']      # Sampling frequency

    # If you have a channel map or mapping for clusters to channels
    # For example, if 'chan_best' maps cluster index to a channel number
    # or if you have a known channel layout:
    # chan_best = kilosort_output.get('chan_best', None)
    # chan_map = kilosort_output.get('chan_map', None)
    # If not available, just plot by cluster index
    # Here we assume a simple channel mapping based on clusters:
    unique_clusters = np.unique(clu)
    cluster_to_channel = {c: i for i, c in enumerate(unique_clusters)}

    # Step 2: Retrieve analog voltage data from ANALOG-IN-2
    if trial_name not in von_frey_analysis_instance.signals.data.analog_data:
        print(f"Trial '{trial_name}' not found in analog_data.")
        return
    recording = von_frey_analysis_instance.signals.data.analog_data[trial_name]
    sampling_rate = recording.get_sampling_frequency()
    if 'ANALOG-IN-2' not in recording.get_channel_ids():
        print(f"ANALOG-IN-2 not found in trial '{trial_name}'.")
        return

    von_frey_data = recording.get_traces(channel_ids=['ANALOG-IN-2'], return_scaled=True).flatten()
    num_samples = len(von_frey_data)
    time_vector = np.arange(num_samples) / sampling_rate

    # Step 3: Convert spike times to seconds
    spike_times_sec = st / fs

    # Step 4: Create a figure with two subplots: top for voltage, bottom for raster
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])

    # Top: Von Frey analog voltage trace
    ax_top = fig.add_subplot(gs[0])
    ax_top.plot(time_vector, von_frey_data, color='blue', linewidth=0.5)
    ax_top.set_ylabel('Voltage (uV)')
    if title is not None:
        ax_top.set_title(title)
    else:
        ax_top.set_title(f"Von Frey Analog Voltage Trace: {trial_name}")
    ax_top.set_xlim([0, time_vector[-1]])

    # Step 5: Raster plot of spikes
    # Convert clusters to channel indices (or use cluster index directly)
    spike_channels = np.array([cluster_to_channel[c] for c in clu])

    ax_bottom = fig.add_subplot(gs[1])
    ax_bottom.scatter(spike_times_sec, spike_channels, s=1, color='k', alpha=0.5)
    ax_bottom.set_xlabel('Time (sec)')
    ax_bottom.set_ylabel('Channel (derived from cluster)')
    ax_bottom.set_title("Raster Plot of Spikes")
    ax_bottom.set_xlim([0, time_vector[-1]])
    ax_bottom.set_ylim([np.max(spike_channels), np.min(spike_channels)]) # if you want to invert

    plt.tight_layout()
    plt.show()

def plot_vf_spike_raster_filtered_units(von_frey_analysis_instance, trial_name, corr_threshold=0.1, title=None):
    # Step 1: Extract spike data and Von Frey data
    if trial_name not in von_frey_analysis_instance.spikes.kilosort_results:
        print(f"No kilosort results for trial '{trial_name}'.")
        return

    kilosort_output = von_frey_analysis_instance.spikes.kilosort_results[trial_name]
    st = kilosort_output['spike_times']  # spike times in samples
    clu = kilosort_output['spike_clusters']  # cluster assignments
    fs = kilosort_output['ops']['fs']  # Sampling rate
    spike_times_sec = st / fs

    # Retrieve Von Frey data
    if trial_name not in von_frey_analysis_instance.signals.data.analog_data:
        print(f"Trial '{trial_name}' not found in analog_data.")
        return

    recording = von_frey_analysis_instance.signals.data.analog_data[trial_name]
    sampling_rate_vf = recording.get_sampling_frequency()

    if 'ANALOG-IN-2' not in recording.get_channel_ids():
        print(f"ANALOG-IN-2 not found in trial '{trial_name}'.")
        return

    von_frey_data = recording.get_traces(channel_ids=['ANALOG-IN-2'], return_scaled=True).flatten()
    num_samples = len(von_frey_data)
    time_vector = np.arange(num_samples) / sampling_rate_vf
    all_clusters = np.unique(clu)

    # Step 2: Compute correlation and filter clusters
    cluster_correlations = {}
    for cluster in all_clusters:
        cluster_spike_times = np.sort(spike_times_sec[clu == cluster])

        # If fewer than 2 spikes, correlation is set to 0
        if len(cluster_spike_times) < 2:
            cluster_correlations[cluster] = 0
            continue

        # Create "time since last spike" array
        time_since_last_spike = np.zeros(num_samples, dtype=float)
        last_spike_idx = 0
        spike_idx = 0
        for i in range(num_samples):
            current_time = time_vector[i]
            while spike_idx < len(cluster_spike_times) and cluster_spike_times[spike_idx] <= current_time:
                last_spike_idx = spike_idx
                spike_idx += 1
            time_since_last_spike[i] = current_time - cluster_spike_times[last_spike_idx]

        # Compute correlation
        if np.std(time_since_last_spike) == 0 or np.std(von_frey_data) == 0:
            corr = 0
        else:
            corr = np.corrcoef(time_since_last_spike, von_frey_data)[0, 1]
        cluster_correlations[cluster] = corr

    filtered_clusters = [c for c, corr_val in cluster_correlations.items() if abs(corr_val) >= corr_threshold]

    if len(filtered_clusters) == 0:
        print(f"No clusters meet the correlation threshold for trial '{trial_name}'.")
        return

    # Step 3 & 4: Create figure with two subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    ax_top, ax_bottom = axes

    # Plot Von Frey voltage trace (entire trial)
    ax_top.plot(time_vector, von_frey_data, color='blue')
    ax_top.set_ylabel('Voltage (uV)')
    
    if title is not None:
        ax_top.set_title(title)
    else:
        ax_top.set_title(f'{trial_name} - Von Frey Voltage Trace & Filtered Spike Raster')


    ## I need to modify my Von Frey class to store correlated units. that way I don't have duplicate code

    # Plot raster of filtered units' spikes
    # Create a channel or cluster map for ordering (adjust if needed)
    # For simplicity, assume cluster IDs are arbitrary. We will assign each filtered unit
    # a y-position based on its sorted order.
    filtered_clusters_sorted = np.sort(filtered_clusters)
    cluster_to_y = {clu_id: idx for idx, clu_id in enumerate(filtered_clusters_sorted)}

    spike_mask = np.isin(clu, filtered_clusters)
    ax_bottom.scatter(spike_times_sec[spike_mask], 
                      [cluster_to_y[c] for c in clu[spike_mask]], 
                      s=0.5, color='k', alpha=0.5)

    ax_bottom.set_ylabel('Filtered Units')
    ax_bottom.set_xlabel('Time (sec)')
    ax_bottom.set_yticks(range(len(filtered_clusters_sorted)))
    ax_bottom.set_yticklabels(filtered_clusters_sorted)
    ax_bottom.set_xlim([time_vector[0], time_vector[-1]])

    plt.tight_layout()
    plt.show()
