# plots.py

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd

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

        # Optionally, annotate points with cluster IDs
        for cluster_id, row in ratio_summary.iterrows():
            if pd.notna(row['pre-stim']) and pd.notna(row['post-stim']):
                plt.text(row['pre-stim'], row['post-stim'], str(cluster_id), fontsize=8, alpha=0.7)

        plt.xlabel('Pre-Stim Avg Ratio (firing_rate/voltage)')
        plt.ylabel('Post-Stim Avg Ratio (firing_rate/voltage)')
        # Include the trial name and frequency in the title
        plt.title(f'{trial_name} | Stim Freq: {freq_hz} Hz')

        # plt.grid(True, linestyle='--', alpha=0.5)
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
    - Adds a linear trendline for each frequency group.
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

        # Compute trendline for this frequency group
        if len(freq_points) > 1:
            # Perform linear regression using np.polyfit
            slope, intercept = np.polyfit(freq_points['pre_stim'], freq_points['post_stim'], 1)
            xvals = np.linspace(freq_points['pre_stim'].min(), freq_points['pre_stim'].max(), 100)
            yvals = intercept + slope * xvals
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
    # commenting out the save because I don't want to bother with making a smart functionion for saving directory
    # plt.savefig("all_trials_ratio_scatter_with_trendlines.png", dpi=300)
    plt.show()

