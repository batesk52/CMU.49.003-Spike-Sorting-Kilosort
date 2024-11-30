import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kilosort import io
from kilosort import run_kilosort, DEFAULT_SETTINGS
from .RM1 import SpikeInterface_wrapper, Kilosort_wrapper, Rat

def analyze_rat_trials(rat_id, trial_names, data_directory, save_directory, probe_directory):
    # Instantiate the Rat class
    rat = Rat(data_directory, probe_directory, rat_id)
    rat.get_sc_data()
    
    # Initialize the SpikeInterface and Kilosort wrappers
    signals = SpikeInterface_wrapper(rat, save_directory)
    spikes = Kilosort_wrapper(save_directory, probe_directory)
    
    # Loop over the specified trials
    for trial_name in trial_names:
        # Perform your analysis for each trial
        analyze_trial(rat, signals, spikes, trial_name)



def analyze_rat_trials(rat_id, trial_names, data_directory, save_directory, probe_directory):
    # Instantiate the Rat class
    rat = Rat(data_directory, probe_directory, rat_id)
    rat.get_sc_data()
    
    # Initialize the SpikeInterface and Kilosort wrappers
    signals = SpikeInterface_wrapper(rat, save_directory)
    spikes = Kilosort_wrapper(save_directory, probe_directory)
    
    # Loop over the specified trials
    for trial_name in trial_names:
        # Perform your analysis for each trial
        analyze_trial(rat, signals, spikes, trial_name)

def analyze_trial(rat, signals, spikes, trial_name):
    # Access recordings
    try:
        recording = signals.data.intan_recordings_stream0[trial_name]
        digital_trace = signals.data.intan_recordings_stream4[trial_name].get_traces(channel_ids=['DIGITAL-IN-01'])
        von_frey = signals.data.intan_recordings_stream3[trial_name]
    except KeyError:
        print(f"Trial {trial_name} data not found.")
        return

    sampling_rate = recording.get_sampling_frequency()

    # Extract stimulation times
    stimulation_times = extract_stimulation_times(digital_trace, sampling_rate)

    # Define the time window for STA
    pre_time = -0.001  # Time before the stimulation (negative value)
    post_time = 0.005  # Time after the stimulation

    # Perform STA
    average_ecap = perform_sta(recording, stimulation_times, pre_time, post_time)

    # Plot average ECAP
    channel_ids = recording.get_channel_ids()
    plot_average_ecap(average_ecap, sampling_rate, pre_time, post_time, channel_ids)

    # Define intervals for firing rate computation (e.g., stimulus periods)
    # For this example, we'll use the stimulation times with buffers
    start_buffer = 0.1  # Time to add to the start of each interval
    end_buffer = 0.05   # Time to subtract from the end of each interval

    adjusted_start_times = stimulation_times + start_buffer
    adjusted_end_times = stimulation_times + end_buffer  # Assuming intervals are short; adjust as needed

    # Ensure that the adjusted times are within valid bounds
    valid_indices = adjusted_start_times < adjusted_end_times
    adjusted_start_times = adjusted_start_times[valid_indices]
    adjusted_end_times = adjusted_end_times[valid_indices]

    # Extract Kilosort outputs or run Kilosort if necessary
    spikes.extract_kilosort_outputs()

    # Analyze Kilosort results
    analyze_kilosort_results(spikes, trial_name, stimulation_times, adjusted_start_times, adjusted_end_times)


def extract_stimulation_times(digital_trace, sampling_rate, threshold=0.5):
    """
    Extract stimulation times from the digital trace.

    Parameters:
        digital_trace (numpy.ndarray): The digital input signal.
        sampling_rate (float): Sampling rate of the recording.
        threshold (float): Threshold to binarize the digital signal.

    Returns:
        numpy.ndarray: Array of stimulation times in seconds.
    """
    # Flatten the digital trace to a 1D array
    digital_trace = digital_trace.flatten()

    # Threshold the digital trace to ensure it's binary (0 or 1)
    digital_trace_binary = (digital_trace > threshold).astype(int)

    # Find indices where the digital trace goes from low to high (rising edge)
    rising_edges = np.where((digital_trace_binary[:-1] == 0) & (digital_trace_binary[1:] == 1))[0] + 1  # +1 to correct index

    # Convert indices to times
    stimulation_times = rising_edges / sampling_rate  # Times in seconds

    return stimulation_times


def perform_sta(recording, stimulation_times, pre_time, post_time):
    """
    Perform Spike-Triggered Averaging (STA) around stimulation times.

    Parameters:
        recording (RecordingExtractor): The recording object from SpikeInterface.
        stimulation_times (numpy.ndarray): Array of stimulation times in seconds.
        pre_time (float): Time before the stimulation to include in the window (negative value).
        post_time (float): Time after the stimulation to include in the window.

    Returns:
        numpy.ndarray: Average ECAP across all stimulations (num_channels x window_samples).
    """
    sampling_rate = recording.get_sampling_frequency()
    window_samples = int((post_time - pre_time) * sampling_rate)
    num_stims = len(stimulation_times)
    channel_ids = recording.get_channel_ids()
    num_channels = len(channel_ids)
    windows = np.zeros((num_stims, num_channels, window_samples))

    # Loop over each stimulation time and extract the window
    for i, stim_time in enumerate(stimulation_times):
        # Convert time to sample index
        stim_sample = int(stim_time * sampling_rate)
        # Define window indices
        start_sample = stim_sample + int(pre_time * sampling_rate)
        end_sample = stim_sample + int(post_time * sampling_rate)
        # Handle boundary conditions
        if start_sample < 0 or end_sample > recording.get_num_frames():
            continue  # Skip if window is out of bounds
        # Extract the window for all channels
        window = recording.get_traces(
            start_frame=start_sample,
            end_frame=end_sample
        )
        windows[i] = window.T  # Transpose to match dimensions

    # Compute the average across all stimulation windows
    average_ecap = np.mean(windows, axis=0)  # Shape: (num_channels, window_samples)
    return average_ecap

def plot_average_ecap(average_ecap, sampling_rate, pre_time, post_time, channel_ids):
    """
    Plot the average ECAP across channels.

    Parameters:
        average_ecap (numpy.ndarray): Average ECAP data (num_channels x window_samples).
        sampling_rate (float): Sampling rate of the recording.
        pre_time (float): Time before the stimulation included in the window (negative value).
        post_time (float): Time after the stimulation included in the window.
        channel_ids (list): List of channel IDs corresponding to the data.
    """
    window_samples = average_ecap.shape[1]
    time_axis = np.linspace(pre_time, post_time, window_samples)  # Time axis for plotting

    plt.figure(figsize=(12, 6))
    num_channels = average_ecap.shape[0]
    # Plot each channel's average ECAP
    for i in range(num_channels):
        plt.plot(time_axis * 1000, average_ecap[i] + i * 50, label=f'Channel {channel_ids[i]}')  # Offset for visualization
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (µV) + Offset')
    plt.title('Average ECAP Across Channels')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), ncol=1, fontsize='small', frameon=False)
    plt.tight_layout()
    plt.show()

def analyze_kilosort_results(spikes, trial_name, stimulation_times, adjusted_start_times, adjusted_end_times):
    """
    Analyze Kilosort results to compute firing rates during intervals.

    Parameters:
        spikes (Kilosort_wrapper): The Kilosort wrapper object.
        trial_name (str): Name of the trial to analyze.
        stimulation_times (numpy.ndarray): Array of stimulation times in seconds.
        adjusted_start_times (numpy.ndarray): Start times of intervals (e.g., stimulus periods).
        adjusted_end_times (numpy.ndarray): End times of intervals.
    """
    # Load Kilosort outputs for the trial
    if trial_name in spikes.kilosort_results:
        kilosort_output = spikes.kilosort_results[trial_name]
        # Compute firing rates during the intervals
        firing_rates_df = compute_firing_rates(kilosort_output, adjusted_start_times, adjusted_end_times)
        # Plot firing rates
        plot_firing_rates(firing_rates_df)
    else:
        print(f"Kilosort results not found for trial {trial_name}.")


def compute_firing_rates(kilosort_output, start_times, end_times):
    """
    Compute firing rates for each cluster during specified intervals.

    Parameters:
        kilosort_output (dict): Kilosort output data for a trial.
        start_times (numpy.ndarray): Array of interval start times in seconds.
        end_times (numpy.ndarray): Array of interval end times in seconds.

    Returns:
        pandas.DataFrame: DataFrame containing firing rates for each cluster across intervals.
    """
    from collections import Counter

    st = kilosort_output['spike_times']  # Spike times in samples
    clu = kilosort_output['spike_clusters']  # Cluster assignments
    sampling_rate = kilosort_output['ops']['fs']  # Sampling rate used in Kilosort

    # Convert spike times to seconds
    spike_times_sec = st.flatten() / sampling_rate

    # Initialize a list to store firing rates per interval
    firing_rates_intervals = []

    # Loop over each interval
    for start_time, end_time in zip(start_times, end_times):
        # Find spikes within the time window
        indices_in_window = np.where((spike_times_sec >= start_time) & (spike_times_sec < end_time))[0]
        clusters_in_window = clu[indices_in_window]

        # Calculate the duration of the interval
        window_duration = end_time - start_time  # In seconds

        # Count the number of spikes per cluster within the window
        cluster_spike_counts = Counter(clusters_in_window.flatten())

        # Get all clusters present in the entire recording
        all_clusters = np.unique(clu)

        # Initialize firing rates with zero for all clusters
        firing_rates = {cluster: 0 for cluster in all_clusters}

        # Compute firing rates for clusters present in the window
        for cluster in all_clusters:
            count = cluster_spike_counts.get(cluster, 0)
            firing_rates[cluster] = count / window_duration  # Firing rate in Hz

        # Append the firing rates dictionary to the list
        firing_rates_intervals.append(firing_rates)

    # Convert the list of dictionaries to a DataFrame
    # Each row corresponds to an interval, columns are clusters
    firing_rates_df = pd.DataFrame(firing_rates_intervals)

    # Optionally, fill NaN values with zeros (clusters not firing in an interval)
    firing_rates_df = firing_rates_df.fillna(0)

    return firing_rates_df



def plot_firing_rates(firing_rates_df):
    """
    Plot the firing rates of clusters across intervals.

    Parameters:
        firing_rates_df (pandas.DataFrame): DataFrame containing firing rates.
    """
    # Calculate the standard deviation for each cluster across intervals
    std_firing_rates = firing_rates_df.std(axis=0)

    # Determine a threshold for "significant change" in firing rate
    # For example, clusters with a standard deviation greater than the median are highlighted
    threshold = std_firing_rates.median()

    # Identify units that show significant changes versus stable units
    significant_units = std_firing_rates > threshold
    stable_units = ~significant_units

    # Plotting
    plt.figure(figsize=(18, 6))

    # Plot the stable units with low opacity and thinner lines
    for cluster in firing_rates_df.columns[stable_units]:
        plt.plot(firing_rates_df[cluster], marker='o', linestyle='-', linewidth=1, alpha=0.5, color='gray')

    # Plot the significant units with higher opacity and thicker lines
    for cluster in firing_rates_df.columns[significant_units]:
        plt.plot(firing_rates_df[cluster], marker='o', linestyle='-', linewidth=2, alpha=0.9, label=f'Cluster {cluster}')

    # Highlighting
    plt.xlabel('Interval Number', fontsize=14)
    plt.ylabel('Firing Rate (Hz)', fontsize=14)
    plt.title('Firing Rates of Clusters Across Intervals', fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Optional: Add a legend for clusters with significant changes outside the plot
    plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize='small', frameon=False)

    # Adjust layout to accommodate the legend outside the plot
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the plot area to make space for the legend

    plt.show()




