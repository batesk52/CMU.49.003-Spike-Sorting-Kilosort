# imports
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import Counter
import pandas as pd
import os

class VonFreyAnalysis:
    def __init__(self, rat_instance, spikeinterface_instance, kilosort_instance):
        """
        Initialize the AnalysisFunctions class with references to the Rat, SpikeInterface_wrapper, and Kilosort_wrapper instances.
        
        Parameters:
        - rat_instance: Instance of the Rat class containing loaded data and metadata.
        - spikeinterface_instance: Instance of the SpikeInterface_wrapper class linked to the same Rat instance.
        - kilosort_instance: Instance of the Kilosort_wrapper class, containing spike sorting data.
        """
        self.rat = rat_instance
        self.signals = spikeinterface_instance
        self.spikes = kilosort_instance
        self.von_frey_time_windows = {} # dictionary of time interfaces where von frey is applied, for each trial
        self.cluster_firing_rates = {} #  dictionary of dataframes containing cluster firing rates during Von Frey intervals, for each trial
        self.windowed_results = {} # dictionary of dataframes containing cluster firing rates during the windowed Von Frey intervals, for each trial

    def extract_von_frey_windows(self, TRIAL_NAMES=None, amplitude_threshold=225000, start_buffer=0.001, end_buffer=0.001): #changed to milliseconds
        """
        Extract time windows where Von Frey stimulus is applied.

        This method can:
        - Process a given list of trial names if TRIAL_NAMES is provided.
        - If TRIAL_NAMES is None, it will attempt to process all trials found in kilosort_results.

        Parameters:
        - TRIAL_NAMES (list or None): List of trial names. If None, all trials in kilosort_results are processed.
        - amplitude_threshold: Threshold value for detecting Von Frey stimulus onset.
        - start_buffer: Time (in seconds) to add after the rising edge.
        - end_buffer: Time (in seconds) to subtract before the falling edge.

        Returns:
        - A dictionary mapping each trial name to its extracted intervals.
        """
        # Determine which trials to process
        if TRIAL_NAMES is not None:
            trial_list = TRIAL_NAMES
        else:
            # If no trial names given, use all trials found in kilosort_results
            if not self.spikes.kilosort_results:
                print("No kilosort results found. Please run Kilosort or load results first.")
                return {}
            trial_list = list(self.spikes.kilosort_results.keys())

        for trial_name in trial_list:
            intervals = self._extract_von_frey_windows_single_trial(trial_name, amplitude_threshold, start_buffer, end_buffer)
            if intervals is not None:
                self.von_frey_time_windows[trial_name] = intervals

        return self.von_frey_time_windows

    def _extract_von_frey_windows_single_trial(self, trial_name, amplitude_threshold=225000, start_buffer=0.01, end_buffer=0.01):
        """
        Internal helper function to process a single trial. Identifies Von Frey windows and plots them.
        Saves the figure as 'VF_window_[trial_name].png' in the 'figures' folder.

        Parameters:
        - trial_name: The name of the trial to process.
        - amplitude_threshold: Threshold value for detecting Von Frey stimulus onset.
        - start_buffer: Time (in seconds) to add after the rising edge.
        - end_buffer: Time (in seconds) to subtract before the falling edge.

        Returns:
        - A dictionary containing start and end times of adjusted windows for this trial, or None if trial not found.
        """
        # Ensure the trial exists in the data
        if trial_name not in self.signals.data.intan_recordings_stream3:
            print(f"Trial '{trial_name}' not found in intan_recordings_stream3.")
            return None

        von_frey_recording = self.rat.analog_data[trial_name]
        sampling_rate = von_frey_recording.get_sampling_frequency()

        # Check for ANALOG-IN-2 channel
        if 'ANALOG-IN-2' not in von_frey_recording.get_channel_ids():
            print(f"ANALOG-IN-2 channel not found in the trial '{trial_name}'.")
            return None

        von_frey_data = von_frey_recording.get_traces(channel_ids=['ANALOG-IN-2'], return_scaled=True).flatten()
        num_samples = len(von_frey_data)
        time_vector = np.arange(num_samples) / sampling_rate

        # Find edges
        rising_edges = np.where((von_frey_data[:-1] < amplitude_threshold) & (von_frey_data[1:] >= amplitude_threshold))[0] + 1
        falling_edges = np.where((von_frey_data[:-1] >= amplitude_threshold) & (von_frey_data[1:] < amplitude_threshold))[0] + 1

        # Adjust for start/end above threshold
        if von_frey_data[0] >= amplitude_threshold:
            rising_edges = np.insert(rising_edges, 0, 0)
        if von_frey_data[-1] >= amplitude_threshold:
            falling_edges = np.append(falling_edges, num_samples - 1)

        min_len = min(len(rising_edges), len(falling_edges))
        rising_edges = rising_edges[:min_len]
        falling_edges = falling_edges[:min_len]

        # Convert indices to times
        start_times = rising_edges / sampling_rate
        end_times = falling_edges / sampling_rate

        # Apply buffers
        adjusted_start_times = start_times + start_buffer
        adjusted_end_times = end_times - end_buffer

        # Validate intervals
        valid_indices = adjusted_start_times < adjusted_end_times
        adjusted_start_times = adjusted_start_times[valid_indices]
        adjusted_end_times = adjusted_end_times[valid_indices]

        # Clip to valid ranges
        adjusted_start_times = np.clip(adjusted_start_times, 0, (num_samples - 1) / sampling_rate)
        adjusted_end_times = np.clip(adjusted_end_times, 0, (num_samples - 1) / sampling_rate)

        # Plot and save
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_vector, von_frey_data, label='Von Frey Data', color='blue')
        for start, end in zip(adjusted_start_times, adjusted_end_times):
            ax.axvspan(start, end, color='orange', alpha=0.3, label='Von Frey Stimulus')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (uV)')
        ax.set_title(f'Von Frey Windows: {trial_name}')

        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        # Save the plot
        figures_dir = Path(self.spikes.SAVE_DIRECTORY) / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        figure_path = figures_dir / f'VF_window_{trial_name}.png'
        plt.tight_layout()
        plt.savefig(figure_path, dpi=300)
        plt.close()

        intervals = {
            'adjusted_start_times': adjusted_start_times,
            'adjusted_end_times': adjusted_end_times
        }

        pd.DataFrame(intervals).to_excel(os.path.join(self.spikes.SAVE_DIRECTORY,"tables", f"{trial_name}_von_frey_time_windows.xlsx"))

        return intervals

    def compute_unit_firing_rates(self, TRIAL_NAMES=None, amplitude_threshold=225000, start_buffer=0.01, end_buffer=0.01):
        """
        Compute the average firing rates of units during detected Von Frey stimulus windows.
        
        If TRIAL_NAMES is provided, computes for those trials only.
        If TRIAL_NAMES is None, computes for all trials found in the kilosort_results.
        
        Parameters:
        - TRIAL_NAMES: list of trial names or None
        
        Returns:
        - A dictionary mapping trial_name to a Pandas DataFrame (firing_rates_df)
          where each row corresponds to one detected interval and each column to a cluster.
        """

        # If no trial names are specified, attempt to process all trials
        if TRIAL_NAMES is None:
            if not self.spikes.kilosort_results:
                print("No kilosort results found. Please run Kilosort or load results first.")
                return {}
            trial_list = list(self.spikes.kilosort_results.keys())
        else:
            trial_list = TRIAL_NAMES

        # Extract Von Frey windows for all required trials
        # This returns a dictionary {trial_name: {'adjusted_start_times':..., 'adjusted_end_times':...}, ...}
        intervals_dict = self.extract_von_frey_windows(TRIAL_NAMES=trial_list,
                                                       amplitude_threshold=amplitude_threshold,
                                                       start_buffer=start_buffer,
                                                       end_buffer=end_buffer)

        for trial_name in trial_list:
            if trial_name not in self.spikes.kilosort_results:
                print(f"Kilosort results not found for trial: {trial_name}. Skipping.")
                continue
            
            if trial_name not in intervals_dict:
                print(f"No Von Frey intervals found for trial: {trial_name}. Skipping.")
                continue

            adjusted_start_times = intervals_dict[trial_name]['adjusted_start_times']
            adjusted_end_times = intervals_dict[trial_name]['adjusted_end_times']

            if len(adjusted_start_times) == 0:
                print(f"No valid intervals after adjustments for trial: {trial_name}. Skipping.")
                continue

            # Load kilosort outputs for this trial
            kilosort_output = self.spikes.kilosort_results[trial_name]
            st = kilosort_output['spike_times']  # Spike times in samples
            clu = kilosort_output['spike_clusters']  # Cluster assignments
            sampling_rate_kilosort = kilosort_output['ops']['fs']
            
            # Convert spike times to seconds
            spike_times_sec = st / sampling_rate_kilosort

            # Initialize a list to store firing rates per interval
            firing_rates_intervals = []

            # Loop over each adjusted time window and compute firing rates
            for start_time, end_time in zip(adjusted_start_times, adjusted_end_times):
                # Find spikes within the time window
                indices_in_window = np.where((spike_times_sec >= start_time) & (spike_times_sec < end_time))[0]
                clusters_in_window = clu[indices_in_window]
                
                # Calculate the duration of the interval
                window_duration = end_time - start_time  # In seconds
                
                # Count the number of spikes per cluster within the window
                cluster_spike_counts = Counter(clusters_in_window)
                
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

            # Fill NaN values with zeros (clusters not firing in an interval)
            firing_rates_df = firing_rates_df.fillna(0)

            # Store the DataFrame in the results dictionary
            self.cluster_firing_rates[trial_name] = firing_rates_df


            firing_rates_df.to_excel(os.path.join(self.spikes.SAVE_DIRECTORY,"tables", f"{trial_name}_cluster_firing_rates.xlsx"))


        return self.cluster_firing_rates

    def subdivide_intervals(self, intervals_dict, subwindow_width):
        """
        Given a dictionary of intervals from `extract_von_frey_windows`, 
        subdivide each interval into smaller windows of `subwindow_width` seconds.

        Parameters:
        - intervals_dict: {trial_name: {'adjusted_start_times': ..., 'adjusted_end_times': ...}}
        - subwindow_width: float, width of each sub-window in seconds

        Returns:
        - A dictionary with the same structure, but now for each trial we have
          'subwindow_start_times' and 'subwindow_end_times' arrays corresponding
          to all smaller windows created from the original intervals.
        """
        subwindow_results = {}

        for trial_name, interval_data in intervals_dict.items():
            start_times = interval_data['adjusted_start_times']
            end_times = interval_data['adjusted_end_times']

            sub_starts = []
            sub_ends = []

            for s, e in zip(start_times, end_times):
                curr = s
                while curr < e:
                    next_w = curr + subwindow_width
                    if next_w > e:
                        next_w = e  # partial sub-window at the end
                    sub_starts.append(curr)
                    sub_ends.append(next_w)
                    curr = next_w

            subwindow_results[trial_name] = {
                'subwindow_start_times': np.array(sub_starts),
                'subwindow_end_times': np.array(sub_ends)
            }

        return subwindow_results

    def compute_average_von_frey_voltage(self, trial_name, subwindow_start_times, subwindow_end_times):
        """
        Compute the average Von Frey voltage for each sub-window for a given trial.

        Parameters:
        - trial_name: str, name of the trial
        - subwindow_start_times: np.array of start times (sec)
        - subwindow_end_times: np.array of end times (sec)

        Returns:
        - A Pandas DataFrame with one row per sub-window and a column 'avg_voltage'.
        """
        # Access the analog_data recording for the trial
        if trial_name not in self.signals.data.analog_data:
            print(f"Trial '{trial_name}' not found in analog_data.")
            return pd.DataFrame()

        recording = self.signals.data.analog_data[trial_name]
        sampling_rate = recording.get_sampling_frequency()

        # Check if ANALOG-IN-2 channel exists
        if 'ANALOG-IN-2' not in recording.get_channel_ids():
            print(f"ANALOG-IN-2 not found in trial '{trial_name}'.")
            return pd.DataFrame()

        # Extract full trace
        von_frey_data = recording.get_traces(channel_ids=['ANALOG-IN-2'], return_scaled=True).flatten()

        # Compute average voltage for each sub-window
        avg_voltages = []
        for start, end in zip(subwindow_start_times, subwindow_end_times):
            start_idx = int(start * sampling_rate)
            end_idx = int(end * sampling_rate)
            segment = von_frey_data[start_idx:end_idx]
            avg_voltages.append(np.mean(segment) if len(segment) > 0 else np.nan)

        return pd.DataFrame({'avg_voltage': avg_voltages})

    def compute_unit_firing_rates_for_subwindows(self, trial_name, subwindow_start_times, subwindow_end_times, corr_threshold=0.1):
        """
        Compute firing rates for each unit during the given sub-windows for a single trial,
        and filter clusters based on correlation with the Von Frey signal.

        Instead of binning spikes, we create a continuous time-series per cluster that encodes
        the time since the last spike at each sample. This gives a continuous measure that we
        can correlate directly with von_frey_data.

        Parameters:
        - trial_name: str, name of the trial
        - subwindow_start_times: array of start times for each sub-window
        - subwindow_end_times: array of end times for each sub-window
        - corr_threshold: float, absolute correlation threshold below which clusters are excluded

        Returns:
        - A DataFrame where each row is a sub-window and each column is a cluster,
        containing the firing rate (Hz), after filtering by correlation.
        """
        if trial_name not in self.spikes.kilosort_results:
            print(f"No kilosort results for trial '{trial_name}'.")
            return pd.DataFrame()

        kilosort_output = self.spikes.kilosort_results[trial_name]
        st = kilosort_output['spike_times']  # spike times in samples
        clu = kilosort_output['spike_clusters']  # cluster assignments
        fs = kilosort_output['ops']['fs']  # Sampling rate used during spike sorting

        spike_times_sec = st / fs
        all_clusters = np.unique(clu)

        # Retrieve Von Frey data
        if trial_name not in self.signals.data.analog_data:
            print(f"Trial '{trial_name}' not found in analog_data.")
            return pd.DataFrame()

        recording = self.signals.data.analog_data[trial_name]
        sampling_rate_vf = recording.get_sampling_frequency()
        if 'ANALOG-IN-2' not in recording.get_channel_ids():
            print(f"ANALOG-IN-2 not found in trial '{trial_name}'.")
            return pd.DataFrame()

        von_frey_data = recording.get_traces(channel_ids=['ANALOG-IN-2'], return_scaled=True).flatten()
        num_samples = len(von_frey_data)
        total_duration = num_samples / sampling_rate_vf

        # Create a time vector for von_frey_data
        time_vector = np.arange(num_samples) / sampling_rate_vf

        # Compute "time since last spike" arrays for each cluster
        cluster_correlations = {}
        for cluster in all_clusters:
            # Get spike times for this cluster
            cluster_spike_times = np.sort(spike_times_sec[clu == cluster])

            # If no spikes or just one spike, correlation likely meaningless
            if len(cluster_spike_times) < 2:
                cluster_correlations[cluster] = 0
                continue

            # Create an array to store time since last spike at each sample
            time_since_last_spike = np.zeros(num_samples, dtype=float)

            # We will walk through the spike times and fill in intervals
            last_spike_idx = 0
            spike_idx = 0

            for i in range(num_samples):
                current_time = time_vector[i]

                # Move spike_idx forward if the next spike is in the past
                while spike_idx < len(cluster_spike_times) and cluster_spike_times[spike_idx] <= current_time:
                    last_spike_idx = spike_idx
                    spike_idx += 1

                # time since last spike = current_time - cluster_spike_times[last_spike_idx]
                time_since_last_spike[i] = current_time - cluster_spike_times[last_spike_idx]

            # Compute correlation with von_frey_data
            if np.std(time_since_last_spike) == 0 or np.std(von_frey_data) == 0:
                corr = 0
            else:
                corr = np.corrcoef(time_since_last_spike, von_frey_data)[0, 1]

            cluster_correlations[cluster] = corr

        # Filter clusters based on correlation threshold
        filtered_clusters = [c for c, corr in cluster_correlations.items() if abs(corr) >= corr_threshold]

        if len(filtered_clusters) == 0:
            print(f"All clusters excluded for trial '{trial_name}' based on correlation threshold.")
            return pd.DataFrame()

        # Now compute firing rates for the filtered clusters within each sub-window
        firing_rates_intervals = []
        for start, end in zip(subwindow_start_times, subwindow_end_times):
            window_duration = end - start
            indices_in_window = np.where((spike_times_sec >= start) & (spike_times_sec < end))[0]
            clusters_in_window = clu[indices_in_window]

            cluster_spike_counts = Counter(clusters_in_window)

            firing_rates = {}
            for cluster in filtered_clusters:
                count = cluster_spike_counts.get(cluster, 0)
                firing_rates[cluster] = count / window_duration if window_duration > 0 else np.nan

            firing_rates_intervals.append(firing_rates)

        firing_rates_df = pd.DataFrame(firing_rates_intervals).fillna(0)
        return firing_rates_df

    def analyze_subwindows(self, TRIAL_NAMES=None, amplitude_threshold=225000, start_buffer=0.001, end_buffer=0.001, subwindow_width=0.5, corr_threshold=0.01):
        """
        Example higher-level method that:
        1. Extracts Von Frey windows.
        2. Subdivides into sub-windows.
        3. Computes average voltage for each sub-window.
        4. Computes unit firing rates for each sub-window.
        5. Classifies sub-windows into 'pre-stim' (first 35s) and 'post-stim' (last 35s).

        Parameters:
        - TRIAL_NAMES: list or None
        - amplitude_threshold, start_buffer, end_buffer: parameters for the initial window extraction
        - subwindow_width: width of each smaller window in seconds

        Returns:
        - A dictionary keyed by trial name. Each value is another dictionary with:
          {
            'avg_voltage_df': DataFrame of avg voltages per sub-window,
            'firing_rates_df': DataFrame of firing rates per sub-window
          }
        """
        # Extract large intervals
        intervals_dict = self.extract_von_frey_windows(TRIAL_NAMES=TRIAL_NAMES,
                                                       amplitude_threshold=amplitude_threshold,
                                                       start_buffer=start_buffer,
                                                       end_buffer=end_buffer)
        if intervals_dict is None:
            print("No intervals extracted.")
            return {}

        # Subdivide intervals
        subwindows_dict = self.subdivide_intervals(intervals_dict, subwindow_width=subwindow_width)

        for trial_name in subwindows_dict:
            subwindow_starts = subwindows_dict[trial_name]['subwindow_start_times']
            subwindow_ends = subwindows_dict[trial_name]['subwindow_end_times']

            # Compute average voltage
            avg_voltage_df = self.compute_average_von_frey_voltage(trial_name, subwindow_starts, subwindow_ends)

            # Compute firing rates
            firing_rates_df = self.compute_unit_firing_rates_for_subwindows(trial_name, subwindow_starts, subwindow_ends,corr_threshold=corr_threshold)

            # Classify windows into 'pre-stim' or 'post-stim' based on start time
            # Assuming total duration ~70 seconds, first 35s = pre-stim, last 35s = post-stim
            groups = ["pre-stim" if start < 35 else "post-stim" for start in subwindow_starts]

            # Add a 'group' column to both DataFrames
            avg_voltage_df['group'] = groups
            firing_rates_df['group'] = groups

            # Save to Excel
            avg_voltage_df.to_excel(os.path.join(self.spikes.SAVE_DIRECTORY,"tables", f"{trial_name}_average_vf_voltage_windowed.xlsx"))
            firing_rates_df.to_excel(os.path.join(self.spikes.SAVE_DIRECTORY,"tables", f"{trial_name}_cluster_firing_rates_windowed.xlsx"))

            self.windowed_results[trial_name] = {
                'avg_voltage_df': avg_voltage_df,
                'firing_rates_df': firing_rates_df
            }