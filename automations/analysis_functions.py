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

    def extract_von_frey_windows(self, TRIAL_NAMES=None, amplitude_threshold=225000, start_buffer=0.01, end_buffer=0.01):
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