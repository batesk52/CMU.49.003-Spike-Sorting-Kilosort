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
        self.cluster_firing_rates = {}  # dictionary of dataframes containing cluster firing rates during Von Frey intervals, for each trial
        self.windowed_results = {}      # dictionary of dataframes containing cluster firing rates during the windowed Von Frey intervals, for each trial
        self.inv_isi_correlations = {}  # {trial_name: {cluster_id: correlation}}
        self.inv_isi_traces = {}        # {trial_name: {cluster_id: np.ndarray}}

    # used in compute_inv_isi_correlation()
    def _smooth_moving_average(self, signal, window_size=15000):
        """
        Smooth 'signal' by convolving with a ones kernel of size 'window_size'.
        window_size=30000 ~ 1 second at 30 kHz, for example.
        """
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(signal, kernel, mode='same')
        return smoothed

    # used in analyze_subwindows()
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

    # used in analyze_subwindows()
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

        # new part: create a raster plot for each trial & save
        if trial_name in self.spikes.kilosort_results:
            ks_data = self.spikes.kilosort_results[trial_name]
            fs = ks_data['ops']['fs']
            spike_times_sec = ks_data['spike_times'] / fs
            clusters = ks_data['spike_clusters']
            
            unique_clusters = np.unique(clusters)
            cluster_order = {c: i for i, c in enumerate(unique_clusters)}
            
            fig, ax = plt.subplots(figsize=(10, 5))
            # Plot spikes for each cluster as tick marks
            for c in unique_clusters:
                idx = clusters == c
                ax.scatter(spike_times_sec[idx], np.full(np.sum(idx), cluster_order[c]),
                        marker='|', color='black', s=100)
            # Overlay the stimulus windows
            for i, (start, end) in enumerate(zip(adjusted_start_times, adjusted_end_times)):
                ax.axvspan(start, end, color='orange', alpha=0.3,
                        label='Von Frey Stimulus' if i == 0 else None)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Cluster')
            ax.set_yticks(list(cluster_order.values()))
            ax.set_yticklabels(list(cluster_order.keys()))
            ax.set_title(f'Raster Plot: {trial_name}')
            
            # Remove duplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys())
            
            # Save the figure
            figures_dir = Path(self.spikes.SAVE_DIRECTORY) / 'figures'
            figures_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(figures_dir / f'Raster_{trial_name}.png', dpi=300)
            plt.close(fig)


        # Plot the von frey trace and save
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
    
    # !!(can I delete this?)!! this method is (supposed to be) used in analyze_subwindows
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

    # used in analyze_subwindows()
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

    # used in analyze_subwindows()
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

    # used in analyze_subwindows()
    def compute_unit_firing_rates_for_subwindows(self, trial_name, subwindow_start_times, subwindow_end_times):
        """
        Compute firing rates for every cluster during the given sub-windows for a single trial.
        This version DOES NOT filter clusters by correlation.

        Parameters:
        - trial_name: str, name of the trial
        - subwindow_start_times: array of start times for each sub-window
        - subwindow_end_times: array of end times for each sub-window

        Returns:
        - A DataFrame where each row is a sub-window and each column is a cluster,
          containing the firing rate (Hz).
        """

        # Retrieve kilosort results
        if trial_name not in self.spikes.kilosort_results:
            print(f"No kilosort results for trial '{trial_name}'.")
            return pd.DataFrame()

        kilosort_output = self.spikes.kilosort_results[trial_name]
        st = kilosort_output['spike_times']  # spike times in samples
        clu = kilosort_output['spike_clusters']  # cluster assignments
        fs = kilosort_output['ops']['fs']  # Sampling rate used during spike sorting

        spike_times_sec = st / fs
        all_clusters = np.unique(clu)

        # Now compute firing rates for all clusters within each sub-window
        firing_rates_intervals = []
        for start, end in zip(subwindow_start_times, subwindow_end_times):
            window_duration = end - start
            indices_in_window = np.where((spike_times_sec >= start) & (spike_times_sec < end))[0]
            clusters_in_window = clu[indices_in_window]

            cluster_spike_counts = Counter(clusters_in_window)

            firing_rates = {}
            for cluster in all_clusters:
                count = cluster_spike_counts.get(cluster, 0)
                firing_rates[cluster] = count / window_duration if window_duration > 0 else np.nan

            firing_rates_intervals.append(firing_rates)

        firing_rates_df = pd.DataFrame(firing_rates_intervals).fillna(0)
        return firing_rates_df
    
    # used in analyze_subwindows()
    # def correlate_von_frey_to_cluster_firing(self, trial_name):
    #     """
    #     Compute correlation with the Von Frey signal for each cluster by creating
    #     a 'time since last spike' trace for that cluster and correlating it with
    #     the entire Von Frey trace.

    #     Parameters:
    #     - trial_name: str, name of the trial

    #     Returns:
    #     - A dictionary mapping cluster ID to correlation coefficient with the Von Frey data.
    #     """

    #     # Retrieve kilosort results
    #     if trial_name not in self.spikes.kilosort_results:
    #         print(f"No kilosort results for trial '{trial_name}'.")
    #         return {}

    #     kilosort_output = self.spikes.kilosort_results[trial_name]
    #     st = kilosort_output['spike_times']  # spike times in samples
    #     clu = kilosort_output['spike_clusters']  # cluster assignments
    #     fs = kilosort_output['ops']['fs']  # Sampling rate used during spike sorting

    #     spike_times_sec = st / fs
    #     all_clusters = np.unique(clu)

    #     # Retrieve Von Frey data
    #     if trial_name not in self.signals.data.analog_data:
    #         print(f"Trial '{trial_name}' not found in analog_data.")
    #         return {}

    #     recording = self.signals.data.analog_data[trial_name]
    #     sampling_rate_vf = recording.get_sampling_frequency()
    #     if 'ANALOG-IN-2' not in recording.get_channel_ids():
    #         print(f"ANALOG-IN-2 not found in trial '{trial_name}'.")
    #         return {}

    #     von_frey_data = recording.get_traces(channel_ids=['ANALOG-IN-2'], return_scaled=True).flatten()
    #     num_samples = len(von_frey_data)
    #     time_vector = np.arange(num_samples) / sampling_rate_vf

    #     cluster_correlations = {}
    #     for cluster in all_clusters:
    #         # Get spike times for this cluster
    #         cluster_spike_times = np.sort(spike_times_sec[clu == cluster])

    #         # If no spikes or just one spike, correlation can be treated as zero or NaN
    #         if len(cluster_spike_times) < 2:
    #             cluster_correlations[cluster] = 0.0
    #             continue

    #         # Create an array to store time since last spike at each sample
    #         time_since_last_spike = np.zeros(num_samples, dtype=float)
    #         last_spike_idx = 0
    #         spike_idx = 0

    #         for i in range(num_samples):
    #             current_time = time_vector[i]
    #             # Move spike_idx forward if the next spike is in the past
    #             while spike_idx < len(cluster_spike_times) and cluster_spike_times[spike_idx] <= current_time:
    #                 last_spike_idx = spike_idx
    #                 spike_idx += 1
    #             # time since last spike
    #             time_since_last_spike[i] = current_time - cluster_spike_times[last_spike_idx]

    #         # Compute correlation with von_frey_data
    #         if np.std(time_since_last_spike) == 0 or np.std(von_frey_data) == 0:
    #             corr = 0.0
    #         else:
    #             corr = np.corrcoef(time_since_last_spike, von_frey_data)[0, 1]

    #         cluster_correlations[cluster] = corr

    #     return cluster_correlations

    # used in analyze_subwindows()
    def compute_inv_isi_correlation(self, trial_name, window_size=15000):
        """
        1) Loads spike times/clusters for the given trial.
        2) Loads the Von Frey data for channel 'ANALOG-IN-2'.
        3) Builds a piecewise constant inverse-ISI trace per cluster.
        4) Smooths it (moving average).
        5) Correlates with Von Frey data.
        6) Stores results in self.inv_isi_correlations / self.inv_isi_traces.

        Returns
        -------
        correlations : dict
            {cluster_id: correlation_coefficient}
        inv_isi_traces : dict
            {cluster_id: np.ndarray of shape (N,)} the smoothed inverse-ISI trace
        """
        if trial_name not in self.spikes.kilosort_results:
            print(f"[WARNING] Kilosort results not found for trial: {trial_name}")
            return {}, {}

        # 1) Retrieve spike data
        kilosort_output = self.spikes.kilosort_results[trial_name]
        st = kilosort_output["spike_times"]      # shape (num_spikes,)
        clu = kilosort_output["spike_clusters"]  # shape (num_spikes,)
        fs  = kilosort_output["ops"]["fs"]       # e.g. 30000

        # 2) Retrieve Von Frey voltage data
        recording = self.signals.data.intan_recordings_stream3.get(trial_name, None)
        if recording is None:
            print(f"[WARNING] No recording found for trial: {trial_name}")
            return {}, {}

        # Extract the entire voltage array from 'ANALOG-IN-2'
        if "ANALOG-IN-2" not in recording.get_channel_ids():
            print(f"[WARNING] Channel 'ANALOG-IN-2' not found for trial: {trial_name}")
            return {}, {}

        # import von frey data and make it the same length as the spikes (first and last 35 seconds or 1,050,000 samples)
        von_frey_data = recording.get_traces(channel_ids=["ANALOG-IN-2"], return_scaled=True).flatten()
        vf_initial = von_frey_data[:1050000]
        vf_end = von_frey_data[-1050000:]
        von_frey_data = np.concatenate([vf_initial, vf_end])
        N = len(von_frey_data)

        max_spike_time = st.max()
        if max_spike_time >= N:
            print("[WARNING] Some spikes occur beyond the length of the Von Frey data. "
                  f"max spike time = {max_spike_time}, length of VF data = {N}.")

        unique_clusters = np.unique(clu)
        correlations = {}
        inv_isi_traces = {}

        # 3) Build inverse-ISI for each cluster
        for cluster_id in unique_clusters:
            cluster_spike_times = st[clu == cluster_id]
            cluster_spike_times.sort()

            inv_isi_trace = np.zeros(N, dtype=float)

            if len(cluster_spike_times) < 2:
                correlations[cluster_id] = 0.0
                inv_isi_traces[cluster_id] = inv_isi_trace
                continue

            # Fill in segments between successive spikes
            for i in range(len(cluster_spike_times) - 1):
                s_k = cluster_spike_times[i]
                s_next = cluster_spike_times[i+1]

                if s_k >= N:
                    break
                if s_next > N:
                    s_next = N

                isi = s_next - s_k
                if isi == 0:
                    isi = 1  # avoid division by zero
                inv_isi_value = 1.0 / isi
                inv_isi_trace[s_k : s_next] = inv_isi_value

            # 4) Smooth the inverse-ISI trace
            inv_isi_trace_smoothed = self._smooth_moving_average(inv_isi_trace, window_size=window_size)

            # 5) Correlate with Von Frey
            std_invisi = np.std(inv_isi_trace_smoothed)
            std_vf = np.std(von_frey_data)
            if std_invisi == 0 or std_vf == 0:
                corr_val = 0.0
            else:
                corr_val = np.corrcoef(inv_isi_trace_smoothed, von_frey_data)[0, 1]

            correlations[cluster_id] = corr_val
            inv_isi_traces[cluster_id] = inv_isi_trace_smoothed

        # 6) Store results in self.* so you can reference later
        self.inv_isi_correlations[trial_name] = correlations
        self.inv_isi_traces[trial_name] = inv_isi_traces

        return correlations, inv_isi_traces

    # this is the main "wrapper" method used to analyze the data
    def analyze_subwindows(self, TRIAL_NAMES=None, amplitude_threshold=225000, start_buffer=0.001, end_buffer=0.001, 
                        subwindow_width=0.5, corr_threshold=0.01):
        """
        Higher-level method that:
        1. Extracts Von Frey windows.
        2. Subdivides into sub-windows.
        3. Computes average voltage for each sub-window.
        4. Computes unit firing rates for each sub-window.
        5. Classifies sub-windows into 'pre-stim' (first 35s) and 'post-stim' (last 35s).
        6. calculates the Pearson correlation coefficient between von frey and inverse spike intervals (ISI)
        7. saves cluster firing rate, inverse ISI, and von frey data for each sub-window, and classification of "pre-stim" or "post-stim" to excel
        """
        # If no trial names are provided, use all
        if TRIAL_NAMES is None:
            # Grab all trials
            TRIAL_NAMES = list(self.spikes.kilosort_results.keys())

        # 1) Extract intervals where Von Frey stimulus is applied
        intervals_dict = self.extract_von_frey_windows(TRIAL_NAMES=TRIAL_NAMES,
                                                    amplitude_threshold=amplitude_threshold,
                                                    start_buffer=start_buffer,
                                                    end_buffer=end_buffer)
        if intervals_dict is None:
            print("No intervals extracted.")
            return {}

        # 2) Subdivide intervals into smaller windows
        subwindows_dict = self.subdivide_intervals(intervals_dict, subwindow_width=subwindow_width)

        for trial_name in subwindows_dict:
            subwindow_starts = subwindows_dict[trial_name]['subwindow_start_times']
            subwindow_ends   = subwindows_dict[trial_name]['subwindow_end_times']

            # 3) Compute average voltage
            avg_voltage_df = self.compute_average_von_frey_voltage(trial_name, 
                                                                subwindow_starts, 
                                                                subwindow_ends)

            # 4) Compute firing rates for each sub-window (for ALL clusters, no filtering)
            firing_rates_df = self.compute_unit_firing_rates_for_subwindows(
                trial_name, 
                subwindow_starts, 
                subwindow_ends
            )

            # Classify windows into 'pre-stim' or 'post-stim' based on start time
            #    (assuming total duration ~70 seconds, first 35s = pre-stim, last 35s = post-stim)
            groups = ["pre-stim" if start < 35 else "post-stim" for start in subwindow_starts]
            # Add a 'group' column to both DataFrames
            avg_voltage_df['group']     = groups
            firing_rates_df['group']    = groups

            # -------------------------------------------------------------------
            # NEW: IMPROVED CORRELATION STEP
            # -------------------------------------------------------------------

            # 5) Compute correlations for each cluster across the *entire* trial
            #    (Here, you can call your new "compute_inv_isi_correlation", or the older
            #     "correlate_von_frey_to_cluster_firing"—whichever method returns {cluster: corr_val}.)
            correlations, _ = self.compute_inv_isi_correlation(trial_name, 
                                                                window_size=15000)
            # correlations -> {cluster_id: correlation_coefficient}
           
            # 6) Convert that dictionary into two DataFrame rows:
            #    row 1: correlation
            #    row 2: is_correlated (True/False based on threshold)
            if correlations:  # make sure dict is not empty
                corr_row = pd.DataFrame(
                    {cluster: [corr_val] 
                    for cluster, corr_val in correlations.items()},
                    index=['correlation']   # Name of the new row
                )
                is_corr_row = pd.DataFrame(
                    {cluster: [abs(corr_val) >= corr_threshold] 
                    for cluster, corr_val in correlations.items()},
                    index=['is_correlated'] # Name of the new row
                )

                # Append them to the *bottom* of firing_rates_df
                # By default, firing_rates_df has shape (n_subwindows, n_clusters) plus 1 'group' column.
                # After concatenation, we'll have (n_subwindows + 2) rows, and the same cluster columns + 'group'.
                firing_rates_df = pd.concat([firing_rates_df, corr_row, is_corr_row], axis=0)
            else:
                print(f"No correlations computed for trial '{trial_name}' or no clusters found.")

            # 7) Save results to Excel
            #    - avg_voltage_df: shape (n_subwindows, some_columns)
            #    - firing_rates_df: shape (n_subwindows + 2, n_clusters + 1), 
            #      because the last 2 rows are correlation info
            avg_voltage_df.to_excel(
                os.path.join(self.spikes.SAVE_DIRECTORY, "tables", 
                            f"{trial_name}_average_vf_voltage_windowed.xlsx"),
                index=False
            )
            firing_rates_df.to_excel(
                os.path.join(self.spikes.SAVE_DIRECTORY, "tables", 
                            f"{trial_name}_cluster_firing_rates_windowed.xlsx"),
                # index=False
            )

            self.windowed_results[trial_name] = {
                'avg_voltage_df': avg_voltage_df,
                'firing_rates_df': firing_rates_df
            }

        return self.windowed_results
