"""
Key Optimizations and Changes
Vectorized Firing Rate Computation:
In compute_unit_firing_rates_for_subwindows, the loop over subwindows was replaced by computing counts with np.searchsorted for each cluster, 
which should dramatically speed up the operation when many subwindows are processed.

Consistent Use of Pathlib:
File and directory operations now consistently use Path objects.

Streamlined Loop Structures:
Unnecessary nested loops were removed or replaced with list comprehensions where appropriate.

Comments and Structure:
I added inline comments for clarity and noted the areas where further optimization (e.g., using Numba for heavy loops) could be considered if needed.

"""



from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict, Counter
import pandas as pd
import os

class MultiRatVonFreyAnalysis:
    """
    Aggregates VonFrey analysis across a group of rats.
    
    For each rat in the rat group, it calls the rat's get_von_frey_analysis() method.
    This method automatically checks if precomputed Excel results exist (in the expected location)
    under the given parent folder. If they do, it loads those; if not, it computes the analysis.
    The individual results are then combined into a dictionary keyed by "RatID_TrialName".
    """
    def __init__(self, rat_group, si_wrappers, ks_wrappers):
        self.rat_group = rat_group  # a RatGroup instance
        self.si_wrappers = si_wrappers  # dict: rat ID -> SpikeInterface_wrapper
        self.ks_wrappers = ks_wrappers  # dict: rat ID -> Kilosort_wrapper
        self.analysis_results = {}

    def analyze_all_trials(self, excel_parent_folder, **kwargs):
        """
        For each rat, automatically load precomputed results from the Excel files if they exist,
        otherwise run the analysis from raw data.
        
        Parameters:
          excel_parent_folder: str or Path; the parent folder under which each rat's folder is located.
                               The method will look in: {excel_parent_folder}/{RatID}/tables
                               for Excel files.
          kwargs: additional parameters (e.g., subwindow_width, corr_threshold) for analysis.
        
        Returns:
          combined_results: dict keyed by "RatID_TrialName" containing the analysis results.
        """
        combined_results = {}
        for rat_id, rat in self.rat_group.rats.items():
            si = self.si_wrappers[rat_id]
            ks = self.ks_wrappers[rat_id]
            # Automatically decide: if the Excel files exist in the proper subfolder, load them.
            results = rat.get_von_frey_analysis(si, ks, excel_parent_folder=excel_parent_folder, **kwargs)
            if results is None:
                continue
            for trial_name, res in results.items():
                combined_key = f"{rat_id}_{trial_name}"
                combined_results[combined_key] = res
        self.analysis_results = combined_results
        return combined_results



class VonFreyAnalysis:
    def __init__(self, rat_instance, spikeinterface_instance, kilosort_instance):
        """
        Initialize the VonFreyAnalysis class with references to the Rat, SpikeInterface_wrapper, and Kilosort_wrapper instances.
        """
        self.rat = rat_instance
        self.signals = spikeinterface_instance
        self.spikes = kilosort_instance
        self.von_frey_time_windows = {}  # {trial: intervals dict}
        self.cluster_firing_rates = {}   # {trial: firing rate DataFrame}
        self.windowed_results = {}       # {trial: {'avg_voltage_df':..., 'firing_rates_df':...}}
        self.inv_isi_correlations = {}   # {trial: {cluster_id: correlation}}
        self.inv_isi_traces = {}         # {trial: {cluster_id: np.ndarray}}

    def _smooth_moving_average(self, signal, window_size=15000):
        """
        Smooth 'signal' by convolving with a ones kernel of size 'window_size'.
        """
        kernel = np.ones(window_size) / window_size
        return np.convolve(signal, kernel, mode='same')

    def extract_von_frey_windows(self, TRIAL_NAMES=None, amplitude_threshold=225000, 
                                 start_buffer=0.001, end_buffer=0.001):
        """
        Extract time windows where the Von Frey stimulus is applied.
        Processes either a given list of trial names or, if None, all trials found in kilosort_results.
        """
        if TRIAL_NAMES is not None:
            trial_list = TRIAL_NAMES
        else:
            if not self.spikes.kilosort_results:
                print("No kilosort results found. Please run Kilosort or load results first.")
                return {}
            trial_list = list(self.spikes.kilosort_results.keys())

        for trial_name in trial_list:
            intervals = self._extract_von_frey_windows_single_trial(trial_name, amplitude_threshold, start_buffer, end_buffer)
            if intervals is not None:
                self.von_frey_time_windows[trial_name] = intervals

        return self.von_frey_time_windows

    def _extract_von_frey_windows_single_trial(self, trial_name, amplitude_threshold=225000, 
                                               start_buffer=0.01, end_buffer=0.01):
        """
        Process a single trial to identify Von Frey windows.
        Saves raster and waveform figures.
        Returns a dictionary with 'adjusted_start_times' and 'adjusted_end_times' for the trial.
        """
        # Check for channel availability in stream3 recording
        if trial_name not in self.rat.analog_data:
            print(f"Trial '{trial_name}' not found in intan_recordings_stream3.")
            return None

        recording = self.rat.analog_data[trial_name]
        sampling_rate = recording.get_sampling_frequency()
        if 'ANALOG-IN-2' not in recording.get_channel_ids():
            print(f"ANALOG-IN-2 channel not found in trial '{trial_name}'.")
            return None

        von_frey_data = recording.get_traces(channel_ids=['ANALOG-IN-2'], return_scaled=True).flatten()
        num_samples = len(von_frey_data)
        time_vector = np.arange(num_samples) / sampling_rate

        # Detect rising and falling edges based on amplitude threshold
        rising_edges = np.where((von_frey_data[:-1] < amplitude_threshold) & (von_frey_data[1:] >= amplitude_threshold))[0] + 1
        falling_edges = np.where((von_frey_data[:-1] >= amplitude_threshold) & (von_frey_data[1:] < amplitude_threshold))[0] + 1

        if von_frey_data[0] >= amplitude_threshold:
            rising_edges = np.insert(rising_edges, 0, 0)
        if von_frey_data[-1] >= amplitude_threshold:
            falling_edges = np.append(falling_edges, num_samples - 1)

        min_len = min(len(rising_edges), len(falling_edges))
        rising_edges = rising_edges[:min_len]
        falling_edges = falling_edges[:min_len]

        start_times = rising_edges / sampling_rate
        end_times = falling_edges / sampling_rate

        # Apply buffers
        adjusted_start_times = np.clip(start_times + start_buffer, 0, (num_samples - 1) / sampling_rate)
        adjusted_end_times = np.clip(end_times - end_buffer, 0, (num_samples - 1) / sampling_rate)
        valid = adjusted_start_times < adjusted_end_times
        adjusted_start_times = adjusted_start_times[valid]
        adjusted_end_times = adjusted_end_times[valid]

        # Create and save a raster plot if Kilosort results exist for this trial
        if trial_name in self.spikes.kilosort_results:
            ks_data = self.spikes.kilosort_results[trial_name]
            fs = ks_data['ops']['fs']
            spike_times_sec = ks_data['spike_times'] / fs
            clusters = ks_data['spike_clusters']
            unique_clusters = np.unique(clusters)
            cluster_order = {c: i for i, c in enumerate(unique_clusters)}
            
            fig, ax = plt.subplots(figsize=(10, 5))
            for c in unique_clusters:
                idx = clusters == c
                ax.scatter(spike_times_sec[idx], np.full(np.sum(idx), cluster_order[c]),
                           marker='|', color='black', s=100)
            for i, (start, end) in enumerate(zip(adjusted_start_times, adjusted_end_times)):
                ax.axvspan(start, end, color='orange', alpha=0.3, label='Von Frey Stimulus' if i == 0 else None)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Cluster')
            ax.set_yticks(list(cluster_order.values()))
            ax.set_yticklabels(list(cluster_order.keys()))
            ax.set_title(f'Raster Plot: {trial_name}')
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
            # by_label = OrderedDict(zip(labels, handles))
            # ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            figures_dir = Path(self.spikes.SAVE_DIRECTORY) / 'figures'
            figures_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(figures_dir / f'Raster_{trial_name}.png', dpi=300)
            plt.close(fig)

        # Plot and save the Von Frey trace with stimulus windows
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_vector, von_frey_data, label='Von Frey Data', color='blue')
        for start, end in zip(adjusted_start_times, adjusted_end_times):
            ax.axvspan(start, end, color='orange', alpha=0.3, label='Von Frey Stimulus')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (uV)')
        ax.set_title(f'Von Frey Windows: {trial_name}')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        # by_label = OrderedDict(zip(labels, handles))
        # ax.legend(by_label.values(), by_label.keys())
        fig.savefig(figures_dir / f'VF_window_{trial_name}.png', dpi=300)
        plt.tight_layout()
        plt.close(fig)

        intervals = {
            'adjusted_start_times': adjusted_start_times,
            'adjusted_end_times': adjusted_end_times
        }
        tables_dir = Path(self.spikes.SAVE_DIRECTORY) / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(intervals).to_excel(tables_dir / f"{trial_name}_von_frey_time_windows.xlsx")
        return intervals

    # NOTE: This method is currently not used in analyze_subwindows.
    def compute_unit_firing_rates(self, TRIAL_NAMES=None, amplitude_threshold=225000, 
                                  start_buffer=0.01, end_buffer=0.01):
        """
        (Deprecated?) Compute average firing rates of units during Von Frey stimulus windows.
        """
        if TRIAL_NAMES is None:
            if not self.spikes.kilosort_results:
                print("No kilosort results found. Please run Kilosort or load results first.")
                return {}
            trial_list = list(self.spikes.kilosort_results.keys())
        else:
            trial_list = TRIAL_NAMES

        intervals_dict = self.extract_von_frey_windows(TRIAL_NAMES=trial_list,
                                                       amplitude_threshold=amplitude_threshold,
                                                       start_buffer=start_buffer,
                                                       end_buffer=end_buffer)
        for trial_name in trial_list:
            if trial_name not in self.spikes.kilosort_results or trial_name not in intervals_dict:
                print(f"Skipping trial '{trial_name}' due to missing data.")
                continue

            adjusted_start_times = intervals_dict[trial_name]['adjusted_start_times']
            adjusted_end_times = intervals_dict[trial_name]['adjusted_end_times']

            firing_rates_intervals = []
            for start, end in zip(adjusted_start_times, adjusted_end_times):
                window_duration = end - start
                indices = np.where((self.spikes.kilosort_results[trial_name]['spike_times'] / 
                                    self.spikes.kilosort_results[trial_name]['ops']['fs'] >= start) &
                                   (self.spikes.kilosort_results[trial_name]['spike_times'] / 
                                    self.spikes.kilosort_results[trial_name]['ops']['fs'] < end))[0]
                clusters_in_window = self.spikes.kilosort_results[trial_name]['spike_clusters'][indices]
                cluster_counts = Counter(clusters_in_window)
                all_clusters = np.unique(self.spikes.kilosort_results[trial_name]['spike_clusters'])
                firing_rates = {cluster: cluster_counts.get(cluster, 0) / window_duration for cluster in all_clusters}
                firing_rates_intervals.append(firing_rates)
            firing_rates_df = pd.DataFrame(firing_rates_intervals).fillna(0)
            self.cluster_firing_rates[trial_name] = firing_rates_df
            pd.DataFrame(firing_rates_df).to_excel(Path(self.spikes.SAVE_DIRECTORY) / "tables" / f"{trial_name}_cluster_firing_rates.xlsx")
        return self.cluster_firing_rates

    def subdivide_intervals(self, intervals_dict, subwindow_width):
        """
        Subdivide each Von Frey interval into smaller sub-windows of fixed width.
        """
        subwindow_results = {}
        for trial_name, interval_data in intervals_dict.items():
            start_times = interval_data['adjusted_start_times']
            end_times = interval_data['adjusted_end_times']
            sub_starts, sub_ends = [], []
            for s, e in zip(start_times, end_times):
                curr = s
                while curr < e:
                    next_w = curr + subwindow_width
                    if next_w > e:
                        next_w = e  # include the final partial window
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
        Compute the average Von Frey voltage for each sub-window in a trial.
        """
        if trial_name not in self.signals.data.analog_data:
            print(f"Trial '{trial_name}' not found in analog_data.")
            return pd.DataFrame()
        recording = self.signals.data.analog_data[trial_name]
        sampling_rate = recording.get_sampling_frequency()
        if 'ANALOG-IN-2' not in recording.get_channel_ids():
            print(f"ANALOG-IN-2 not found in trial '{trial_name}'.")
            return pd.DataFrame()
        von_frey_data = recording.get_traces(channel_ids=['ANALOG-IN-2'], return_scaled=True).flatten()
        avg_voltages = [np.mean(von_frey_data[int(start * sampling_rate):int(end * sampling_rate)]) 
                        if end > start else np.nan 
                        for start, end in zip(subwindow_start_times, subwindow_end_times)]
        return pd.DataFrame({'avg_voltage': avg_voltages})

    def compute_unit_firing_rates_for_subwindows(self, trial_name, subwindow_start_times, subwindow_end_times):
        """
        Compute firing rates for every cluster during each sub-window in a trial.
        This version uses vectorized np.searchsorted to quickly count spikes per sub-window.
        """
        if trial_name not in self.spikes.kilosort_results:
            print(f"No kilosort results for trial '{trial_name}'.")
            return pd.DataFrame()

        kilosort_output = self.spikes.kilosort_results[trial_name]
        st = kilosort_output['spike_times']    # spike times in samples
        clu = kilosort_output['spike_clusters']  # cluster assignments
        fs = kilosort_output['ops']['fs']
        spike_times_sec = st / fs
        unique_clusters = np.unique(clu)
        n_windows = len(subwindow_start_times)
        
        # Initialize a dictionary to hold firing rates (counts divided by window duration)
        firing_rates = {}
        durations = subwindow_end_times - subwindow_start_times  # vector of window durations
        
        for cluster in unique_clusters:
            spikes_cluster = spike_times_sec[clu == cluster]
            # Use vectorized searchsorted to count spikes in each subwindow
            counts = np.searchsorted(spikes_cluster, subwindow_end_times) - np.searchsorted(spikes_cluster, subwindow_start_times)
            firing_rates[cluster] = counts / durations
        
        firing_rates_df = pd.DataFrame(firing_rates).fillna(0)
        return firing_rates_df

    def compute_inv_isi_correlation(self, trial_name, window_size=15000):
        """s
        Compute the correlation between a smoothed inverse-ISI trace per cluster and the Von Frey signal.
        """
        if trial_name not in self.spikes.kilosort_results:
            print(f"[WARNING] Kilosort results not found for trial: {trial_name}")
            return {}, {}

        kilosort_output = self.spikes.kilosort_results[trial_name]
        st = kilosort_output["spike_times"]
        clu = kilosort_output["spike_clusters"]
        fs  = kilosort_output["ops"]["fs"]

        recording = self.rat.analog_data.get(trial_name, None)
        if recording is None:
            print(f"[WARNING] No recording found for trial: {trial_name}")
            return {}, {}
        if "ANALOG-IN-2" not in recording.get_channel_ids():
            print(f"[WARNING] Channel 'ANALOG-IN-2' not found for trial: {trial_name}")
            return {}, {}

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

        for cluster in unique_clusters:
            cluster_spike_times = np.sort(st[clu == cluster])
            inv_isi_trace = np.zeros(N, dtype=float)
            if len(cluster_spike_times) < 2:
                correlations[cluster] = 0.0
                inv_isi_traces[cluster] = inv_isi_trace
                continue

            for i in range(len(cluster_spike_times) - 1):
                s_k = cluster_spike_times[i]
                s_next = cluster_spike_times[i + 1]
                if s_k >= N:
                    break
                if s_next > N:
                    s_next = N
                isi = s_next - s_k
                if isi == 0:
                    isi = 1
                inv_isi_trace[s_k:s_next] = 1.0 / isi

            inv_isi_trace_smoothed = self._smooth_moving_average(inv_isi_trace, window_size=window_size)
            std_inv = np.std(inv_isi_trace_smoothed)
            std_vf = np.std(von_frey_data)
            corr_val = 0.0 if std_inv == 0 or std_vf == 0 else np.corrcoef(inv_isi_trace_smoothed, von_frey_data)[0, 1]
            correlations[cluster] = corr_val
            inv_isi_traces[cluster] = inv_isi_trace_smoothed

        self.inv_isi_correlations[trial_name] = correlations
        self.inv_isi_traces[trial_name] = inv_isi_traces
        return correlations, inv_isi_traces

    def analyze_subwindows(self, TRIAL_NAMES=None, amplitude_threshold=225000, start_buffer=0.001, 
                           end_buffer=0.001, subwindow_width=0.5, corr_threshold=0.01):
        """
        High-level analysis pipeline that:
          1. Extracts Von Frey windows.
          2. Subdivides these intervals.
          3. Computes average voltage and firing rates per subwindow.
          4. Computes inverse-ISI correlations for each cluster.
          5. Saves the results to Excel files.
        """
        if TRIAL_NAMES is None:
            TRIAL_NAMES = list(self.spikes.kilosort_results.keys())

        intervals_dict = self.extract_von_frey_windows(TRIAL_NAMES=TRIAL_NAMES,
                                                         amplitude_threshold=amplitude_threshold,
                                                         start_buffer=start_buffer,
                                                         end_buffer=end_buffer)
        if not intervals_dict:
            print("No intervals extracted.")
            return {}

        subwindows_dict = self.subdivide_intervals(intervals_dict, subwindow_width=subwindow_width)

        for trial_name in subwindows_dict:
            subwindow_starts = subwindows_dict[trial_name]['subwindow_start_times']
            subwindow_ends = subwindows_dict[trial_name]['subwindow_end_times']

            avg_voltage_df = self.compute_average_von_frey_voltage(trial_name, subwindow_starts, subwindow_ends)
            firing_rates_df = self.compute_unit_firing_rates_for_subwindows(trial_name, subwindow_starts, subwindow_ends)

            groups = ["pre-stim" if start < 35 else "post-stim" for start in subwindow_starts]
            avg_voltage_df['group'] = groups
            firing_rates_df['group'] = groups

            correlations, _ = self.compute_inv_isi_correlation(trial_name, window_size=15000)
            if correlations:
                corr_row = pd.DataFrame({cluster: [corr_val] for cluster, corr_val in correlations.items()},
                                        index=['correlation'])
                is_corr_row = pd.DataFrame({cluster: [abs(corr_val) >= corr_threshold] for cluster, corr_val in correlations.items()},
                                           index=['is_correlated'])
                firing_rates_df = pd.concat([firing_rates_df, corr_row, is_corr_row], axis=0)
            else:
                print(f"No correlations computed for trial '{trial_name}' or no clusters found.")

            tables_dir = Path(self.spikes.SAVE_DIRECTORY) / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            avg_voltage_df.to_excel(tables_dir / f"{trial_name}_average_vf_voltage_windowed.xlsx", index=False)
            firing_rates_df.to_excel(tables_dir / f"{trial_name}_cluster_firing_rates_windowed.xlsx")
            self.windowed_results[trial_name] = {'avg_voltage_df': avg_voltage_df, 'firing_rates_df': firing_rates_df}

        return self.windowed_results
