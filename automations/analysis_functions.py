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
          kwargs: additional parameters (e.g., subwindow_width, corr_threshold, fast_mode, skip_correlations, etc.) for analysis.
        
        Returns:
          combined_results: dict keyed by "RatID_TrialName" containing the analysis results.
        """
        combined_results = {}
        
        # Extract optimization parameters and show what's being used
        fast_mode = kwargs.get('fast_mode', False)
        skip_correlations = kwargs.get('skip_correlations', False)
        correlation_window_size = kwargs.get('correlation_window_size', None)
        cluster_types = kwargs.get('cluster_types', 'good')
        
        print(f"[INFO] Analysis settings:")
        print(f"  - Cluster types: {cluster_types}")
        print(f"  - Fast mode: {fast_mode}")
        print(f"  - Skip correlations: {skip_correlations}")
        if correlation_window_size:
            print(f"  - Custom correlation window: {correlation_window_size}")
        
        for rat_id, rat in self.rat_group.rats.items():
            print(f"\n[INFO] Processing rat: {rat_id}")
            si = self.si_wrappers[rat_id]
            ks = self.ks_wrappers[rat_id]
            # Automatically decide: if the Excel files exist in the proper subfolder, load them.
            # Pass good_clusters_by_trial if present in kwargs (for per-rat customization)
            rat_kwargs = dict(kwargs)
            if 'good_clusters_by_trial' in kwargs:
                rat_kwargs['good_clusters_by_trial'] = kwargs['good_clusters_by_trial'].get(rat_id, None)
            
            # Ensure optimization parameters are explicitly passed
            rat_kwargs['fast_mode'] = fast_mode
            rat_kwargs['skip_correlations'] = skip_correlations
            if correlation_window_size is not None:
                rat_kwargs['correlation_window_size'] = correlation_window_size
                
            results = rat.get_von_frey_analysis(si, ks, excel_parent_folder=excel_parent_folder, **rat_kwargs)
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

    def _smooth_moving_average(self, signal, window_size=15000, causal=False, use_fast_method=True):
        """
        Smooth 'signal' by convolving with a ones kernel of size 'window_size'.
        If causal=True, only past values are used (moving average up to current point).
        If causal=False, uses centered window (default, as before).
        If use_fast_method=True, uses scipy for faster computation on large signals.
        """
        if use_fast_method and len(signal) > 100000:  # Use faster method for large signals
            try:
                from scipy import ndimage
                if causal:
                    # For causal, we'll use a simple cumulative approach which is much faster
                    padded = np.concatenate([np.zeros(window_size-1), signal])
                    cumsum = np.cumsum(padded)
                    smoothed = (cumsum[window_size-1:] - np.concatenate([np.zeros(1), cumsum[:-window_size]])) / window_size
                    return smoothed[:len(signal)]
                else:
                    # Use scipy's uniform filter which is much faster than np.convolve
                    smoothed = ndimage.uniform_filter1d(signal.astype(np.float64), size=window_size, mode='constant')
                    return smoothed
            except ImportError:
                print("Warning: scipy not available, falling back to numpy method")
        
        # Original numpy method (fallback)
        kernel = np.ones(window_size) / window_size
        if causal:
            # Causal: only past and current values
            smoothed = np.convolve(signal, kernel, mode='full')[:len(signal)]
        else:
            # Centered (default)
            smoothed = np.convolve(signal, kernel, mode='same')
        return smoothed

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
                                               start_buffer=0.01, end_buffer=0.01, custom_data=None, custom_sampling_rate=None):
        """
        Process a single trial to identify Von Frey windows.
        If custom_data and custom_sampling_rate are provided, use them instead of loading from the trial.
        Returns a dictionary with 'adjusted_start_times' and 'adjusted_end_times' for the trial or segment.
        """
        # Use custom data if provided
        if custom_data is not None and custom_sampling_rate is not None:
            von_frey_data = custom_data
            sampling_rate = custom_sampling_rate
            num_samples = len(von_frey_data)
            time_vector = np.arange(num_samples) / sampling_rate
        else:
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
        intervals = {
            'adjusted_start_times': adjusted_start_times,
            'adjusted_end_times': adjusted_end_times
        }
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

    def compute_unit_firing_rates_for_subwindows(self, trial_name, subwindow_start_times, subwindow_end_times, good_clusters=None):
        """
        Compute firing rates for every cluster during each sub-window in a trial.
        This version uses vectorized np.searchsorted to quickly count spikes per subwindow.
        If good_clusters is provided, only those clusters are included.
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
        if good_clusters is not None:
            unique_clusters = [c for c in unique_clusters if c in good_clusters]
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

    def compute_unit_firing_rates_for_subwindows_modular(self, spike_times_sec, clusters, subwindow_start_times, subwindow_end_times):
        """
        Modular version: Compute firing rates for every cluster during each sub-window, given spike_times (in sec) and clusters.
        """
        unique_clusters = np.unique(clusters)
        durations = subwindow_end_times - subwindow_start_times
        firing_rates = {}
        for cluster in unique_clusters:
            spikes_cluster = spike_times_sec[clusters == cluster]
            counts = np.searchsorted(spikes_cluster, subwindow_end_times) - np.searchsorted(spikes_cluster, subwindow_start_times)
            firing_rates[cluster] = counts / durations
        firing_rates_df = pd.DataFrame(firing_rates).fillna(0)
        return firing_rates_df

    def compute_inv_isi_correlation(self, trial_name, window_size=15000, good_clusters=None):
        """s
        Compute the correlation between a smoothed inverse-ISI trace per cluster and the Von Frey signal.
        If good_clusters is provided, only those clusters are included.
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
        if good_clusters is not None:
            unique_clusters = [c for c in unique_clusters if c in good_clusters]
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

    def compute_inv_isi_correlation_modular(self, spike_times, clusters, fs, von_frey_data, window_size=15000, time_window=None, vf_magnitude=225000):
        """
        Modular version: Compute the correlation between a smoothed inverse-ISI trace per cluster and the Von Frey signal.
        Uses causal smoothing for ISI trace.
        Arguments:
            spike_times: array-like, spike times (in samples or seconds; if samples, fs must be provided)
            clusters: array-like, cluster assignments for each spike
            fs: sampling rate (Hz)
            von_frey_data: array-like, analog trace (should be same length as recording)
            window_size: smoothing window size (in samples)
            time_window: tuple of (start_time, end_time) in seconds, or None for full trace
        Returns:
            correlations: dict of cluster_id -> correlation value
            inv_isi_traces: dict of cluster_id -> smoothed inverse-ISI trace
        """
        # Pre-allocate arrays and dictionaries
        N = len(von_frey_data)
        unique_clusters = np.unique(clusters)
        correlations = {}
        inv_isi_traces = {}

        # Convert spike times to indices once
        if np.max(spike_times) > 1000:  # crude check: likely in samples
            spike_indices = spike_times.astype(int)
        else:  # likely in seconds
            spike_indices = (spike_times * fs).astype(int)

        # Handle time window if specified
        if time_window is not None:
            start_idx = int(time_window[0] * fs)
            end_idx = int(time_window[1] * fs)
            von_frey_data = von_frey_data[start_idx:end_idx]
            N = len(von_frey_data)
            # Filter spikes to only those within the time window
            mask = (spike_indices >= start_idx) & (spike_indices < end_idx)
            spike_indices = spike_indices[mask] - start_idx  # Adjust indices relative to window start
            clusters = clusters[mask]

        # Pre-compute von frey std since it's used for every cluster
        std_vf = np.std(von_frey_data)

        # Process each cluster
        for cluster in unique_clusters:
            # Get sorted spike indices for this cluster
            cluster_mask = clusters == cluster
            cluster_spike_indices = np.sort(spike_indices[cluster_mask])

            # Skip clusters with too few spikes
            if len(cluster_spike_indices) < 2:
                correlations[cluster] = 0.0
                inv_isi_traces[cluster] = np.zeros(N, dtype=float)
                continue

            # Pre-allocate inverse ISI trace
            inv_isi_trace = np.zeros(N, dtype=float)

            # Compute ISIs in one go
            isis = np.diff(cluster_spike_indices)
            isis[isis == 0] = 1  # Avoid division by zero
            
            # Fill the trace more efficiently
            for i, (start, isi) in enumerate(zip(cluster_spike_indices[:-1], isis)):
                if start >= N:
                    break
                end = min(cluster_spike_indices[i + 1], N)
                inv_isi_trace[start:end] = 1.0 / isi

            # Smooth and compute correlation (now using causal smoothing)
            inv_isi_trace_smoothed = self._smooth_moving_average(inv_isi_trace, window_size=window_size, causal=True)
            std_inv = np.std(inv_isi_trace_smoothed)
            
            corr_val = 0.0 if std_inv == 0 or std_vf == 0 else np.corrcoef(inv_isi_trace_smoothed, von_frey_data)[0, 1]
            
            correlations[cluster] = corr_val
            inv_isi_traces[cluster] = inv_isi_trace_smoothed

        return correlations, inv_isi_traces

    def plot_inv_isi_vs_von_frey(self, von_frey_data, inv_isi_traces, correlations, trial_name, cluster_id, title=None, vf_magnitude=225000, custom_windows=None, mask_range=None, psth_responsive=None):
        """
        Plot the smoothed inverse-ISI trace and Von Frey signal for a given cluster.
        Also highlight von Frey windows and show windowed correlation values.
        If custom_windows is provided, use those windows for highlighting and windowed correlation calculation.
        If mask_range=(start_idx, end_idx) is provided, mask (set to np.nan) the data outside this range before plotting.
        If psth_responsive is provided (list/array, one per window), display both r and responsiveness above each window.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        N = len(von_frey_data)
        t = np.arange(N)
        # Masking logic
        vf_plot = np.copy(von_frey_data)
        isi_plot = np.copy(inv_isi_traces[cluster_id])
        if mask_range is not None:
            start_idx, end_idx = mask_range
            if start_idx > 0:
                vf_plot[:start_idx] = np.nan
                isi_plot[:start_idx] = np.nan
            if end_idx < N:
                vf_plot[end_idx:] = np.nan
                isi_plot[end_idx:] = np.nan
        fig, ax1 = plt.subplots(figsize=(16, 8))  # Make plot taller
        ax1.plot(t, vf_plot, color='tab:blue', label='Von Frey Signal', alpha=0.7)
        ax1.set_ylabel('Von Frey Voltage', color='tab:blue')
        ax2 = ax1.twinx()
        ax2.plot(t, isi_plot, color='tab:red', label='Smoothed Inverse-ISI', alpha=0.7)
        ax2.set_ylabel('Smoothed Inverse-ISI', color='tab:red')
        ax1.set_xlabel('Sample Index')
        plot_title = title if title is not None else f'Trial: {trial_name}, Cluster: {cluster_id}\nCorrelation: {correlations[cluster_id]:.3f}'
        plt.title(plot_title, pad=20)  # Add padding above the title
        # --- Von Frey window highlighting and windowed correlation calculation ---
        if custom_windows is not None:
            intervals = custom_windows
        elif hasattr(self, 'von_frey_time_windows') and trial_name in self.von_frey_time_windows:
            intervals = self.von_frey_time_windows[trial_name]
        else:
            intervals = self._extract_von_frey_windows_single_trial(trial_name, amplitude_threshold=vf_magnitude, start_buffer=0.01, end_buffer=0.01)
        start_times = intervals['adjusted_start_times']
        end_times = intervals['adjusted_end_times']
        fs = N / (t[-1] if t[-1] > 0 else 1)
        if hasattr(self, 'spikes') and hasattr(self.spikes, 'kilosort_results') and trial_name in self.spikes.kilosort_results:
            fs = self.spikes.kilosort_results[trial_name]['ops']['fs']
        windowed_corrs = []
        for i, (start, end) in enumerate(zip(start_times, end_times)):
            start_idx_w = int(start * fs)
            end_idx_w = int(end * fs)
            if end_idx_w > N:
                end_idx_w = N
            if start_idx_w >= end_idx_w:
                continue
            isi_win = inv_isi_traces[cluster_id][start_idx_w:end_idx_w]
            vf_win = von_frey_data[start_idx_w:end_idx_w]
            std_isi = np.std(isi_win)
            std_vf = np.std(vf_win)
            if std_isi == 0 or std_vf == 0 or len(isi_win) < 2:
                corr = np.nan
            else:
                corr = np.corrcoef(isi_win, vf_win)[0, 1]
            windowed_corrs.append(corr)
            ax1.axvspan(start_idx_w, end_idx_w, color='orange', alpha=0.2)
            y_pos = np.nanmax(vf_plot) if np.any(~np.isnan(vf_plot)) else 0
            # Display both r and PSTH responsiveness if provided
            if psth_responsive is not None and i < len(psth_responsive):
                resp_label = 'Responsive' if psth_responsive[i] else 'Not responsive'
                text_str = f"r={corr:.2f} {resp_label}"
                ax1.text((start_idx_w + end_idx_w) / 2, y_pos * 1.05, text_str, color='black', fontsize=14, ha='center', va='bottom')
            else:
                ax1.text((start_idx_w + end_idx_w) / 2, y_pos * 1.05, f"r={corr:.2f}", color='black', fontsize=12, ha='center', va='bottom')
        if not hasattr(self, 'windowed_correlations'):
            self.windowed_correlations = {}
        if trial_name not in self.windowed_correlations:
            self.windowed_correlations[trial_name] = {}
        self.windowed_correlations[trial_name][cluster_id] = windowed_corrs
        fig.tight_layout()
        # Move legend below the plot
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2)

        plt.show()

    def analyze_subwindows(self, TRIAL_NAMES=None, amplitude_threshold=225000, start_buffer=0.001, 
                           end_buffer=0.001, subwindow_width=0.5, corr_threshold=0.01, good_clusters_by_trial=None,
                           fast_mode=False, skip_correlations=False, correlation_window_size=None):
        """
        Analyze Von Frey subwindows with optional performance optimizations.
        
        Parameters:
        -----------
        fast_mode : bool, default False
            If True, automatically optimizes parameters for large cluster counts
        skip_correlations : bool, default False
            If True, skips inverse ISI correlation computation (much faster)
        correlation_window_size : int, optional
            Custom window size for correlation smoothing. If None, uses 15000 (or 5000 in fast_mode)
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
            groups = ["pre-stim" if start < 35 else "post-stim" for start in subwindow_starts]
            avg_voltage_df.insert(0, 'group', groups)
            good_clusters = None
            if good_clusters_by_trial is not None and trial_name in good_clusters_by_trial:
                good_clusters = good_clusters_by_trial[trial_name]
            firing_rates_df = self.compute_unit_firing_rates_for_subwindows(trial_name, subwindow_starts, subwindow_ends, good_clusters=good_clusters)

            # Ensure 'group' is the first column before saving
            if 'group' not in firing_rates_df.columns:
                firing_rates_df.insert(0, 'group', groups)
            else:
                # Move 'group' to the first column if it's not already
                cols = ['group'] + [col for col in firing_rates_df.columns if col != 'group']
                firing_rates_df = firing_rates_df[cols]

            # Debug: print columns before saving
            if list(firing_rates_df.columns) == ['group']:
                print(f"[WARNING] Only 'group' column present in firing_rates_df for {trial_name}. No cluster columns will be saved!")
            else:
                if 'group' in firing_rates_df.columns:
                    cluster_cols = firing_rates_df.columns.drop('group').tolist()
                else:
                    cluster_cols = firing_rates_df.columns.tolist()
                print(f"[INFO] Cluster columns present: {cluster_cols}")

            # Apply performance optimizations based on cluster count and settings
            n_clusters = len(cluster_cols) if 'cluster_cols' in locals() else 0
            
            # Auto-enable fast mode for large cluster counts
            if n_clusters > 50 and not fast_mode:
                print(f"[INFO] Large number of clusters ({n_clusters}) detected. Consider using fast_mode=True for better performance.")
            
            # Set correlation window size based on mode
            if correlation_window_size is None:
                corr_window_size = 5000 if fast_mode else 15000
            else:
                corr_window_size = correlation_window_size
            
            # Compute correlations (unless skipped)
            if skip_correlations:
                print(f"[INFO] Skipping correlation computation for {trial_name} (skip_correlations=True)")
                correlations = {}
            else:
                if fast_mode and n_clusters > 30:
                    print(f"[INFO] Computing correlations for {n_clusters} clusters with fast mode (window_size={corr_window_size})...")
                correlations, _ = self.compute_inv_isi_correlation(trial_name, window_size=corr_window_size, good_clusters=good_clusters)
            
            if correlations:
                corr_row = pd.DataFrame({cluster: [corr_val] for cluster, corr_val in correlations.items()},
                                        index=['correlation'])
                is_corr_row = pd.DataFrame({cluster: [abs(corr_val) >= corr_threshold] for cluster, corr_val in correlations.items()},
                                           index=['is_correlated'])
                firing_rates_df = pd.concat([firing_rates_df, corr_row, is_corr_row], axis=0)
            else:
                if not skip_correlations:
                    print(f"No correlations computed for trial '{trial_name}' or no clusters found.")
                # Add empty correlation rows when skipped
                if skip_correlations and n_clusters > 0:
                    empty_corr = pd.DataFrame({col: [np.nan] for col in cluster_cols}, index=['correlation'])
                    empty_is_corr = pd.DataFrame({col: [False] for col in cluster_cols}, index=['is_correlated'])
                    firing_rates_df = pd.concat([firing_rates_df, empty_corr, empty_is_corr], axis=0)

            tables_dir = Path(self.spikes.SAVE_DIRECTORY) / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            avg_voltage_df.to_excel(tables_dir / f"{trial_name}_average_vf_voltage_windowed.xlsx", index=False)
            firing_rates_df.to_excel(tables_dir / f"{trial_name}_cluster_firing_rates_windowed.xlsx")
            self.windowed_results[trial_name] = {'voltage_df': avg_voltage_df, 'firing_df': firing_rates_df}

        return self.windowed_results

    def psth(self, spike_times, event_onsets, window=(-1, 3), bin_ms=50, fs=30000):
        import numpy as np
        bins = np.arange(window[0]*1e3, window[1]*1e3+bin_ms, bin_ms)
        counts = np.zeros_like(bins[:-1], dtype=float)
        for t0 in event_onsets:
            rel = (spike_times - t0) / fs * 1e3  # ms
            counts += np.histogram(rel, bins)[0]
        return bins[:-1]+bin_ms/2, counts / (len(event_onsets)*bin_ms/1e3)  # Hz

    def is_responsive_psth(self, spike_times, event_onsets, window=(-1,3), bin_ms=50, fs=30000, N=3, n_shuffles=1000, alpha=0.05, random_state=None):
        import numpy as np
        rng = np.random.default_rng(random_state)
        bins = np.arange(window[0]*1e3, window[1]*1e3+bin_ms, bin_ms)
        # Real PSTH
        t_centers, real_psth = self.psth(spike_times, event_onsets, window, bin_ms, fs)
        # Shuffle PSTHs
        shuff_psths = np.zeros((n_shuffles, len(real_psth)))
        event_onsets = np.array(event_onsets)
        min_onset = np.min(event_onsets)
        max_onset = np.max(event_onsets)
        window_samples = int((window[1] - window[0]) * fs)
        for i in range(n_shuffles):
            # Jitter each onset within the valid range
            jitter = rng.uniform(-0.5, 0.5, size=event_onsets.shape) * window_samples
            shuffled_onsets = event_onsets + jitter.astype(int)
            # Keep within bounds
            shuffled_onsets = np.clip(shuffled_onsets, min_onset, max_onset)
            _, shuff_psths[i] = self.psth(spike_times, shuffled_onsets, window, bin_ms, fs)
        # For each bin, get the (1-alpha) quantile of the shuffled distribution
        threshold = np.quantile(shuff_psths, 1-alpha, axis=0)
        # Find bins where real PSTH exceeds threshold
        above = real_psth > threshold
        # Find runs of consecutive bins above threshold
        from itertools import groupby
        from operator import itemgetter
        idx = np.where(above)[0]
        runs = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(idx), lambda x: x[1]-x[0])]
        max_run = max((len(run) for run in runs), default=0)
        is_resp = max_run >= N
        return is_resp, real_psth, threshold, t_centers, above

    def analyze_trial_clusters(self, rat_id, trial_name=None, plot_results=True, save_plots=True, psth_responsive=None):    
        """
        Performs comprehensive analysis of spike data, waveforms, and von Frey correlations for all good clusters in a trial.
        Now includes PSTH-based responsiveness detection for each cluster.
        """
        from automations.plots import plot_raster, plot_waveform
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path
        # Get the kilosort wrapper for this rat
        ksw = self.spikes
        # Get list of trials to analyze
        if trial_name is not None:
            trial_list = [trial_name]
        else:
            trial_list = list(ksw.kilosort_results.keys())
        results = {}
        for trial in trial_list:
            print(f"\nAnalyzing trial: {trial}")
            trial_results = {}
            # Get good clusters
            good_clusters = ksw.get_good_clusters(trial)
            if not good_clusters:
                print(f"No good clusters found for trial {trial}")
                continue
            # Get spike data
            spike_times_all = ksw.get_spike_times(trial)
            clusters = ksw.kilosort_results[trial]['spike_clusters']
            fs = ksw.kilosort_results[trial]['ops']['fs']
            # Filter for good clusters
            good_mask = np.isin(clusters, good_clusters)
            spike_times_good = spike_times_all[good_mask]
            clusters_good = clusters[good_mask]
            # Store basic info
            trial_results['good_clusters'] = good_clusters
            trial_results['fs'] = fs
            # Get von Frey data
            recording = self.rat.analog_data[trial]
            von_frey_data = recording.get_traces(channel_ids=['ANALOG-IN-2'], return_scaled=True).flatten()
            # Calculate correlations
            correlations, inv_isi_traces = self.compute_inv_isi_correlation_modular(
                spike_times_good, clusters_good, fs, von_frey_data, window_size=15000
            )
            # Store correlation results
            trial_results['correlations'] = correlations
            trial_results['inv_isi_traces'] = inv_isi_traces
            # PSTH-based responsiveness for each cluster
            # Use von Frey window onsets as events
            if trial in self.von_frey_time_windows:
                vf_windows = self.von_frey_time_windows[trial]
            else:
                vf_windows = self._extract_von_frey_windows_single_trial(trial)
            event_onsets = vf_windows['adjusted_start_times'] * fs  # in samples
            psth_responsive = {}
            psth_details = {}
            for cluster in good_clusters:
                # Get spike times for this cluster (in samples)
                st_samples = ksw.kilosort_results[trial]['spike_times'][ksw.kilosort_results[trial]['spike_clusters'] == cluster]
                is_resp, real_psth, threshold, t_centers, above = self.is_responsive_psth(
                    st_samples, event_onsets,
                    window=(-1,3), bin_ms=50, fs=fs, N=3, n_shuffles=1000, alpha=0.05
                )
                psth_responsive[cluster] = is_resp
                psth_details[cluster] = {
                    'real_psth': real_psth,
                    'threshold': threshold,
                    't_centers': t_centers,
                    'above': above
                }
            trial_results['psth_responsive'] = psth_responsive
            trial_results['psth_details'] = psth_details
            if plot_results:
                figures_dir = Path(self.spikes.SAVE_DIRECTORY) / 'figures' / trial
                if save_plots:
                    figures_dir.mkdir(parents=True, exist_ok=True)
                fig_raster = plot_raster(spike_times_good, clusters_good, fs=fs, 
                                       title=f'Raster: {trial}')
                if save_plots:
                    plt.savefig(figures_dir / f'raster_{trial}.png')
                    plt.close()
                for cluster in good_clusters:
                    waveform = ksw.get_waveform(trial, cluster)
                    if waveform is not None:
                        fig_wave = plot_waveform(waveform, 
                                               title=f'Waveform: {trial}, Cluster {cluster}')
                        if save_plots:
                            plt.savefig(figures_dir / f'waveform_{trial}_cluster_{cluster}.png')
                            plt.close()
                for cluster in correlations.keys():
                    self.plot_inv_isi_vs_von_frey(von_frey_data, inv_isi_traces, 
                                                correlations, trial, cluster,
                                                title=f'Full Recording: {trial}, Cluster {cluster}')
                    if save_plots:
                        plt.savefig(figures_dir / f'correlation_full_{trial}_cluster_{cluster}.png')
                        plt.close()
            results[trial] = trial_results
        return results

    def test_responsiveness_in_windows(self, spike_times, windows, fs, baseline=(-2, 0), n_shuffles=1000, alpha=0.05):
        """
        For each window (start, end in seconds), compare firing rate inside vs. baseline.
        Returns: list of dicts with window info and significance.
        """
        import numpy as np
        results = []
        spike_times_sec = spike_times / fs
        for start, end in zip(windows['adjusted_start_times'], windows['adjusted_end_times']):
            # Firing rate in window
            n_spikes = np.sum((spike_times_sec >= start) & (spike_times_sec < end))
            duration = end - start
            fr_in = n_spikes / duration if duration > 0 else 0

            # Baseline: same duration, just before window
            base_start = start + baseline[0]
            base_end = start + baseline[1]
            n_spikes_base = np.sum((spike_times_sec >= base_start) & (spike_times_sec < base_end))
            base_duration = base_end - base_start
            fr_base = n_spikes_base / base_duration if base_duration > 0 else 0

            # Shuffle: randomly shift window within recording, recompute difference
            diffs = []
            rec_duration = spike_times_sec.max()
            for _ in range(n_shuffles):
                shift = np.random.uniform(0, rec_duration - duration)
                n_spikes_shuff = np.sum((spike_times_sec >= shift) & (spike_times_sec < shift + duration))
                n_spikes_base_shuff = np.sum((spike_times_sec >= shift + baseline[0]) & (spike_times_sec < shift + baseline[1]))
                fr_shuff = n_spikes_shuff / duration if duration > 0 else 0
                fr_base_shuff = n_spikes_base_shuff / base_duration if base_duration > 0 else 0
                diffs.append(fr_shuff - fr_base_shuff)
            # p-value: how often is the real difference greater than shuffled
            real_diff = fr_in - fr_base
            pval = np.mean(np.array(diffs) >= real_diff)
            is_resp = pval < alpha
            results.append({'start': start, 'end': end, 'fr_in': fr_in, 'fr_base': fr_base, 'pval': pval, 'is_responsive': is_resp})
        return results
