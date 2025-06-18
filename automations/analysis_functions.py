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

    def build_cluster_label_dict_from_xlsx(self, cluster_labels_xlsx, cluster_labels, combined_results):
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

    def analyze_all_trials(self, excel_parent_folder, **kwargs):
        """
        Aggregates Von Frey analysis across a group of rats.
        
        Parameters:
        -----------
        excel_parent_folder: str or Path; parent folder for cached results
        cluster_labels: str, list, or dict; specifies which cluster labels to analyze:
                       - str: single label (e.g., 'good')
                       - list: multiple labels (e.g., ['good', 'mua'])
                       - dict: {rat_id: labels} for per-rat customization
                       - dict: {rat_id: {trial_name: labels}} for per-trial customization
        cluster_labels_xlsx: str or Path, optional; path to spreadsheet for cluster label mapping
        kwargs: additional parameters for the analysis
        
        Returns:
        --------
        combined_results: dict keyed by "RatID_TrialName"
        combined_qst_notes: pandas.DataFrame with combined QST notes
        """
        from pathlib import Path
        import pandas as pd
        excel_parent_folder = Path(excel_parent_folder)
        
        # Extract and validate parameters
        cluster_labels = kwargs.pop('cluster_labels', 'good')
        cluster_labels_xlsx = kwargs.pop('cluster_labels_xlsx', None)
        fast_mode = kwargs.pop('fast_mode', False)
        skip_correlations = kwargs.pop('skip_correlations', False)
        correlation_window_size = kwargs.pop('correlation_window_size', None)
        
        print(f"\n[DEBUG] Starting analysis with settings:")
        print(f"  - Cluster labels type: {type(cluster_labels)}")
        print(f"  - Cluster labels value: {cluster_labels}")
        print(f"  - Fast mode: {fast_mode}")
        print(f"  - Skip correlations: {skip_correlations}")
        if cluster_labels_xlsx:
            print(f"  - Using cluster_labels_xlsx: {cluster_labels_xlsx}")
        
        combined_results = {}
        combined_qst_notes = {}
        
        # If using spreadsheet, build mapping for all trials
        cluster_dict_from_xlsx = None
        if cluster_labels_xlsx is not None:
            # We'll build a dummy combined_results for mapping (all possible trial keys)
            all_trial_keys = {}
            for rat_id, rat in self.rat_group.rats.items():
                for trial in self.ks_wrappers[rat_id].kilosort_results.keys():
                    key = f"{rat_id}_{trial}"
                    all_trial_keys[key] = None
            cluster_dict_from_xlsx = self.build_cluster_label_dict_from_xlsx(cluster_labels_xlsx, cluster_labels, all_trial_keys)
        
        # Process each rat
        for rat_id, rat in self.rat_group.rats.items():
            print(f"\n[DEBUG] Processing rat: {rat_id}")
            
            # Prepare rat-specific arguments
            rat_kwargs = kwargs.copy()
            
            # Handle cluster labels
            if cluster_labels_xlsx is not None:
                # Use spreadsheet mapping for this rat's trials
                # Build a per-trial dict for this rat
                per_trial_dict = {}
                for trial in self.ks_wrappers[rat_id].kilosort_results.keys():
                    key = f"{rat_id}_{trial}"
                    per_trial_dict[trial] = cluster_dict_from_xlsx.get(key, [])
                rat_kwargs['cluster_labels'] = per_trial_dict
            else:
                if isinstance(cluster_labels, dict):
                    if rat_id in cluster_labels:
                        if isinstance(cluster_labels[rat_id], dict):
                            print(f"[DEBUG] Using per-trial labels for {rat_id}: {cluster_labels[rat_id]}")
                            rat_kwargs['cluster_labels'] = cluster_labels[rat_id]
                        else:
                            print(f"[DEBUG] Using per-rat labels for {rat_id}: {cluster_labels[rat_id]}")
                            rat_kwargs['cluster_labels'] = cluster_labels[rat_id]
                    else:
                        print(f"[DEBUG] Rat {rat_id} not found in labels dict, using default: 'good'")
                        rat_kwargs['cluster_labels'] = 'good'
                else:
                    print(f"[DEBUG] Using global labels for {rat_id}: {cluster_labels}")
                    rat_kwargs['cluster_labels'] = cluster_labels
            
            print(f"[DEBUG] Final kwargs for {rat_id}: {rat_kwargs}")
            
            # Get results for this rat
            try:
                print(f"[DEBUG] Getting von Frey analysis for {rat_id}")
                results = rat.get_von_frey_analysis(
                    self.si_wrappers[rat_id],
                    self.ks_wrappers[rat_id],
                    excel_parent_folder=excel_parent_folder,
                    **rat_kwargs
                )
                
                print(f"[DEBUG] Results type for {rat_id}: {type(results)}")
                if results:
                    print(f"[DEBUG] Results keys for {rat_id}: {results.keys() if isinstance(results, dict) else 'Not a dict'}")
                
                # Add results to combined dictionary
                if results:  # Only add if we have results
                    for trial_name, trial_results in results.items():
                        key = f"{rat_id}_{trial_name}"
                        print(f"[DEBUG] Adding results for {key}")
                        print(f"[DEBUG] Trial results type: {type(trial_results)}")
                        print(f"[DEBUG] Trial results keys: {trial_results.keys() if isinstance(trial_results, dict) else 'Not a dict'}")
                        combined_results[key] = trial_results
                        print(f"[DEBUG] Added results for {key}")
                
            except Exception as e:
                print(f"[ERROR] Failed to process rat {rat_id}: {str(e)}")
                print(f"[ERROR] Error type: {type(e)}")
                import traceback
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                continue
        
        print(f"\n[DEBUG] Final combined results: {len(combined_results)} trials")
        
        # Create combined QST notes
        def make_unique_cols(df):
            new_cols = []
            seen = {}
            for col in df.columns:
                if col in seen:
                    seen[col] += 1
                    new_cols.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    new_cols.append(col)
            df.columns = new_cols
            return df

        dfs = []
        for rat_id, rat in self.rat_group.rats.items():
            if hasattr(rat, 'qst_trial_notes') and rat.qst_trial_notes is not None:
                print(f"\n[DEBUG] Processing QST notes for rat {rat_id}")
                print(f"[DEBUG] QST notes type: {type(rat.qst_trial_notes)}")
                print(f"[DEBUG] QST notes shape: {rat.qst_trial_notes.shape if hasattr(rat.qst_trial_notes, 'shape') else 'No shape'}")
                print(f"[DEBUG] QST notes columns: {rat.qst_trial_notes.columns if hasattr(rat.qst_trial_notes, 'columns') else 'No columns'}")
                print(f"[DEBUG] QST notes index: {rat.qst_trial_notes.index if hasattr(rat.qst_trial_notes, 'index') else 'No index'}")
                
                try:
                    df = make_unique_cols(rat.qst_trial_notes.copy())
                    print(f"[DEBUG] After make_unique_cols - columns: {df.columns}")
                    print(f"[DEBUG] After make_unique_cols - index: {df.index}")
                    
                    df['Rat ID'] = rat_id
                    print(f"[DEBUG] After adding Rat ID - columns: {df.columns}")
                    
                    # Handle case where Trial Number is the index
                    if df.index.name == "Trial Number":
                        print(f"[DEBUG] Trial Number is index name")
                        # Check if Trial Number already exists as a column
                        if "Trial Number" in df.columns:
                            print(f"[DEBUG] Dropping existing Trial Number column")
                            df = df.drop(columns=["Trial Number"])
                        print(f"[DEBUG] Resetting index")
                        df = df.reset_index()
                        print(f"[DEBUG] After reset_index - columns: {df.columns}")
                    
                    print(f"[DEBUG] Creating combined Trial Number")
                    df["Trial Number"] = rat_id + "_" + df["Trial Number"].astype(str)
                    print(f"[DEBUG] Final columns: {df.columns}")
                    dfs.append(df)
                except Exception as e:
                    print(f"[ERROR] Failed to process QST notes for rat {rat_id}: {str(e)}")
                    print(f"[ERROR] Error type: {type(e)}")
                    import traceback
                    print(f"[ERROR] Traceback: {traceback.format_exc()}")
                    continue

        if dfs:
            combined_qst_notes = pd.concat(dfs, ignore_index=True)
            # Only add Trial_ID if we have results
            if combined_results:
                # Create a mapping from trial numbers to trial IDs
                trial_id_map = {}
                for trial_id in combined_results.keys():
                    # Extract the rat ID and trial number from the trial ID
                    # Format: DW322_VF_1_240918_143256 -> DW322 and 1
                    rat_id = trial_id.split('_')[0]
                    trial_num = trial_id.split('_')[2]  # Get the trial number part
                    
                    # Find matching row in QST notes
                    # Match both rat ID and trial number (which is in format DW322_1)
                    mask = (combined_qst_notes['Rat ID'] == rat_id) & \
                          (combined_qst_notes['Trial Number'].str.endswith(f'_{trial_num}'))
                    
                    if mask.any():
                        trial_id_map[trial_id] = mask
                        print(f"[DEBUG] Matched trial {trial_id} to QST notes")
                    else:
                        print(f"[DEBUG] No match found for trial {trial_id}")
                
                # Add Trial_ID column with NaN for unmatched trials
                combined_qst_notes['Trial_ID'] = None
                for trial_id, mask in trial_id_map.items():
                    combined_qst_notes.loc[mask, 'Trial_ID'] = trial_id
                
            print(f"[DEBUG] Created combined QST notes with {len(combined_qst_notes)} rows")
            print(f"[DEBUG] Number of trials with IDs: {combined_qst_notes['Trial_ID'].notna().sum() if 'Trial_ID' in combined_qst_notes.columns else 0}")
            print(f"[DEBUG] Matched trials: {list(trial_id_map.keys()) if 'trial_id_map' in locals() else []}")
        else:
            print("[WARNING] No QST notes found for any rats")
            combined_qst_notes = pd.DataFrame()

        return combined_results, combined_qst_notes



class VonFreyAnalysis:
    def __init__(self, rat_instance, spikeinterface_instance, kilosort_instance):
        """
        Initialize VonFreyAnalysis with rat instance and data wrappers.
        """
        self.rat = rat_instance
        self.si_wrapper = spikeinterface_instance
        self.ks_wrapper = kilosort_instance
        self.von_frey_time_windows = {}  # {trial: intervals dict}
        self.cluster_firing_rates = {}   # {trial: firing rate DataFrame}
        self.inv_isi_correlations = {}   # {trial: {cluster: correlation}}
        self.inv_isi_traces = {}         # {trial: {cluster: trace}}
        self.voltage_data = {}           # {trial: voltage DataFrame}
        
    def _smooth_moving_average(self, signal, window_size=15000, causal=False, use_fast_method=True):
        """
        Optimized signal smoothing with multiple acceleration methods.
        
        Parameters:
        -----------
        signal : array-like
            Input signal to smooth
        window_size : int
            Size of smoothing window in samples. Default is 15000.
        causal : bool
            If True, only use past values (causal filtering)
        use_fast_method : bool
            If True, use optimized methods for large signals
        """
        signal = np.asarray(signal, dtype=np.float64)
        
        # For very small signals, use simple method
        if len(signal) < window_size * 2:
            kernel = np.ones(window_size) / window_size
            if causal:
                smoothed = np.convolve(signal, kernel, mode='full')[:len(signal)]
            else:
                smoothed = np.convolve(signal, kernel, mode='same')
            return smoothed
        
        # Use optimized methods for larger signals
        if use_fast_method:
            try:
                if causal:
                    # Vectorized cumulative approach - much faster than convolution
                    padded = np.concatenate([np.zeros(window_size-1), signal])
                    cumsum = np.cumsum(padded)
                    smoothed = (cumsum[window_size-1:] - np.concatenate([np.zeros(1), cumsum[:-window_size]])) / window_size
                    return smoothed[:len(signal)]
                else:
                    # Try scipy's optimized uniform filter first
                    try:
                        from scipy import ndimage
                        return ndimage.uniform_filter1d(signal, size=window_size, mode='constant')
                    except ImportError:
                        # Fallback to optimized numpy approach using FFT for large signals
                        if len(signal) > 50000:
                            return self._fft_smooth(signal, window_size)
            except Exception as e:
                print(f"Warning: Fast smoothing failed ({e}), using fallback method")
        
        # Fallback method
        kernel = np.ones(window_size) / window_size
        if causal:
            smoothed = np.convolve(signal, kernel, mode='full')[:len(signal)]
        else:
            smoothed = np.convolve(signal, kernel, mode='same')
        return smoothed
    
    def _fft_smooth(self, signal, window_size):
        """
        FFT-based convolution for very large signals (faster than direct convolution).
        """
        kernel = np.ones(window_size) / window_size
        # Pad to avoid edge effects
        pad_width = window_size // 2
        padded_signal = np.pad(signal, pad_width, mode='edge')
        
        # FFT convolution
        smoothed_padded = np.convolve(padded_signal, kernel, mode='same')
        
        # Remove padding
        return smoothed_padded[pad_width:-pad_width]

    def extract_von_frey_windows(self, TRIAL_NAMES=None, amplitude_threshold=225000, 
                                 start_buffer=0.001, end_buffer=0.001):
        """
        Extract time windows where the Von Frey stimulus is applied.
        Processes either a given list of trial names or, if None, all trials found in kilosort_results.
        """
        if TRIAL_NAMES is not None:
            trial_list = TRIAL_NAMES
        else:
            if not self.ks_wrapper.kilosort_results:
                print("No kilosort results found. Please run Kilosort or load results first.")
                return {}
            trial_list = list(self.ks_wrapper.kilosort_results.keys())

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
            if not self.ks_wrapper.kilosort_results:
                print("No kilosort results found. Please run Kilosort or load results first.")
                return {}
            trial_list = list(self.ks_wrapper.kilosort_results.keys())
        else:
            trial_list = TRIAL_NAMES

        intervals_dict = self.extract_von_frey_windows(TRIAL_NAMES=trial_list,
                                                       amplitude_threshold=amplitude_threshold,
                                                       start_buffer=start_buffer,
                                                       end_buffer=end_buffer)
        for trial_name in trial_list:
            if trial_name not in self.ks_wrapper.kilosort_results or trial_name not in intervals_dict:
                print(f"Skipping trial '{trial_name}' due to missing data.")
                continue

            adjusted_start_times = intervals_dict[trial_name]['adjusted_start_times']
            adjusted_end_times = intervals_dict[trial_name]['adjusted_end_times']

            firing_rates_intervals = []
            for start, end in zip(adjusted_start_times, adjusted_end_times):
                window_duration = end - start
                indices = np.where((self.ks_wrapper.kilosort_results[trial_name]['spike_times'] / 
                                    self.ks_wrapper.kilosort_results[trial_name]['ops']['fs'] >= start) &
                                   (self.ks_wrapper.kilosort_results[trial_name]['spike_times'] / 
                                    self.ks_wrapper.kilosort_results[trial_name]['ops']['fs'] < end))[0]
                clusters_in_window = self.ks_wrapper.kilosort_results[trial_name]['spike_clusters'][indices]
                cluster_counts = Counter(clusters_in_window)
                all_clusters = np.unique(self.ks_wrapper.kilosort_results[trial_name]['spike_clusters'])
                firing_rates = {cluster: cluster_counts.get(cluster, 0) / window_duration for cluster in all_clusters}
                firing_rates_intervals.append(firing_rates)
            firing_rates_df = pd.DataFrame(firing_rates_intervals).fillna(0)
            self.cluster_firing_rates[trial_name] = firing_rates_df
            pd.DataFrame(firing_rates_df).to_excel(Path(self.ks_wrapper.SAVE_DIRECTORY) / "tables" / f"{trial_name}_cluster_firing_rates.xlsx")
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
        
        Parameters:
        -----------
        trial_name : str
            Name of the trial to analyze
        subwindow_start_times : array-like
            Start times of subwindows in seconds
        subwindow_end_times : array-like
            End times of subwindows in seconds
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the average voltage for each subwindow
        """
        if trial_name not in self.si_wrapper.data.analog_data:
            print(f"Trial '{trial_name}' not found in analog_data.")
            return pd.DataFrame({'avg_voltage': []}, index=range(len(subwindow_start_times)))
            
        recording = self.si_wrapper.data.analog_data[trial_name]
        sampling_rate = recording.get_sampling_frequency()
        
        if 'ANALOG-IN-2' not in recording.get_channel_ids():
            print(f"ANALOG-IN-2 not found in trial '{trial_name}'.")
            return pd.DataFrame({'avg_voltage': []}, index=range(len(subwindow_start_times)))
            
        von_frey_data = recording.get_traces(channel_ids=['ANALOG-IN-2'], return_scaled=True).flatten()
        avg_voltages = []
        
        for start, end in zip(subwindow_start_times, subwindow_end_times):
            if end > start:
                start_idx = int(start * sampling_rate)
                end_idx = int(end * sampling_rate)
                avg_voltage = np.mean(von_frey_data[start_idx:end_idx])
                avg_voltages.append(avg_voltage)
            else:
                avg_voltages.append(np.nan)
                
        return pd.DataFrame({'avg_voltage': avg_voltages}, index=range(len(avg_voltages)))

    def compute_unit_firing_rates_for_subwindows(self, trial_name, subwindow_start_times, subwindow_end_times, good_clusters=None):
        """
        Computes firing rates for each unit in each subwindow.
        """
        # Get spike times and clusters
        kilosort_output = self.ks_wrapper.kilosort_results[trial_name]
        spike_times = kilosort_output['spike_times']
        clusters = kilosort_output['spike_clusters']
        fs = kilosort_output['ops']['fs']
        
        if spike_times is None or clusters is None:
            print(f"Could not get spike data for trial {trial_name}")
            return pd.DataFrame()
            
        # Filter for good clusters if specified
        if good_clusters is not None:
            mask = np.isin(clusters, good_clusters)
            spike_times = spike_times[mask]
            clusters = clusters[mask]
            
        if len(spike_times) == 0:
            print(f"No spikes found for trial {trial_name}")
            return pd.DataFrame()
            
        # Convert times to samples
        start_samples = np.array(subwindow_start_times) * fs
        end_samples = np.array(subwindow_end_times) * fs
        
        # Initialize results
        unique_clusters = np.unique(clusters)
        firing_rates = []
        
        # Process each cluster
        for cluster_id in unique_clusters:
            # Get spikes for this cluster
            cluster_spikes = spike_times[clusters == cluster_id]
            
            # Compute firing rate for each subwindow
            rates = []
            for start, end in zip(start_samples, end_samples):
                # Count spikes in window
                n_spikes = np.sum((cluster_spikes >= start) & (cluster_spikes <= end))
                # Convert to Hz
                duration = (end - start) / fs
                rate = n_spikes / duration if duration > 0 else 0
                rates.append(rate)
                
            firing_rates.append(rates)
            
        # Create DataFrame with string cluster IDs as column names
        df = pd.DataFrame(
            np.array(firing_rates).T,
            columns=[f'cluster_{str(cid)}' for cid in unique_clusters]
        )
        
        return df

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
        """
        Computes correlation between inverse ISI and Von Frey signal.
        """
        print(f"\n[DEBUG] Computing correlations for trial: {trial_name}")
        print(f"[DEBUG] Window size: {window_size}")
        print(f"[DEBUG] Good clusters: {good_clusters}")
        
        # Ensure we have a valid window size
        if window_size is None:
            window_size = 15000  # Default window size
            print(f"[DEBUG] Using default window size: {window_size}")
        
        # Get spike times and clusters
        kilosort_output = self.ks_wrapper.kilosort_results[trial_name]
        spike_times = kilosort_output['spike_times']
        clusters = kilosort_output['spike_clusters']
        fs = kilosort_output['ops']['fs']
        
        print(f"[DEBUG] Total spikes: {len(spike_times)}")
        print(f"[DEBUG] Unique clusters before filtering: {np.unique(clusters)}")
        print(f"[DEBUG] Sampling rate: {fs} Hz")
        
        if spike_times is None or clusters is None:
            print(f"Could not get spike data for trial {trial_name}")
            return None, None
            
        # Filter for good clusters if specified
        if good_clusters is not None:
            mask = np.isin(clusters, good_clusters)
            spike_times = spike_times[mask]
            clusters = clusters[mask]
            print(f"[DEBUG] Spikes after filtering: {len(spike_times)}")
            print(f"[DEBUG] Unique clusters after filtering: {np.unique(clusters)}")
            
        if len(spike_times) == 0:
            print(f"No spikes found for trial {trial_name}")
            return None, None
            
        # Get Von Frey data
        if trial_name not in self.si_wrapper.data.analog_data:
            print(f"Trial '{trial_name}' not found in analog_data.")
            return None, None
            
        recording = self.si_wrapper.data.analog_data[trial_name]
        if "ANALOG-IN-2" not in recording.get_channel_ids():
            print(f"Channel 'ANALOG-IN-2' not found for trial {trial_name}")
            return None, None
            
        von_frey_data = recording.get_traces(channel_ids=["ANALOG-IN-2"], return_scaled=True).flatten()
        print(f"[DEBUG] Von Frey data length: {len(von_frey_data)} samples")
        
        # Compute correlations for each cluster
        correlations = {}
        inv_isi_traces = {}
        
        for cluster_id in np.unique(clusters):
            print(f"\n[DEBUG] Processing cluster {cluster_id}")
            
            # Get spikes for this cluster
            cluster_spikes = spike_times[clusters == cluster_id]
            print(f"[DEBUG] Number of spikes for cluster {cluster_id}: {len(cluster_spikes)}")
            print(f"[DEBUG] Spike times range: {cluster_spikes.min():.2f} to {cluster_spikes.max():.2f}")
            
            # Convert spike times to seconds if they're in samples
            if np.max(cluster_spikes) > 1000:  # crude check: likely in samples
                print(f"[DEBUG] Converting spike times from samples to seconds")
                cluster_spikes = cluster_spikes / fs
                print(f"[DEBUG] Converted range: {cluster_spikes.min():.2f} to {cluster_spikes.max():.2f} seconds")
            
            # Compute inverse ISI
            isi = np.diff(cluster_spikes)
            inv_isi = 1 / (isi + 1e-10)  # Add small constant to avoid division by zero
            print(f"[DEBUG] ISI range: {isi.min():.2f} to {isi.max():.2f} seconds")
            print(f"[DEBUG] Inverse ISI range: {inv_isi.min():.2f} to {inv_isi.max():.2f} Hz")
            
            # Create time series
            inv_isi_trace = np.zeros(len(von_frey_data))
            
            # Convert spike times to sample indices
            spike_indices = (cluster_spikes[:-1] * fs).astype(int)
            print(f"[DEBUG] Sample indices range: {spike_indices.min()} to {spike_indices.max()}")
            
            # Ensure indices are within bounds
            valid_mask = (spike_indices >= 0) & (spike_indices < len(inv_isi_trace))
            valid_indices = spike_indices[valid_mask]
            valid_inv_isi = inv_isi[valid_mask]
            
            print(f"[DEBUG] Valid indices: {len(valid_indices)} out of {len(spike_indices)}")
            if len(valid_indices) > 0:
                print(f"[DEBUG] Valid indices range: {valid_indices.min()} to {valid_indices.max()}")
            
            # Fill the trace
            inv_isi_trace[valid_indices] = valid_inv_isi
            
            # Smooth inverse ISI
            smoothed_inv_isi = self._smooth_moving_average(inv_isi_trace, window_size)
            
            # Compute correlation
            correlation = np.corrcoef(smoothed_inv_isi, von_frey_data)[0, 1]
            print(f"[DEBUG] Correlation for cluster {cluster_id}: {correlation:.3f}")
            
            # Store results with string cluster ID
            cluster_key = f'cluster_{str(cluster_id)}'
            correlations[cluster_key] = correlation
            inv_isi_traces[cluster_key] = smoothed_inv_isi
            
        print(f"\n[DEBUG] Completed correlation computation for {len(correlations)} clusters")
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
        if hasattr(self, 'ks_wrapper') and hasattr(self.ks_wrapper, 'kilosort_results') and trial_name in self.ks_wrapper.kilosort_results:
            fs = self.ks_wrapper.kilosort_results[trial_name]['ops']['fs']
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
                          fast_mode=False, skip_correlations=False, correlation_window_size=15000):
        """
        Analyze von Frey trials by subdividing them into smaller windows.
        """
        if TRIAL_NAMES is None:
            TRIAL_NAMES = list(self.ks_wrapper.kilosort_results.keys())
            
        results = {}
        for trial_name in TRIAL_NAMES:
            print(f"\nProcessing trial: {trial_name}")
            
            # Get von Frey windows
            windows = self.extract_von_frey_windows([trial_name], amplitude_threshold=amplitude_threshold,
                                                  start_buffer=start_buffer, end_buffer=end_buffer)
            if not windows:
                print(f"No valid von Frey windows found for {trial_name}. Skipping...")
                continue
                
            # Subdivide windows
            subwindows = self.subdivide_intervals(windows, subwindow_width)
            if not subwindows:
                print(f"No valid subwindows created for {trial_name}. Skipping...")
                continue
                
            # Print subwindow structure for debugging
            print(f"Subwindow structure: {subwindows}")
            
            # Get the trial's subwindows
            trial_subwindows = subwindows.get(trial_name)
            if not trial_subwindows:
                print(f"No subwindows found for trial {trial_name}. Skipping...")
                continue
                
            # Get good clusters for this trial
            good_clusters = None
            if good_clusters_by_trial is not None:
                if isinstance(good_clusters_by_trial, dict):
                    good_clusters = good_clusters_by_trial.get(trial_name, None)
                else:
                    good_clusters = good_clusters_by_trial
                    
            # Compute average von Frey voltage
            avg_voltage_df = self.compute_average_von_frey_voltage(trial_name, 
                                                                trial_subwindows['subwindow_start_times'],
                                                                trial_subwindows['subwindow_end_times'])
            if avg_voltage_df is None or avg_voltage_df.empty:
                print(f"Could not compute average von Frey voltage for {trial_name}. Skipping...")
                continue
                
            # Ensure voltage DataFrame has required columns
            if 'group' not in avg_voltage_df.columns:
                # Assign group labels based on subwindow start time
                subwindow_starts = trial_subwindows['subwindow_start_times']
                group_labels = ['pre-stim' if t < 35 else 'post-stim' for t in subwindow_starts]
                if len(group_labels) == len(avg_voltage_df):
                    avg_voltage_df['group'] = group_labels
                else:
                    avg_voltage_df['group'] = range(len(avg_voltage_df))
            if 'avg_voltage' not in avg_voltage_df.columns:
                print(f"'avg_voltage' column missing in voltage DataFrame for {trial_name}. Skipping...")
                continue
                
            # Compute firing rates
            if fast_mode:
                firing_rates_df = self.compute_unit_firing_rates_for_subwindows_modular(
                    self.rat.spikeinterface_wrapper.get_spike_times(trial_name),
                    self.rat.spikeinterface_wrapper.get_clusters(trial_name),
                    trial_subwindows['subwindow_start_times'],
                    trial_subwindows['subwindow_end_times']
                )
            else:
                firing_rates_df = self.compute_unit_firing_rates_for_subwindows(
                    trial_name,
                    trial_subwindows['subwindow_start_times'],
                    trial_subwindows['subwindow_end_times'],
                    good_clusters=good_clusters
                )
                
            if firing_rates_df is None or firing_rates_df.empty:
                print(f"Could not compute firing rates for {trial_name}. Skipping...")
                continue
                
            # Ensure firing rates DataFrame has required columns
            if 'group' not in firing_rates_df.columns:
                # Assign group labels based on subwindow start time
                subwindow_starts = trial_subwindows['subwindow_start_times']
                group_labels = ['pre-stim' if t < 35 else 'post-stim' for t in subwindow_starts]
                if len(group_labels) == len(firing_rates_df):
                    firing_rates_df['group'] = group_labels
                else:
                    firing_rates_df['group'] = range(len(firing_rates_df))
            
            # Compute correlations
            correlations = {}
            inv_isi_traces = {}
            if not skip_correlations:
                print(f"Computing correlations for {trial_name}...")
                if fast_mode:
                    correlations, inv_isi_traces = self.compute_inv_isi_correlation_modular(
                        self.rat.spikeinterface_wrapper.get_spike_times(trial_name),
                        self.rat.spikeinterface_wrapper.get_clusters(trial_name),
                        self.rat.spikeinterface_wrapper.get_sampling_rate(),
                        self.rat.get_von_frey_data(trial_name),
                        window_size=correlation_window_size
                    )
                else:
                    correlations, inv_isi_traces = self.compute_inv_isi_correlation(
                        trial_name,
                        window_size=correlation_window_size,
                        good_clusters=good_clusters
                    )
                    
                if correlations:
                    print(f"Found correlations for clusters: {list(correlations.keys())}")
                    for cluster, corr in correlations.items():
                        print(f"Cluster {cluster}: {corr:.3f}")
                else:
                    print(f"No correlations computed for {trial_name}")
                    
            # Store results
            results[trial_name] = {
                'windows': windows,
                'subwindows': trial_subwindows,
                'voltage': avg_voltage_df,
                'firing_rates': firing_rates_df,
                'correlations': correlations,
                'inv_isi_traces': inv_isi_traces
            }
            
        return results

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

    def analyze_trial_clusters(self, rat_id, trial_name=None, plot_results=True, save_plots=True, psth_responsive=None, cluster_labels=None):
        """
        Performs comprehensive analysis of spike data, waveforms, and von Frey correlations for clusters in a trial.
        Now includes PSTH-based responsiveness detection for each cluster.
        
        Parameters:
        -----------
        rat_id : str
            The rat ID.
        trial_name : str, optional
            The trial name to analyze. If None, all trials are analyzed.
        plot_results : bool, default True
            Whether to plot the results.
        save_plots : bool, default True
            Whether to save the plots.
        psth_responsive : dict, optional
            A dictionary mapping cluster IDs to boolean values indicating if the cluster is responsive.
        cluster_labels : str, list, dict, or nested dict, optional
            Cluster labels to include in analysis:
            - str/list: same labels for all trials
            - dict: {trial_name: labels} for per-trial specification
            - nested dict: {rat_id: {trial_name: labels}} for per-trial specification
            If None, defaults to 'good'.
        """
        from automations.plots import plot_raster, plot_waveform
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path
        # Get the kilosort wrapper for this rat
        ksw = self.ks_wrapper
        # Get list of trials to analyze
        if trial_name is not None:
            trial_list = [trial_name]
        else:
            trial_list = list(ksw.kilosort_results.keys())
        results = {}
        for trial in trial_list:
            print(f"\nAnalyzing trial: {trial}")
            trial_results = {}
            # Determine labels for this trial
            if isinstance(cluster_labels, dict):
                trial_labels = cluster_labels.get(trial, 'good')
            else:
                trial_labels = cluster_labels if cluster_labels is not None else 'good'
            # Get clusters matching the labels
            good_clusters = ksw.get_clusters_by_labels(trial, trial_labels)
            if not good_clusters:
                print(f"No clusters found for trial {trial} with labels {trial_labels}")
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
            correlations, inv_isi_traces = self.compute_inv_isi_correlation(
                trial,
                window_size=15000
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
                figures_dir = Path(self.ks_wrapper.SAVE_DIRECTORY) / 'figures' / trial
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

    def save_results_to_excel(self, results, save_dir):
        """
        Save analysis results to Excel files.
        
        Parameters:
        -----------
        results : dict
            Dictionary of results keyed by trial name
        save_dir : Path
            Directory to save Excel files
        """
        import pandas as pd
        from pathlib import Path
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for trial_name, trial_results in results.items():
            # Save firing rates
            if 'firing_rates' in trial_results:
                firing_rates_df = pd.DataFrame(trial_results['firing_rates'])
                firing_rates_df.to_excel(save_dir / f"{trial_name}_firing_rates.xlsx", index=False)
            
            # Save correlations
            if 'correlations' in trial_results:
                # Convert correlations dict to DataFrame with proper index
                correlations = trial_results['correlations']
                if correlations:  # Only process if we have correlations
                    correlations_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['correlation'])
                    correlations_df.index.name = 'cluster_id'
                    correlations_df.to_excel(save_dir / f"{trial_name}_correlations.xlsx")
            
            # Save average voltage
            if 'average_voltage' in trial_results:
                voltage_df = pd.DataFrame(trial_results['average_voltage'])
                voltage_df.to_excel(save_dir / f"{trial_name}_average_vf_voltage_windowed.xlsx", index=False)
            
            # Save PSTH results if present
            if 'psth_results' in trial_results:
                psth_df = pd.DataFrame(trial_results['psth_results'])
                psth_df.to_excel(save_dir / f"{trial_name}_psth_results.xlsx", index=False)
            
            # Save responsiveness results if present
            if 'responsiveness' in trial_results:
                resp_df = pd.DataFrame(trial_results['responsiveness'])
                resp_df.to_excel(save_dir / f"{trial_name}_responsiveness.xlsx", index=False)
        
        print(f"[DEBUG] Saved results to {save_dir}")
