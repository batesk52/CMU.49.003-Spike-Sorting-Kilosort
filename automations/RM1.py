"""
Below is a refactored version of your RM1.py file. The changes include:

Streamlining Data Import:
Instead of five separate dictionaries for Intan recordings, all streams are stored in a single dictionary (keyed by stream ID), reducing code duplication in the import routine.

Helper Function for Window Removal:
A private method (_remove_window) was added so that the code for slicing and concatenating recordings is shared between spinal cord and analog data removal.

Improved Exception Handling and Use of Pathlib:
File/directory operations are now handled via pathlib where possible, and try/except blocks now report the encountered exceptions.

Meta Class for Multiple Rats:
A new class, RatGroup, aggregates multiple Rat objects, allowing you to run preprocessing across all rats and create wrappers for SpikeInterface and Kilosort. This class also stores each rat's ID (which will be useful for subsequent mixed-model analyses).

Minor Simplifications in the Wrappers:
The wrappers (both SpikeInterface and Kilosort) have been slightly streamlined (for example, using Path objects for directories).

"""


import os
from pathlib import Path
import scipy.io as sio
from spikeinterface.extractors import read_intan
import pandas as pd
import numpy as np
from kilosort import io
from kilosort import run_kilosort, DEFAULT_SETTINGS
from kilosort.io import load_ops
import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams
from spikeinterface import concatenate_recordings
from kilosort.run_kilosort import load_sorting
from kilosort.data_tools import (
    mean_waveform, cluster_templates, get_good_cluster, get_cluster_spikes,
    get_spike_waveforms, get_best_channel
    )
import shutil
import spikeinterface.preprocessing as spre

class Rat:
    """
    Processes and stores data from rat experiments.
    """
    def __init__(self, DATA_DIRECTORY, PROBE_DIRECTORY, RAT_ID, **kwargs):
        self.RAT_ID = RAT_ID
        self.PROBE_DIRECTORY = Path(PROBE_DIRECTORY)
        self.DATA_DIRECTORY = Path(DATA_DIRECTORY)
        # Consolidate intan recordings from different streams into one dict
        self.intan_recordings = {stream_id: {} for stream_id in range(5)}
        self.emg_data, self.nerve_cuff_data, self.sc_data = {}, {}, {}
        self.analog_data = {}
        self.st_experiment_notes = None
        self.qst_experiment_notes = None
        self.drgs_experiment_notes = None
        self.sc_experiment_notes = None
        self.st_trial_notes = None
        self.qst_trial_notes = None
        self.drgs_trial_notes = None
        self.sc_trial_notes = None
        self.mat_files_dict = {}
        self.trial_names = []
        
        self.import_metadata()
        self.import_matlab_files()
        self.import_intan_data()

    def get_von_frey_analysis(self, si_wrapper, ks_wrapper, excel_parent_folder=None, cluster_types='good', **kwargs):
        """
        Computes or loads the VonFrey analysis for this rat.
        
        If excel_parent_folder is provided, this method checks for a subfolder:
        {excel_parent_folder}/{RatID}/tables
        and if Excel files matching the naming pattern are found, it loads those results.
        Otherwise, it runs the analysis.
        
        Parameters:
        si_wrapper: The SpikeInterface_wrapper for this rat.
        ks_wrapper: The Kilosort_wrapper for this rat.
        excel_parent_folder: str or Path; the parent folder under which each rat's results
                            (in a subfolder named by RatID with a "tables" folder) might exist.
        cluster_types: str; specifies which cluster types to analyze. Options:
                      - 'good': only good clusters (default)
                      - 'mua': only MUA clusters  
                      - 'both': both good and MUA clusters
        kwargs: additional parameters for the analysis (e.g., subwindow_width, corr_threshold).
        
        Returns:
        results: dict keyed by trial names.
        """
        from pathlib import Path
        if excel_parent_folder is not None:
            candidate = Path(excel_parent_folder) / self.RAT_ID / "tables"
            if candidate.exists() and any(candidate.glob("*_average_vf_voltage_windowed.xlsx")):
                return self.load_von_frey_analysis_results(candidate)
                
        # Get clusters for each trial based on specified cluster types
        clusters_by_trial = {}
        for trial in ks_wrapper.kilosort_results.keys():
            clusters_for_trial = []
            
            if cluster_types in ['good', 'both']:
                good_clusters = ks_wrapper.get_good_clusters_from_group(trial)
                clusters_for_trial.extend(good_clusters)
                
            if cluster_types in ['mua', 'both']:
                mua_clusters = ks_wrapper.get_mua_clusters_from_group(trial)
                clusters_for_trial.extend(mua_clusters)
            
            # Remove duplicates (shouldn't be any, but just in case) and sort
            clusters_by_trial[trial] = sorted(list(set(clusters_for_trial)))
            
            print(f"Trial {trial}: Using {len(clusters_by_trial[trial])} clusters (type: {cluster_types})")
        
        # Remove the extra key so that analyze_subwindows() does not receive it.
        kwargs.pop('excel_parent_folder', None)
        kwargs.pop('cluster_types', None)  # Remove cluster_types from kwargs
        
        # Extract and preserve optimization parameters, then remove them from kwargs
        fast_mode = kwargs.pop('fast_mode', False)
        skip_correlations = kwargs.pop('skip_correlations', False) 
        correlation_window_size = kwargs.pop('correlation_window_size', None)
        
        # Show what optimization settings are being used
        print(f"[INFO] Using optimization settings: fast_mode={fast_mode}, skip_correlations={skip_correlations}")
        if correlation_window_size:
            print(f"[INFO] Custom correlation window size: {correlation_window_size}")
        
        from automations.analysis_functions import VonFreyAnalysis
        vfa = VonFreyAnalysis(self, si_wrapper, ks_wrapper)
        return vfa.analyze_subwindows(
            good_clusters_by_trial=clusters_by_trial, 
            fast_mode=fast_mode,
            skip_correlations=skip_correlations,
            correlation_window_size=correlation_window_size,
            **kwargs
        )


    def load_von_frey_analysis_results(self, excel_folder):
        """
        Loads precomputed VonFrey analysis results from Excel files.
        Assumes that excel_folder contains pairs of files for each trial:
          * {trial_name}_average_vf_voltage_windowed.xlsx 
          * {trial_name}_cluster_firing_rates_windowed.xlsx
        
        The method returns a dictionary keyed by trial name, where each value is
        a dictionary containing the loaded DataFrames.
        """
        results = {}
        folder = Path(excel_folder)
        # Look for all voltage files as indicator files.
        for file in folder.glob("*_average_vf_voltage_windowed.xlsx"):
            trial_name = file.stem.replace("_average_vf_voltage_windowed", "")
            voltage_file = folder / f"{trial_name}_average_vf_voltage_windowed.xlsx"
            firing_file = folder / f"{trial_name}_cluster_firing_rates_windowed.xlsx"
            if voltage_file.exists() and firing_file.exists():
                voltage_df = pd.read_excel(voltage_file)
                firing_df = pd.read_excel(firing_file)
                results[trial_name] = {"voltage_df": voltage_df, "firing_df": firing_df}
        return results
    
    def import_metadata(self):
        metadata_path = self.DATA_DIRECTORY / self.RAT_ID / f'{self.RAT_ID}_TermExpSchedule.xlsx'
        # Import metadata for different experiment types
        for sheet_name in ['ST', 'DRGS', 'SC', 'QST']:
            sheet = pd.read_excel(metadata_path, sheet_name=sheet_name)
            exp_note = sheet.iloc[4, 1]
            trial_notes = sheet.copy()
            trial_notes.columns = trial_notes.iloc[5, :]
            trial_notes.index = trial_notes["Trial Number"]
            if sheet_name == 'ST':
                self.st_experiment_notes = exp_note
                self.st_trial_notes = trial_notes.iloc[6:, :]
            elif sheet_name == 'DRGS':
                self.drgs_experiment_notes = exp_note
                self.drgs_trial_notes = trial_notes.iloc[6:, :]
            elif sheet_name == 'SC':
                self.sc_experiment_notes = exp_note
                self.sc_trial_notes = trial_notes.iloc[6:, :]
            elif sheet_name == 'QST':
                self.qst_experiment_notes = exp_note
                self.qst_trial_notes = trial_notes.iloc[6:, :]
    
    def import_matlab_files(self):
        rat_dir = self.DATA_DIRECTORY / self.RAT_ID / self.RAT_ID
        for file in rat_dir.glob("*.mat"):
            mat_data = sio.loadmat(file)
            self.mat_files_dict[file.stem] = mat_data
    
    def import_intan_data(self):
        data_path = self.DATA_DIRECTORY / self.RAT_ID / self.RAT_ID
        for folder in data_path.iterdir():
            if folder.is_dir():
                self.trial_names.append(folder.name)
                rhd_file = folder / f"{folder.name}.rhd"
                if rhd_file.exists():
                    print(f"Reading {folder.name}...")
                    # Loop over stream IDs (0-4) to reduce redundancy
                    for stream_id in range(5):
                        try:
                            rec = read_intan(str(rhd_file), stream_id=str(stream_id))
                            self.intan_recordings[stream_id][folder.name] = rec
                        except Exception as e:
                            print(f"Error reading stream {stream_id} for {folder.name}: {e}")
                            continue
    
    def get_sc_data(self):
        for name, rec in self.intan_recordings[0].items():
            try:
                self.sc_data[name] = rec.remove_channels(["B-000", "B-001"])
            except Exception as e:
                self.sc_data[name] = None
                print(f"Error processing SC data for {name}: {e}")
    
    def get_nerve_cuff_data(self):
        for name, rec in self.intan_recordings[0].items():
            try:
                self.nerve_cuff_data[name] = rec.select_channels(["B-000"])
            except Exception as e:
                self.nerve_cuff_data[name] = None
                print(f"Error processing nerve cuff data for {name}: {e}")
    
    def get_emg_data(self):
        for name, rec in self.intan_recordings[0].items():
            try:
                self.emg_data[name] = rec.select_channels(["B-001"])
            except Exception as e:
                self.emg_data[name] = None
                print(f"Error processing EMG data for {name}: {e}")
    
    def get_analog_data(self):
        for name, rec in self.intan_recordings[3].items():
            self.analog_data[name] = rec
    
    def slice_and_concatenate_recording(self, base_recording, first_window=(0, 1050000), last_window=None):
        recording_first = base_recording.frame_slice(start_frame=first_window[0], end_frame=first_window[1])
        total_frames = base_recording.get_num_frames()
        if last_window is None:
            last_window = (total_frames - 1050000, total_frames)
        else:
            last_window = (last_window[0], min(last_window[1], total_frames))
        recording_last = base_recording.frame_slice(start_frame=last_window[0], end_frame=last_window[1])
        recording_combined = concatenate_recordings([recording_first, recording_last])
        return recording_combined
    
    def _remove_window(self, data_dict, first_window, last_window):
        processed = {}
        for name, rec in data_dict.items():
            if rec is not None:
                try:
                    processed[name] = self.slice_and_concatenate_recording(rec, first_window, last_window)
                except Exception as e:
                    print(f"Error processing {name}: {e}")
                    processed[name] = None
            else:
                print(f"No recording found for {name}")
                processed[name] = None
        return processed

    def remove_drg_stim_window_sc(self, first_window=(0, 1050000), last_window=None):
        self.sc_data = self._remove_window(self.sc_data, first_window, last_window)
    
    def remove_drg_stim_window_analog(self, first_window=(0, 1050000), last_window=None):
        self.analog_data = self._remove_window(self.analog_data, first_window, last_window)


class SpikeInterface_wrapper:
    def __init__(self, rat_instance, SAVE_DIRECTORY):
        self.RAT_ID = rat_instance.RAT_ID
        self.data = rat_instance
        self.PROBE_DIRECTORY = rat_instance.PROBE_DIRECTORY
        self.DATA_DIRECTORY = rat_instance.DATA_DIRECTORY
        self.SAVE_DIRECTORY = Path(SAVE_DIRECTORY)
        self.kilosort_results = {}
        print(f"Preparing SpikeInterface wrapper for rat {self.RAT_ID}")

    def save_spinalcord_data_to_binary(self, TRIAL_NAMES=None, bandpass_freq_min=300, bandpass_freq_max=3000, notch_freq=False):
        trial_list = TRIAL_NAMES if TRIAL_NAMES is not None else self.data.sc_data.keys()
        for recording in trial_list:
            try: # check if recording works
                rec = self.data.sc_data[recording]

                ### apply filtering
                # apply a bandpass filter
                recording_filtered = spre.bandpass_filter(rec, freq_min=bandpass_freq_min, freq_max=bandpass_freq_max)
                if notch_freq != False:
                    # apply a notch filter at 60Hz, the frequency of the power line in US (this comes from Wang et al., 2024)
                    recording_filtered = spre.notch_filter(recording_filtered, freq=notch_freq)
                # apply a global common reference
                recording_preprocessed = spre.common_reference(recording_filtered, reference='global', operator='average')
                dtype = np.int16
                filename, N, c, s, fs, probe_path = io.spikeinterface_to_binary(
                    recording_preprocessed, 
                    str(self.SAVE_DIRECTORY / 'binary' / recording), 
                    data_name=f'{self.RAT_ID}_{recording}_data.bin', 
                    dtype=dtype,
                    chunksize=60000, 
                    export_probe=False, 
                    probe_name=str(self.PROBE_DIRECTORY)
                )
                print(f"Data saved to {filename}")
            except Exception as e:
                print(f"ERROR: issue importing data for {recording}: {e}")
    
    def export_raw_spikes_and_von_frey_all_trials(self, kilosort_wrapper_instance):
        tables_dir = self.SAVE_DIRECTORY / 'tables'
        tables_dir.mkdir(parents=True, exist_ok=True)
        for trial_name, ks_data in kilosort_wrapper_instance.kilosort_results.items():
            spike_times = ks_data['spike_times']
            spike_clusters = ks_data['spike_clusters']
            spike_csv_path = tables_dir / f"{trial_name}_spikes.csv"
            pd.DataFrame({'spike_times': spike_times, 'spike_clusters': spike_clusters}).to_csv(spike_csv_path, index=False)
            print(f"Spike data exported for trial {trial_name} to {spike_csv_path}")
            if trial_name in self.data.analog_data and self.data.analog_data[trial_name] is not None:
                von_frey_recording = self.data.analog_data[trial_name]
                channel_ids = von_frey_recording.get_channel_ids()
                if 'ANALOG-IN-2' in channel_ids:
                    try:
                        von_frey_data = von_frey_recording.get_traces(channel_ids=['ANALOG-IN-2'], return_scaled=True).flatten()
                        sampling_rate = von_frey_recording.get_sampling_frequency()
                        von_frey_csv_path = tables_dir / f"{trial_name}_von_frey.csv"
                        pd.DataFrame({'von_frey_voltage': von_frey_data}).to_csv(von_frey_csv_path, index=False)
                        with open(von_frey_csv_path, 'a') as f:
                            f.write(f"# sampling_rate: {sampling_rate}\n")
                        print(f"Von Frey data exported for trial {trial_name} to {von_frey_csv_path}")
                    except Exception as e:
                        print(f"Error processing Von Frey data for trial {trial_name}: {e}")
                else:
                    print(f"ANALOG-IN-2 channel not found for trial {trial_name}. No Von Frey data exported.")
            else:
                print(f"No analog data found for trial {trial_name}. No Von Frey data exported.")

class Kilosort_wrapper:
    def __init__(self, SAVE_DIRECTORY, PROBE_DIRECTORY):
        self.PROBE_DIRECTORY = Path(PROBE_DIRECTORY)
        self.SAVE_DIRECTORY = Path(SAVE_DIRECTORY)
        self.kilosort_results = {}
        print(f"Preparing Kilosort wrapper for {self.SAVE_DIRECTORY.name[-5:]}")
    
    def run_or_load_kilosort(self, trial_name, run_kilosort=False):
        if run_kilosort:
            self.run_kilosort_trial_summary()
        else:
            self.extract_kilosort_outputs(trial_name)
    
    def apply_custom_labels_to_trial(self, trial_name, custom_criteria=None):
        try:
            results_dir = self.SAVE_DIRECTORY / 'binary' / trial_name / 'kilosort4'
            if not results_dir.exists():
                print(f"Kilosort directory not found for trial: {trial_name}")
                return
            ops, st, clu, similar_templates, is_ref, est_contam_rate, kept_spikes = load_sorting(results_dir)
            cluster_labels = np.unique(clu)
            fs = ops['fs']
            label_good = is_ref.copy()
            if custom_criteria is not None:
                label_good = np.logical_and(label_good, custom_criteria(cluster_labels, st, clu, est_contam_rate, fs))
            else:
                contam_good = est_contam_rate < 0.2
                fr_good = np.array([ (st[clu==c].size / ((st[clu==c].max() - st[clu==c].min())/fs)) >= 1 for c in cluster_labels ])
                label_good = np.logical_and(label_good, contam_good, fr_good)
            ks_labels = ['good' if b else 'mua' for b in label_good]
            save_1 = results_dir / 'cluster_KSLabel.tsv'
            save_2 = results_dir / 'cluster_group.tsv'
            backup = results_dir / 'cluster_KSLabel_backup.tsv'
            if not backup.exists():
                shutil.copyfile(save_1, backup)
            with open(save_1, 'w') as f:
                f.write("cluster_id\tKSLabel\n")
                for i, label in enumerate(ks_labels):
                    f.write(f"{i}\t{label}\n")
            shutil.copyfile(save_1, save_2)
            print(f"Custom labels applied and saved for trial: {trial_name}")
        except Exception as e:
            print(f"Error applying custom labels to trial {trial_name}: {e}")
    
    def run_kilosort_trial_summary(self, new_settings=None, custom_criteria=None, **kwargs):
        binary_folder = self.SAVE_DIRECTORY
        folders_with_bin = [Path(root) for root, dirs, files in os.walk(binary_folder) if any(f.endswith('.bin') for f in files)]
        for folder in folders_with_bin:
            print(f"\nRunning kilosort on {folder.name}\n")
            try:
                if new_settings is None:
                    settings = {'data_dir': str(self.SAVE_DIRECTORY / 'binary' / folder.name), 'n_chan_bin': 32}
                elif new_settings == "vf_settings":
                    settings = {'data_dir': str(self.SAVE_DIRECTORY / 'binary' / folder.name), 'n_chan_bin': 32, 'nblocks': 0, "batch_size": 1500000}
                elif new_settings == "vf_settings_flex_probe": # kilosort docs: "for < 32 channels, drift correction will be inaccurate, so disable with n=0"
                    settings = {'data_dir': str(self.SAVE_DIRECTORY / 'binary' / folder.name), 'n_chan_bin': 32, 'nblocks': 5, "batch_size": 1500000}
                elif new_settings == "70s batch, flex": # kilosort docs: "nblocks=5 can be a good choice for single-shank Neuropixels probes"
                    settings = {'data_dir': str(self.SAVE_DIRECTORY / 'binary' / folder.name), 'n_chan_bin': 32, 'nblocks': 5, "batch_size": 2100000}
                ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = run_kilosort(
                    settings=settings, probe_name=str(self.PROBE_DIRECTORY)
                )
                results_dir = self.SAVE_DIRECTORY / 'binary' / folder.name / 'kilosort4'
                ops = np.load(results_dir / 'ops.npy', allow_pickle=True).item()
                camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values
                contam_pct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values
                min_len = min(len(camps), len(contam_pct))
                camps = camps[:min_len]
                contam_pct = contam_pct[:min_len]
                N_clusters = min_len
                all_clusters = np.arange(N_clusters)
                st = np.load(results_dir / 'spike_times.npy')
                clu = np.load(results_dir / 'spike_clusters.npy')
                firing_rates = np.array([(clu == c).sum() * 30000 / st.max() for c in all_clusters])
                assert len(firing_rates) == len(camps) == len(contam_pct), f"Mismatch: firing_rates={len(firing_rates)}, camps={len(camps)}, contam_pct={len(contam_pct)}"
                chan_map = np.load(results_dir / 'channel_map.npy')
                templates = np.load(results_dir / 'templates.npy')
                chan_best = chan_map[(templates**2).sum(axis=1).argmax(axis=-1)]
                amplitudes = np.load(results_dir / 'amplitudes.npy')
                dshift = ops['dshift']
                self.kilosort_results[folder.name] = {
                    'ops': ops,
                    'cluster_amplitudes': camps,
                    'contamination_percentage': contam_pct,
                    'channel_mapping': chan_map,
                    'templates': templates,
                    'chan_best': chan_best,
                    'amplitudes': amplitudes,
                    'spike_times': st,
                    'spike_clusters': clu,
                    'firing_rates': firing_rates,
                    'dshift': dshift
                }
            except Exception as e:
                print(f"Error processing folder {folder.name}: {e}")
                continue
            self.apply_custom_labels_to_trial(folder.name, custom_criteria=custom_criteria)
    
    def extract_kilosort_outputs(self):
        """
        Extracts and organizes Kilosort4 output files for each trial into a structured dictionary.
        
        The following files are loaded for each trial:
        - ops.npy: Contains all Kilosort configuration parameters and runtime information
        - cluster_Amplitude.tsv: Average spike amplitude for each neural cluster
        - cluster_ContamPct.tsv: Estimated contamination percentage for each cluster
        - spike_times.npy: Timestamps of all detected spikes (in samples)
        - spike_clusters.npy: Cluster ID assignments for each spike
        - channel_map.npy: Mapping of channel indices to physical probe locations
        - templates.npy: Average spike waveform templates for each cluster
        - amplitudes.npy: Scaling factors for each spike relative to its template

        The data is stored in self.kilosort_results[trial_name] as a dictionary.
        """
        binary_dir = self.SAVE_DIRECTORY / 'binary'
        trial_folders = [f for f in binary_dir.iterdir() if f.is_dir()]
        if not trial_folders:
            print("No trial folders found.")
            return
        for trial_folder in trial_folders:
            try:
                trial_name = trial_folder.name
                results_dir = trial_folder / 'kilosort4'
                if not results_dir.exists():
                    print(f"Kilosort directory not found for trial: {trial_name}")
                    continue
                
                # Load Kilosort configuration and runtime info
                ops = np.load(results_dir / 'ops.npy', allow_pickle=True).item()
                
                # Load cluster quality metrics
                camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values  # Spike amplitudes per cluster
                contam_pct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values  # Contamination % per cluster
                
                # Ensure consistent array lengths
                min_len = min(len(camps), len(contam_pct))
                camps = camps[:min_len]
                contam_pct = contam_pct[:min_len]
                N_clusters = min_len
                all_clusters = np.arange(N_clusters)
                
                # Load spike timing data
                st = np.load(results_dir / 'spike_times.npy')  # Timestamps in samples (30kHz sampling rate)
                clu = np.load(results_dir / 'spike_clusters.npy')  # Cluster assignments for each spike
                
                # Calculate firing rates (Hz) for each cluster
                firing_rates = np.array([(clu == c).sum() * 30000 / st.max() for c in all_clusters])
                assert len(firing_rates) == len(camps) == len(contam_pct), f"Mismatch: firing_rates={len(firing_rates)}, camps={len(camps)}, contam_pct={len(contam_pct)}"
                
                # Load channel and template data
                chan_map = np.load(results_dir / 'channel_map.npy')  # Maps channel indices to probe locations
                templates = np.load(results_dir / 'templates.npy')  # Average waveform template for each cluster
                chan_best = chan_map[(templates**2).sum(axis=1).argmax(axis=-1)]  # Best channel for each cluster
                amplitudes = np.load(results_dir / 'amplitudes.npy')  # Scaling factor for each spike
                dshift = ops['dshift']  # Estimated drift over time in micrometers
                
                # Store all extracted data in results dictionary
                self.kilosort_results[trial_name] = {
                    'ops': ops,
                    'cluster_amplitudes': camps,
                    'contamination_percentage': contam_pct,
                    'channel_mapping': chan_map,
                    'templates': templates,
                    'chan_best': chan_best,
                    'amplitudes': amplitudes,
                    'spike_times': st,
                    'spike_clusters': clu,
                    'firing_rates': firing_rates,
                    'dshift': dshift
                }
                print(f"Kilosort outputs successfully loaded for trial: {trial_name}")
            except Exception as trial_error:
                print(f"Error loading Kilosort outputs for trial {trial_folder.name}: {trial_error}")
                continue



    ##### new method

    def plot_trial_results(self, trial_names):
        """
        Accepts a list of trial names (or a single trial name as a string) and iteratively
        creates summary and waveform plots for each trial.
        """

        # Define the main figures directory (which is at the same level as "binary" and "tables")
        figures_dir = self.SAVE_DIRECTORY / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Allow a single trial name to be passed as a string.
        if not isinstance(trial_names, list):
            trial_names = [trial_names]
        
        for trial_name in trial_names:
            try:
                results_dir = self.SAVE_DIRECTORY / 'binary' / trial_name / 'kilosort4'
                if not results_dir.exists():
                    print(f"Results directory not found for trial: {trial_name}")
                    continue

                # Load Kilosort configuration and outputs
                # ops.npy contains all the parameters and settings used during the Kilosort run
                ops = np.load(results_dir / 'ops.npy', allow_pickle=True).item()
                # dshift tracks the estimated drift of neurons over time (in micrometers)
                dshift = ops['dshift']

                # Load cluster (neuron) quality metrics
                # cluster_Amplitude.tsv contains the average spike amplitude for each cluster
                camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values
                # cluster_ContamPct.tsv contains the estimated contamination % for each cluster
                # (how many spikes might be from other neurons)
                contam_pct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values

                # Ensure camps and contam_pct arrays have same length by truncating to shorter one
                min_len = min(len(camps), len(contam_pct))
                camps = camps[:min_len]
                contam_pct = contam_pct[:min_len]
                N_clusters = min_len  # Total number of neural clusters/units
                all_clusters = np.arange(N_clusters)

                # Load spike data
                # spike_times.npy contains timestamps of all detected spikes (in samples)
                st = np.load(results_dir / 'spike_times.npy')
                # spike_clusters.npy contains cluster ID for each spike
                clu = np.load(results_dir / 'spike_clusters.npy')

                # Calculate firing rates for each cluster (in Hz)
                # Formula: (number of spikes) * (sampling rate) / (total recording time)
                firing_rates = np.array([(clu == c).sum() * 30000 / st.max() for c in all_clusters])
                # Verify all metrics arrays have matching lengths
                assert len(firing_rates) == len(camps) == len(contam_pct), f"Mismatch: firing_rates={len(firing_rates)}, camps={len(camps)}, contam_pct={len(contam_pct)}"

                # Load channel and template information
                # channel_map.npy contains the mapping of channel indices to physical probe channels
                chan_map = np.load(results_dir / 'channel_map.npy')
                # templates.npy contains the average spike waveform for each cluster
                templates = np.load(results_dir / 'templates.npy')

                # Find the best (strongest signal) channel for each cluster
                # by finding channel with maximum energy in the template
                chan_best_idx = (templates**2).sum(axis=1).argmax(axis=-1)
                chan_best = chan_map[chan_best_idx]

                # amplitudes.npy contains the scaling factor for each spike relative to its template
                amplitudes = np.load(results_dir / 'amplitudes.npy')
                
                # Configure matplotlib style
                rcParams['axes.spines.top'] = False
                rcParams['axes.spines.right'] = False
                gray = 0.5 * np.ones(3)
                
                # Summary plots: 3x3 grid figure
                fig = plt.figure(figsize=(10,10), dpi=100)
                grid = plt.matplotlib.gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.5)
                
                # Drift plot
                ax = fig.add_subplot(grid[0,0])
                nbatches = ops.get('Nbatches')
                if nbatches is None or dshift is None:
                    ax.text(0.5, 0.5, "(drift disabled)", horizontalalignment='center', 
                            verticalalignment='center', transform=ax.transAxes)
                    ax.set_xlabel('time (sec.)')
                    ax.set_ylabel('drift (um)')
                else:
                    ax.plot(np.arange(0, nbatches) * 2, dshift)
                    ax.set_xlabel('time (sec.)')
                    ax.set_ylabel('drift (um)')
                
                # Spike scatter for first 5 sec.
                ax = fig.add_subplot(grid[0,1:])
                try:
                    t1 = np.nonzero(st > ops['fs'] * 5)[0][0]
                    valid_mask = clu[:t1] < len(chan_best)
                    ax.scatter(st[:t1][valid_mask] / 30000., chan_best[clu[:t1][valid_mask]], s=0.5, color='k', alpha=0.25)
                    ax.set_xlim([0, 5])
                    ax.set_ylim([chan_map.max(), 0])
                    ax.set_xlabel('time (sec.)')
                    ax.set_ylabel('channel')
                    ax.set_title('spikes from units')
                except IndexError as e:
                    print(f"Error plotting spike scatter for {trial_name}: {str(e)}")
                    print("Skipping spike scatter plot and continuing with remaining plots...")
                    ax.text(0.5, 0.5, "Error plotting spikes", horizontalalignment='center', 
                           verticalalignment='center', transform=ax.transAxes)
                
                # Histogram: firing rates
                ax = fig.add_subplot(grid[1,0])
                ax.hist(firing_rates, 20, color=gray)
                ax.set_xlabel('firing rate (Hz)')
                ax.set_ylabel('# of units')
                
                # Histogram: amplitude
                ax = fig.add_subplot(grid[1,1])
                ax.hist(camps, 20, color=gray)
                ax.set_xlabel('amplitude')
                ax.set_ylabel('# of units')
                
                # Histogram: contamination percentage
                ax = fig.add_subplot(grid[1,2])
                nb = ax.hist(np.minimum(100, contam_pct), np.arange(0,105,5), color=gray)
                ax.plot([10, 10], [0, nb[0].max()], 'k--')
                ax.set_xlabel('% contamination')
                ax.set_ylabel('# of units')
                ax.set_title('< 10% = good units')
                
                # Scatter plots: firing rate vs. amplitude (linear and log scales)
                for k in range(2):
                    ax = fig.add_subplot(grid[2,k])
                    is_good = contam_pct < 10.
                    ax.scatter(firing_rates[~is_good], camps[~is_good], s=3, color='r', label='mua', alpha=0.25)
                    ax.scatter(firing_rates[is_good], camps[is_good], s=3, color='b', label='good', alpha=0.25)
                    ax.set_xlabel('firing rate (Hz)')
                    ax.set_ylabel('amplitude (a.u.)')
                    ax.legend()
                    if k == 1:
                        ax.set_xscale('log')
                        ax.set_yscale('log')
                        ax.set_title('loglog')
            
                # Save the figure to the main figures folder with the desired filename
                save_path = figures_dir / f"{trial_name}_summmary_plots.png"
                fig.savefig(save_path, dpi=100, bbox_inches="tight")
                print(f"Saved plot for trial {trial_name}, at {save_path}")
                
                plt.show()

                
                # Waveform plots for good and mua units
                probe = ops['probe']
                xc, yc = probe['xc'], probe['yc']
                nc = 16  # channels to show around best channel
                groups = [('good', np.nonzero(contam_pct <= 0.1)[0]),
                        ('mua', np.nonzero(contam_pct > 0.1)[0])]
                
                for label, units in groups:
                    print(f'Plotting {label} units for trial: {trial_name}')
                    if len(units) == 0:
                        # Create a figure with a message if no units are available
                        fig = plt.figure(figsize=(6,2), dpi=150)
                        ax = fig.add_subplot(111)
                        ax.text(0.5, 0.5, f'No {label} units found', 
                                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                        ax.axis('off')
                        plt.show()
                        continue

                    # Determine the number of plots based on available unique units
                    n_plots = min(40, len(units))
                    # Sample without replacement so that each unit is only selected once
                    selected_units = np.random.choice(units, size=n_plots, replace=False)

                    # Adjust grid layout: maintain 2 rows and compute required number of columns.
                    import math
                    n_cols = math.ceil(n_plots / 2)
                    fig = plt.figure(figsize=(12, 3), dpi=150)
                    grid = plt.matplotlib.gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.25, wspace=0.5)
                    
                    for i, unit in enumerate(selected_units):
                        wv = templates[unit].copy()  # waveform for this unit
                        best_chan = chan_best[unit]
                        spike_count = (clu == unit).sum()
                        
                        # Determine subplot index based on the grid layout
                        row = i // n_cols
                        col = i % n_cols
                        ax = fig.add_subplot(grid[row, col])
                        
                        n_chan = wv.shape[-1]
                        ic0 = max(0, best_chan - nc//2)
                        ic1 = min(n_chan, best_chan + nc//2)
                        wv_crop = wv[:, ic0:ic1]
                        x0, y0 = xc[ic0:ic1], yc[ic0:ic1]
                        amp = 4
                        for xi, yi, waveform in zip(x0, y0, wv_crop.T):
                            t = np.arange(-wv_crop.shape[0]//2, wv_crop.shape[0]//2, dtype='float32')
                            t /= wv_crop.shape[0] / 20
                            ax.plot(xi + t, yi + waveform * amp, lw=0.5, color='k')
                        ax.set_title(f'{spike_count}', fontsize='small')
                        ax.axis('off')

                    # Save the figure to the main figures folder with the desired filename
                    save_path = figures_dir / f"{trial_name}_{label}_waveforms_on_probe.png"
                    fig.savefig(save_path, dpi=100, bbox_inches="tight")
                    print(f"Saved plot for trial {trial_name}, at {save_path}")

                    plt.show()

            except Exception as e:
                print(f"Error processing trial {trial_name}: {str(e)}")
                print("Moving to next trial...")
                continue



    def plot_cluster_waveforms(self, trial_names):
        """
        Accepts a list of trial names (or a single trial name as a string) and iteratively
        generates and saves a plot of the characteristic waveform for every "good" cluster in each trial.
        The plots are saved in the main "figures" folder (inside self.SAVE_DIRECTORY) with filenames
        that include the trial name and the cluster id.
        """
        # Allow a single trial name to be passed as a string.
        if not isinstance(trial_names, list):
            trial_names = [trial_names]
        
        # Define the main figures directory (which is at the same level as "binary" and "tables")
        figures_dir = self.SAVE_DIRECTORY / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        for trial_name in trial_names:
            # Build the path to the kilosort results for this trial
            trial_results_dir = self.SAVE_DIRECTORY / "binary" / trial_name / "kilosort4"
            if not trial_results_dir.exists():
                print(f"Results directory not found for trial: {trial_name}")
                continue
            
            # Load ops to get parameters for time axis and compute time in ms
            ops = load_ops(trial_results_dir / "ops.npy")
            t = (np.arange(ops["nt"]) / ops["fs"]) * 1000
            
            # Load custom cluster labels from TSV file
            cluster_file = trial_results_dir / "cluster_KSLabel.tsv"
            if not cluster_file.exists():
                print(f"Cluster label file not found for trial: {trial_name}")
                continue
            df_labels = pd.read_csv(cluster_file, sep="\t")
            # Select clusters marked as "good"
            good_clusters = df_labels[df_labels["KSLabel"].str.lower() == "good"]["cluster_id"].tolist()
            if len(good_clusters) == 0:
                print(f"No good clusters found for trial: {trial_name}")
                continue
            
            print(f"Plotting waveforms for trial {trial_name}: Good clusters found: {good_clusters}")
            
            # For each good cluster, generate the plot
            for cluster_id in good_clusters:
                try:
                    mean_wv = mean_waveform(cluster_id, trial_results_dir, n_spikes=100, bfile=None, best=True)
                    mean_temp = cluster_templates(cluster_id, trial_results_dir, mean=True, best=True)
                except Exception as e:
                    print(f"Error generating waveform for cluster {cluster_id} in trial {trial_name}: {e}")
                    continue
                
                fig, ax = plt.subplots(figsize=(6,4), dpi=100)
                ax.plot(t, mean_wv, c="black", linestyle="dashed", linewidth=2, label="100 spike mean waveform")
                ax.plot(t, mean_temp, linewidth=1, label="kilosort cluster template")
                ax.set_title(f"Trial {trial_name}: Cluster {cluster_id}")
                ax.set_xlabel("Time (ms)")
                ax.legend()
                
                # Save the figure to the main figures folder with the desired filename
                save_path = figures_dir / f"{trial_name}_cluster_{cluster_id}.png"
                fig.savefig(save_path, dpi=100, bbox_inches="tight")
                print(f"Saved plot for trial {trial_name}, cluster {cluster_id} at {save_path}")
                
                # Display the figure
                plt.show()

    def get_clusters(self, trial_name):
        """
        Return a sorted array of unique cluster IDs for a given trial.
        """
        if trial_name not in self.kilosort_results:
            print(f"Trial '{trial_name}' not found in kilosort_results.")
            return np.array([])
        return np.unique(self.kilosort_results[trial_name]['spike_clusters'])

    def get_spike_times(self, trial_name, cluster_id=None):
        """
        Return spike times (in seconds) for a given trial. If cluster_id is specified, return only for that cluster.
        """
        if trial_name not in self.kilosort_results:
            print(f"Trial '{trial_name}' not found in kilosort_results.")
            return np.array([])
        st = self.kilosort_results[trial_name]['spike_times']
        clu = self.kilosort_results[trial_name]['spike_clusters']
        fs = self.kilosort_results[trial_name]['ops']['fs']
        if cluster_id is None:
            return st / fs
        else:
            return st[clu == cluster_id] / fs

    def get_waveform(self, trial_name, cluster_id, n_spikes=100):
        """
        Return the mean waveform for a given cluster in a trial. Uses mean_waveform from kilosort.data_tools.
        """
        results_dir = self.SAVE_DIRECTORY / 'binary' / trial_name / 'kilosort4'
        try:
            wv = mean_waveform(cluster_id, results_dir, n_spikes=n_spikes, bfile=None, best=True)
            return wv
        except Exception as e:
            print(f"Error extracting waveform for cluster {cluster_id} in trial {trial_name}: {e}")
            return None

    def get_good_clusters(self, trial_name):
        """
        Return a list of cluster IDs labeled as 'good' for a given trial.
        First tries to read from cluster_group.tsv, falls back to cluster_KSLabel.tsv if not found.
        """
        import pandas as pd
        trial_results_dir = self.SAVE_DIRECTORY / "binary" / trial_name / "kilosort4"
        
        # First try cluster_group.tsv
        group_file = trial_results_dir / "cluster_group.tsv"
        if group_file.exists():
            try:
                # Read TSV file with explicit column names
                df_labels = pd.read_csv(group_file, sep='\t')
                print(f"Found columns in cluster_group.tsv: {df_labels.columns.tolist()}")
                
                # Check if we have the expected columns
                if 'cluster_id' in df_labels.columns and 'group' in df_labels.columns:
                    good_clusters = df_labels[df_labels['group'].str.lower() == 'good']['cluster_id'].tolist()
                    print(f"Found {len(good_clusters)} good clusters in cluster_group.tsv")
                    return good_clusters
                else:
                    print(f"Warning: Expected columns not found in {group_file}")
            except Exception as e:
                print(f"Error reading {group_file}: {str(e)}")
        
        # Fall back to cluster_KSLabel.tsv
        print(f"Trying cluster_KSLabel.tsv for trial: {trial_name}")
        cluster_file = trial_results_dir / "cluster_KSLabel.tsv"
        if not cluster_file.exists():
            print(f"Cluster label file not found for trial: {trial_name}")
            return []
        
        try:
            # Read TSV file with explicit column names
            df_labels = pd.read_csv(cluster_file, sep='\t')
            print(f"Found columns in cluster_KSLabel.tsv: {df_labels.columns.tolist()}")
            
            # Check if we have the expected columns
            if 'cluster_id' in df_labels.columns and 'KSLabel' in df_labels.columns:
                good_clusters = df_labels[df_labels['KSLabel'].str.lower() == 'good']['cluster_id'].tolist()
                print(f"Found {len(good_clusters)} good clusters in cluster_KSLabel.tsv")
                return good_clusters
            else:
                print(f"Warning: Expected columns not found in {cluster_file}")
                return []
        except Exception as e:
            print(f"Error reading {cluster_file}: {str(e)}")
            return []

    def get_good_clusters_from_group(self, trial_name):
        """
        Returns a list of cluster IDs labeled as 'good' in the cluster_group.tsv file for the given trial.
        Handles possible whitespace and capitalization issues in column names.
        """
        import pandas as pd
        from pathlib import Path

        group_file = Path(self.SAVE_DIRECTORY) / "binary" / trial_name / "kilosort4" / "cluster_group.tsv"
        if not group_file.exists():
            print(f"[WARNING] cluster_group.tsv not found for trial: {trial_name}")
            return []

        df = pd.read_csv(group_file, sep='\t')
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        # Find the correct columns
        label_col = None
        for candidate in ['group', 'kslabel', 'ks_label']:
            if candidate in df.columns:
                label_col = candidate
                break
        if label_col is None:
            print(f"[ERROR] No label column found in {group_file}. Columns are: {df.columns.tolist()}")
            return []
        cluster_col = None
        for candidate in ['cluster_id', 'clusterid', 'cluster']:
            if candidate in df.columns:
                cluster_col = candidate
                break
        if cluster_col is None:
            print(f"[ERROR] No cluster_id column found in {group_file}. Columns are: {df.columns.tolist()}")
            return []
        # Filter for 'good' clusters (case-insensitive, strip whitespace)
        good_clusters = df[df[label_col].str.strip().str.lower() == 'good'][cluster_col].tolist()
        return good_clusters

    def get_mua_clusters_from_group(self, trial_name):
        """
        Returns a list of cluster IDs labeled as 'mua' in the cluster_group.tsv file for the given trial.
        Handles possible whitespace and capitalization issues in column names.
        """
        import pandas as pd
        from pathlib import Path

        group_file = Path(self.SAVE_DIRECTORY) / "binary" / trial_name / "kilosort4" / "cluster_group.tsv"
        if not group_file.exists():
            print(f"[WARNING] cluster_group.tsv not found for trial: {trial_name}")
            return []

        df = pd.read_csv(group_file, sep='\t')
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        # Find the correct columns
        label_col = None
        for candidate in ['group', 'kslabel', 'ks_label']:
            if candidate in df.columns:
                label_col = candidate
                break
        if label_col is None:
            print(f"[ERROR] No label column found in {group_file}. Columns are: {df.columns.tolist()}")
            return []
        cluster_col = None
        for candidate in ['cluster_id', 'clusterid', 'cluster']:
            if candidate in df.columns:
                cluster_col = candidate
                break
        if cluster_col is None:
            print(f"[ERROR] No cluster_id column found in {group_file}. Columns are: {df.columns.tolist()}")
            return []
        # Filter for 'mua' clusters (case-insensitive, strip whitespace)
        mua_clusters = df[df[label_col].str.strip().str.lower() == 'mua'][cluster_col].tolist()
        return mua_clusters


class RatGroup:
    """
    Aggregates multiple Rat objects to provide a unified interface for analysis and plotting.
    
    This version allows you to specify a single parent folder, and it will create a subfolder for each rat,
    named after the rat's ID, to store that rat's output.
    """
    def __init__(self, rat_list):
        self.rats = {rat.RAT_ID: rat for rat in rat_list}
    
    def run_preprocessing(self, first_window=(0, 1050000), last_window=None, remove_drg_stim=True):
        for rat in self.rats.values():
            rat.get_sc_data()
            rat.get_analog_data()
            if remove_drg_stim == True:
                rat.remove_drg_stim_window_sc(first_window, last_window)
                rat.remove_drg_stim_window_analog(first_window, last_window)
    

    def create_spikeinterface_wrappers(self, parent_folder):
        """
        Creates a SpikeInterface_wrapper for each rat.
        
        Parameters:
          parent_folder: the path to the parent folder where subfolders (named after each rat ID)
                         will be created for saving results.
        
        Returns a dictionary mapping rat ID to its SpikeInterface_wrapper.
        """
        self.si_wrappers = {}
        parent_folder = Path(parent_folder)
        for rat_id, rat in self.rats.items():
            rat_folder = parent_folder / rat_id
            rat_folder.mkdir(parents=True, exist_ok=True)
            self.si_wrappers[rat_id] = SpikeInterface_wrapper(rat, str(rat_folder))
        return self.si_wrappers
    
    def create_kilosort_wrappers(self, parent_folder, probe_directory):
        """
        Creates a Kilosort_wrapper for each rat.
        
        Parameters:
          parent_folder: the path to the parent folder where subfolders (named after each rat ID)
                         will be created for saving results.
          probe_directory: the path to the probe file.
        
        Returns a dictionary mapping rat ID to its Kilosort_wrapper.
        """
        self.ks_wrappers = {}
        parent_folder = Path(parent_folder)
        for rat_id, rat in self.rats.items():
            rat_folder = parent_folder / rat_id
            rat_folder.mkdir(parents=True, exist_ok=True)
            self.ks_wrappers[rat_id] = Kilosort_wrapper(str(rat_folder), probe_directory)
        return self.ks_wrappers

    # === Modular Data Access Methods ===
    def get_spike_times(self, trial_name, cluster_id=None):
        """
        Return spike times (in seconds) for a given trial. If cluster_id is specified, return only for that cluster.
        """
        if trial_name not in self.ks_wrappers:
            print(f"Trial '{trial_name}' not found in kilosort_wrappers.")
            return np.array([])
        ks = self.ks_wrappers[trial_name]
        st = ks.kilosort_results[trial_name]['spike_times']
        clu = ks.kilosort_results[trial_name]['spike_clusters']
        fs = ks.kilosort_results[trial_name]['ops']['fs']
        if cluster_id is None:
            return st / fs
        else:
            return st[clu == cluster_id] / fs

    def get_clusters(self, trial_name):
        """
        Return a sorted array of unique cluster IDs for a given trial.
        """
        if trial_name not in self.ks_wrappers:
            print(f"Trial '{trial_name}' not found in kilosort_wrappers.")
            return np.array([])
        return np.unique(self.ks_wrappers[trial_name].kilosort_results[trial_name]['spike_clusters'])

    def get_waveform(self, trial_name, cluster_id, n_spikes=100):
        """
        Return the mean waveform for a given cluster in a trial. Uses mean_waveform from kilosort.data_tools.
        """
        results_dir = self.SAVE_DIRECTORY / 'binary' / trial_name / 'kilosort4'
        try:
            wv = mean_waveform(cluster_id, results_dir, n_spikes=n_spikes, bfile=None, best=True)
            return wv
        except Exception as e:
            print(f"Error extracting waveform for cluster {cluster_id} in trial {trial_name}: {e}")
            return None
 