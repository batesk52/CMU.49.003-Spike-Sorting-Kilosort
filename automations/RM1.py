"""
Below is a refactored version of your RM1.py file. The changes include:

Streamlining Data Import:
Instead of five separate dictionaries for Intan recordings, all streams are stored in a single dictionary (keyed by stream ID), reducing code duplication in the import routine.

Helper Function for Window Removal:
A private method (_remove_window) was added so that the code for slicing and concatenating recordings is shared between spinal cord and analog data removal.

Improved Exception Handling and Use of Pathlib:
File/directory operations are now handled via pathlib where possible, and try/except blocks now report the encountered exceptions.

Meta Class for Multiple Rats:
A new class, RatGroup, aggregates multiple Rat objects, allowing you to run preprocessing across all rats and create wrappers for SpikeInterface and Kilosort. This class also stores each rat’s ID (which will be useful for subsequent mixed-model analyses).

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
import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams
from spikeinterface import concatenate_recordings
from kilosort.run_kilosort import load_sorting
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

    def get_von_frey_analysis(self, si_wrapper, ks_wrapper, excel_parent_folder=None, **kwargs):
        """
        Computes or loads the VonFrey analysis for this rat.
        
        If excel_parent_folder is provided, this method checks for a subfolder:
        {excel_parent_folder}/{RatID}/tables
        and if Excel files matching the naming pattern are found, it loads those results.
        Otherwise, it runs the analysis.
        
        Parameters:
        si_wrapper: The SpikeInterface_wrapper for this rat.
        ks_wrapper: The Kilosort_wrapper for this rat.
        excel_parent_folder: str or Path; the parent folder under which each rat’s results
                            (in a subfolder named by RatID with a "tables" folder) might exist.
        kwargs: additional parameters for the analysis (e.g., subwindow_width, corr_threshold).
        
        Returns:
        results: dict keyed by trial names.
        """
        from pathlib import Path
        if excel_parent_folder is not None:
            candidate = Path(excel_parent_folder) / self.RAT_ID / "tables"
            if candidate.exists() and any(candidate.glob("*_average_vf_voltage_windowed.xlsx")):
                return self.load_von_frey_analysis_results(candidate)
        # Remove the extra key so that analyze_subwindows() does not receive it.
        kwargs.pop('excel_parent_folder', None)
        from automations import analysis_functions  # local import to avoid circular dependencies
        vfa = analysis_functions.VonFreyAnalysis(self, si_wrapper, ks_wrapper)
        return vfa.analyze_subwindows(**kwargs)


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

    def save_spinalcord_data_to_binary(self, TRIAL_NAMES=None):
        trial_list = TRIAL_NAMES if TRIAL_NAMES is not None else self.data.sc_data.keys()
        for recording in trial_list:
            try:
                rec = self.data.sc_data[recording]
                recording_filtered = spre.bandpass_filter(rec, freq_min=300, freq_max=14000)
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
                ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = run_kilosort(
                    settings=settings, probe_name=str(self.PROBE_DIRECTORY)
                )
                results_dir = self.SAVE_DIRECTORY / 'binary' / folder.name / 'kilosort4'
                ops = np.load(results_dir / 'ops.npy', allow_pickle=True).item()
                camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values
                contam_pct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values
                chan_map = np.load(results_dir / 'channel_map.npy')
                templates = np.load(results_dir / 'templates.npy')
                chan_best = chan_map[(templates**2).sum(axis=1).argmax(axis=-1)]
                amplitudes = np.load(results_dir / 'amplitudes.npy')
                st = np.load(results_dir / 'spike_times.npy')
                clu = np.load(results_dir / 'spike_clusters.npy')
                firing_rates = np.unique(clu, return_counts=True)[1] * 30000 / st.max()
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
                # (Optional) Plotting code can be added here if desired.
            except Exception as e:
                print(f"Error processing folder {folder.name}: {e}")
                continue
            self.apply_custom_labels_to_trial(folder.name, custom_criteria=custom_criteria)
    
    def extract_kilosort_outputs(self):
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
                ops = np.load(results_dir / 'ops.npy', allow_pickle=True).item()
                camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values
                contam_pct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values
                chan_map = np.load(results_dir / 'channel_map.npy')
                templates = np.load(results_dir / 'templates.npy')
                chan_best = chan_map[(templates**2).sum(axis=1).argmax(axis=-1)]
                amplitudes = np.load(results_dir / 'amplitudes.npy')
                st = np.load(results_dir / 'spike_times.npy')
                clu = np.load(results_dir / 'spike_clusters.npy')
                firing_rates = np.unique(clu, return_counts=True)[1] * 30000 / st.max()
                dshift = ops['dshift']
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



class RatGroup:
    """
    Aggregates multiple Rat objects to provide a unified interface for analysis and plotting.
    
    This version allows you to specify a single parent folder, and it will create a subfolder for each rat,
    named after the rat's ID, to store that rat's output.
    """
    def __init__(self, rat_list):
        self.rats = {rat.RAT_ID: rat for rat in rat_list}
    
    def run_preprocessing(self, first_window=(0, 1050000), last_window=None):
        for rat in self.rats.values():
            rat.get_sc_data()
            rat.get_analog_data()
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
 