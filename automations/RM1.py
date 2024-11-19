import os
import scipy.io as sio
from spikeinterface.extractors import read_intan
import pandas as pd
import numpy as np
from kilosort import io
from kilosort import run_kilosort, DEFAULT_SETTINGS

class Rat:
    
    def __init__(self, DATA_DIRECTORY, PROBE_DIRECTORY, RAT_ID):
        
        self.RAT_ID = RAT_ID
        self.PROBE_DIRECTORY = PROBE_DIRECTORY
        self.DATA_DIRECTORY = DATA_DIRECTORY
        self.intan_recordings = {}
        self.emg_data,self.nerve_cuff_data,self.sc_data = {},{},{}
        self.st_experiment_notes = None
        self.qst_experiment_notes = None
        self.drgs_experiment_notes = None
        self.sc_experiment_notes = None
        self.st_trial_notes = None
        self.qst_trial_notes = None
        self.drgs_trial_notes = None
        self.sc_trial_notes = None
        self.mat_files_dict = {}
        self.trial_names = [] # trial names refer to the folder names with the experiment type, trial number, and timestamp ex ST_8_240918_121741

        # importing metadata from the experiment
        self.import_metadata()

        # importing matlab files for the whole animal
        self.import_matlab_files()

        # importing intan data
        self.import_intan_data()
        self.get_nerve_cuff_data()  
        self.get_sc_data()
        self.get_emg_data()

    def import_metadata(self):
        metadata_path = os.path.join(self.DATA_DIRECTORY, self.RAT_ID, f'{self.RAT_ID}_TermExpSchedule.xlsx')
        metadata = pd.read_excel(metadata_path)
        
        # importing the stimulation threshold (ST) experiment and trial metadata
        sheet = pd.read_excel(metadata_path,sheet_name='ST')
        self.st_experiment_notes = sheet.iloc[4,1]
        trial_notes = sheet.iloc[:,:]
        trial_notes.columns = trial_notes.iloc[5,:]
        trial_notes.index = trial_notes["Trial Number"]
        self.st_trial_notes = trial_notes.iloc[6:,:]

        # importing the dorsal root ganglion stimulation (DRGS) experiment and trial metadata
        sheet = pd.read_excel(metadata_path,sheet_name='DRGS')
        self.drgs_experiment_notes = sheet.iloc[4,1]
        trial_notes = sheet.iloc[:,:]
        trial_notes.columns = trial_notes.iloc[5,:]
        trial_notes.index = trial_notes["Trial Number"]
        self.drgs_trial_notes = trial_notes.iloc[6:,:]

        # importing the spinal cord (SC) experiment and trial metadata
        sheet = pd.read_excel(metadata_path,sheet_name='SC')
        self.sc_experiment_notes = sheet.iloc[4,1]
        trial_notes = sheet.iloc[:,:]
        trial_notes.columns = trial_notes.iloc[5,:]
        trial_notes.index = trial_notes["Trial Number"]
        self.sc_trial_notes = trial_notes.iloc[6:,:]

        # importing the pain stimuli (QST) experiment and trial metadata
        sheet = pd.read_excel(metadata_path,sheet_name='QST')
        self.qst_experiment_notes = sheet.iloc[4,1]
        trial_notes = sheet.iloc[:,:]
        trial_notes.columns = trial_notes.iloc[5,:]
        trial_notes.index = trial_notes["Trial Number"]
        self.qst_trial_notes = trial_notes.iloc[6:,:]


    def import_matlab_files(self):
        # Dictionary to store data from all .mat files
        # Iterate over all files in the directory
        for filename in os.listdir(os.path.join(self.DATA_DIRECTORY,self.RAT_ID,self.RAT_ID)):
            if filename.endswith(".mat"):
                file_path = os.path.join(self.DATA_DIRECTORY,self.RAT_ID,self.RAT_ID, filename)
                
                # Load the .mat file
                mat_data = sio.loadmat(file_path)
                
                # Use the filename (without extension) as the key in the dictionary
                self.mat_files_dict[filename[:-4]] = mat_data


    # Import all intan data from rhd file and store in a dictionary
    def import_intan_data(self):
        data_path = os.path.join(self.DATA_DIRECTORY, self.RAT_ID,self.RAT_ID)
        for folder_name in os.listdir(data_path):
            self.trial_names.append(folder_name)
            folder_path = os.path.join(data_path, folder_name)
            if os.path.isdir(folder_path):
                rhd_file = f'{folder_name}.rhd'
                rhd_file_path = os.path.join(folder_path, rhd_file)
                if os.path.exists(rhd_file_path):
                    # Read the recording and store it in the dictionary
                    recording = read_intan(rhd_file_path, stream_id='0')
                    self.intan_recordings[folder_name] = recording
                    print(f'Reading {folder_name}...')
        return self.intan_recordings
        
    """
    Separate channels according to their function based on channel IDs.
    Channels 0-31: Intraspinal recordings (Neural Nexus probe)
    Channel 32: Nerve cuff electrode (Peripheral nerve)
    Channel 33: Muscle response (EMG)
    """

    def get_sc_data(self):
        for recording in self.intan_recordings:
            self.sc_data[recording] = self.intan_recordings[recording].remove_channels(["B-000","B-001"])   

    def get_nerve_cuff_data(self):
        for recording in self.intan_recordings:
            self.nerve_cuff_data[recording] = self.intan_recordings[recording].select_channels(["B-000"])

    def get_emg_data(self):
        for recording in self.intan_recordings:
            self.emg_data[recording] = self.intan_recordings[recording].select_channels(["B-001"])

class kilosort_wrapper:
    def __init__(self, rat_class_instance, SAVE_DIRECTORY): # for this class, you will pass the class instance of a rat object as an argument
        self.RAT_ID = rat_class_instance.RAT_ID
        self.data = rat_class_instance
        self.PROBE_DIRECTORY = rat_class_instance.PROBE_DIRECTORY
        self.DATA_DIRECTORY = rat_class_instance.DATA_DIRECTORY
        self.SAVE_DIRECTORY = SAVE_DIRECTORY
        print(self.RAT_ID)


    def save_spinalcord_data_to_binary(self, TRIAL_NAMES = None):
        # NOTE: Data will be saved as np.int16 by default since that is the standard
        # for ephys data. If you need a different data type for whatever reason
        # such as `np.uint16`, be sure to update this.
        
        if TRIAL_NAMES == None: # if nothing is passed, it will run through all of the trials in the rat object
            for recording in self.data.intan_recordings:
                dtype = np.int16
                filename, N, c, s, fs, probe_path = io.spikeinterface_to_binary(
                    self.data.intan_recordings[recording], self.SAVE_DIRECTORY, data_name=f'{self.RAT_ID}_{recording}_data.bin', dtype=dtype,
                    chunksize=60000, export_probe=False, probe_name= self.PROBE_DIRECTORY # export the probe file is false because there is no probe file to begin with
                    )
        else:
            try:
                for recording in TRIAL_NAMES:
                    dtype = np.int16
                    filename, N, c, s, fs, probe_path = io.spikeinterface_to_binary(
                        self.data.intan_recordings[recording], self.SAVE_DIRECTORY, data_name=f'{self.RAT_ID}_{recording}_data.bin', dtype=dtype,
                        chunksize=60000, export_probe=False, probe_name= self.PROBE_DIRECTORY # export the probe file is false because there is no probe file to begin with
                        )
            except:
                print(f'No data found for {recording}')

        print(f'Data saved to {filename}')

    def run_spinalcord_kilosort(self, new_settings = None, **kwargs):

        # NOTE: 'n_chan_bin' is a required setting, and should reflect the total number
        #       of channels in the binary file. For information on other available
        #       settings, see `kilosort.run_kilosort.default_settings`.

        if new_settings is None:
            settings = DEFAULT_SETTINGS
            settings = {'data_dir': self.SAVE_DIRECTORY, 'n_chan_bin': 32}
        
        else:
            settings = DEFAULT_SETTINGS
            settings = new_settings

        ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
            run_kilosort(
                settings=settings, probe_name=self.PROBE_DIRECTORY,
                # save_preprocessed_copy=True
                )
        
        