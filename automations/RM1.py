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

class Rat:
    
    def __init__(self, DATA_DIRECTORY, PROBE_DIRECTORY, RAT_ID):
        
        self.RAT_ID = RAT_ID
        self.PROBE_DIRECTORY = PROBE_DIRECTORY
        self.DATA_DIRECTORY = DATA_DIRECTORY
        self.intan_recordings_stream0 = {}
        self.intan_recordings_stream1 = {}
        self.intan_recordings_stream2 = {}
        self.intan_recordings_stream3 = {}
        self.intan_recordings_stream4 = {}
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
                    print(f'Reading {folder_name}...')
                    # using a for loop, try to import each stream separately
                    try:
                        recording = read_intan(rhd_file_path, stream_id='0')
                        self.intan_recordings_stream0[folder_name] = recording
                    except:
                        print(f'Error reading stream 0 for {folder_name}. continuing...')
                        pass
                    try:
                        recording = read_intan(rhd_file_path, stream_id='1')
                        self.intan_recordings_stream1[folder_name] = recording
                    except:
                        print(f'Error reading stream 1 for {folder_name}. continuing...')
                        pass
                    try:
                        recording = read_intan(rhd_file_path, stream_id='2')
                        self.intan_recordings_stream2[folder_name] = recording
                    except:
                        print(f'Error reading stream 2 for {folder_name}. continuing...')
                        pass
                    try:
                        recording = read_intan(rhd_file_path, stream_id='3')
                        self.intan_recordings_stream3[folder_name] = recording
                    except:
                        print(f'Error reading stream 3 for {folder_name}. continuing...')
                        pass                   
                    try:
                        recording = read_intan(rhd_file_path, stream_id='4')
                        self.intan_recordings_stream4[folder_name] = recording
                    except:
                        print(f'Error reading stream 4 for {folder_name}. continuing...')
                        pass
    """
    Separate channels according to their function based on channel IDs.
    Channels 0-31: Intraspinal recordings (Neural Nexus probe)
    Channel 32: Nerve cuff electrode (Peripheral nerve)
    Channel 33: Muscle response (EMG)
    """

    def get_sc_data(self):
        for recording in self.intan_recordings_stream0:
            try:
                self.sc_data[recording] = self.intan_recordings_stream0[recording].remove_channels(["B-000","B-001"])   
            except:
                self.sc_data[recording] = None  
                print(f'Error recording SC data for {recording}')
    def get_nerve_cuff_data(self):
        for recording in self.intan_recordings_stream0:
            try:
                self.nerve_cuff_data[recording] = self.intan_recordings_stream0[recording].select_channels(["B-000"])
            except:
                self.nerve_cuff_data[recording] = None
                print(f'Error recording nerve cuff data for {recording}')
                pass

    def get_emg_data(self):
        for recording in self.intan_recordings_stream0:
            try:
                self.emg_data[recording] = self.intan_recordings_stream0[recording].select_channels(["B-001"])
            except:
                self.emg_data[recording] = None
                print(f'Error recording emg data for {recording}')
                pass

class kilosort_wrapper:
    def __init__(self, rat_class_instance, SAVE_DIRECTORY): # for this class, you will pass the class instance of a rat object as an argument
        self.RAT_ID = rat_class_instance.RAT_ID
        self.data = rat_class_instance
        self.PROBE_DIRECTORY = rat_class_instance.PROBE_DIRECTORY
        self.DATA_DIRECTORY = rat_class_instance.DATA_DIRECTORY
        self.SAVE_DIRECTORY = SAVE_DIRECTORY
        print(f"running kilosort wrapper for rat {self.RAT_ID}")

    def save_spinalcord_data_to_binary(self, TRIAL_NAMES = None):
        # NOTE: Data will be saved as np.int16 by default since that is the standard
        # for ephys data. If you need a different data type for whatever reason
        # such as `np.uint16`, be sure to update this.
        
        # these files will be saved to the subfolder: SAVE_DIRECTORY/binary/TRIAL_NAMES

        if TRIAL_NAMES == None: # if nothing is passed, it will run through all of the trials in the rat object
            try:
                for recording in self.data.sc_data:
                    dtype = np.int16
                    filename, N, c, s, fs, probe_path = io.spikeinterface_to_binary(
                        self.data.sc_data[recording], os.path.join(self.SAVE_DIRECTORY, 'binary',f"{recording}"), data_name=f'{self.RAT_ID}_{recording}_data.bin', dtype=dtype,
                        chunksize=60000, export_probe=False, probe_name= self.PROBE_DIRECTORY # export the probe file is false because there is no probe file to begin with
                        )
                    print(f'Data saved to {filename}')
            except:
                print(f'ERROR: issue importing data for {recording}')
        else:
            try:
                for recording in TRIAL_NAMES:
                    dtype = np.int16
                    filename, N, c, s, fs, probe_path = io.spikeinterface_to_binary(
                        self.data.sc_data[recording], os.path.join(self.SAVE_DIRECTORY, 'binary',f"{recording}"), data_name=f'{self.RAT_ID}_{recording}_data.bin', dtype=dtype,
                        chunksize=60000, export_probe=False, probe_name= self.PROBE_DIRECTORY # export the probe file is false because there is no probe file to begin with
                        )
            except:
                print(f'ERROR: issue importing data for {recording}')

    def run_spinalcord_kilosort(self, new_settings = None, **kwargs): # runs kilosort in a loop through all of the binary files saved to the SAVE_DIRECTORY/binary folder

        # Get a list of all folders that contain `.bin` files
        folders_with_bin = []
        binary_folder = self.SAVE_DIRECTORY
        # Walk through the binary folder
        for root, dirs, files in os.walk(binary_folder):
            if any(file.endswith('.bin') for file in files):
                folders_with_bin.append(root)

        for folder in folders_with_bin:
            print(f'|\n|\n|\n|\n|\n|\nRunning kilosort on {os.path.basename(folder)}\n|\n|\n|\n|\n|\n|')

            try:
                    
                # NOTE: 'n_chan_bin' is a required setting, and should reflect the total number
                #       of channels in the binary file. For information on other available
                #       settings, see `kilosort.run_kilosort.default_settings`.

                if new_settings is None:
                    settings = DEFAULT_SETTINGS
                    settings = {'data_dir': os.path.join(self.SAVE_DIRECTORY, 'binary', folder), 'n_chan_bin': 32}
                
                else:
                    settings = DEFAULT_SETTINGS
                    settings = new_settings

                ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
                    run_kilosort(
                        settings=settings, probe_name=self.PROBE_DIRECTORY,
                        # save_preprocessed_copy=True
                        )
                
                try: # save the kilosort outputs
                    # outputs saved to results_dir
                    results_dir = Path(os.path.join(self.SAVE_DIRECTORY, 'binary',os.path.basename(folder), 'kilosort4')) # Path(settings['data_dir']).joinpath('kilosort4')
                    ops = np.load(results_dir / 'ops.npy', allow_pickle=True).item()
                    camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values
                    contam_pct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values
                    chan_map =  np.load(results_dir / 'channel_map.npy')
                    templates =  np.load(results_dir / 'templates.npy')
                    chan_best = (templates**2).sum(axis=1).argmax(axis=-1)
                    chan_best = chan_map[chan_best]
                    amplitudes = np.load(results_dir / 'amplitudes.npy')
                    st = np.load(results_dir / 'spike_times.npy')
                    clu = np.load(results_dir / 'spike_clusters.npy')
                    firing_rates = np.unique(clu, return_counts=True)[1] * 30000 / st.max()
                    dshift = ops['dshift']
                except:
                    print(f'Error saving kilosort outputs for {folder}')
                    pass

                try: # plotting the kilosort outputs
                    rcParams['axes.spines.top'] = False
                    rcParams['axes.spines.right'] = False
                    gray = .5 * np.ones(3)

                    fig = plt.figure(figsize=(10,10), dpi=100)
                    fig.suptitle(os.path.basename(folder))
                    grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.5)

                    ax = fig.add_subplot(grid[0,0])
                    ax.plot(np.arange(0, ops['Nbatches'])*2, dshift);
                    ax.set_xlabel('time (sec.)')
                    ax.set_ylabel('drift (um)')

                    ax = fig.add_subplot(grid[0,1:])
                    t0 = 0
                    t1 = np.nonzero(st > ops['fs']*5)[0][0]
                    ax.scatter(st[t0:t1]/30000., chan_best[clu[t0:t1]], s=0.5, color='k', alpha=0.25)
                    ax.set_xlim([0, 5])
                    ax.set_ylim([chan_map.max(), 0])
                    ax.set_xlabel('time (sec.)')
                    ax.set_ylabel('channel')
                    ax.set_title('spikes from units')

                    ax = fig.add_subplot(grid[1,0])
                    nb=ax.hist(firing_rates, 20, color=gray)
                    ax.set_xlabel('firing rate (Hz)')
                    ax.set_ylabel('# of units')

                    ax = fig.add_subplot(grid[1,1])
                    nb=ax.hist(camps, 20, color=gray)
                    ax.set_xlabel('amplitude')
                    ax.set_ylabel('# of units')

                    ax = fig.add_subplot(grid[1,2])
                    nb=ax.hist(np.minimum(100, contam_pct), np.arange(0,105,5), color=gray)
                    ax.plot([10, 10], [0, nb[0].max()], 'k--')
                    ax.set_xlabel('% contamination')
                    ax.set_ylabel('# of units')
                    ax.set_title('< 10% = good units')

                    for k in range(2):
                        ax = fig.add_subplot(grid[2,k])
                        is_ref = contam_pct<10.
                        ax.scatter(firing_rates[~is_ref], camps[~is_ref], s=3, color='r', label='mua', alpha=0.25)
                        ax.scatter(firing_rates[is_ref], camps[is_ref], s=3, color='b', label='good', alpha=0.25)
                        ax.set_ylabel('amplitude (a.u.)')
                        ax.set_xlabel('firing rate (Hz)')
                        ax.legend()
                        if k==1:
                            ax.set_xscale('log')
                            ax.set_yscale('log')
                            ax.set_title('loglog')
                    plt.savefig(os.path.join(self.SAVE_DIRECTORY, 'figures', f'review_plots_{self.RAT_ID}_{os.path.basename(folder)}.png'), dpi=150)
                    plt.show()

                except ValueError as e:
                    print(f'Error plotting kilosort outputs for {folder}')
                    raise e

                try:
                    probe = ops['probe']
                    # x and y position of probe sites
                    xc, yc = probe['xc'], probe['yc']
                    nc = 16 # number of channels to show
                    good_units = np.nonzero(contam_pct <= 0.1)[0]
                    mua_units = np.nonzero(contam_pct > 0.1)[0]


                    gstr = ['good', 'mua']
                    for j in range(2):
                        try:
                            # print(f'~~~~~~~~~~~~~~ {gstr[j]} units ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                            # print('title = number of spikes from each unit')
                            # print(os.path.basename(folder))
                            units = good_units if j==0 else mua_units 
                            fig = plt.figure(figsize=(12,3), dpi=150)
                            grid = gridspec.GridSpec(2,20, figure=fig, hspace=0.25, wspace=0.5)
                            fig.suptitle(f"trial: {os.path.basename(folder)}: {gstr[j]} units | title = number of spikes from each unit ")

                            for k in range(40):
                                wi = units[np.random.randint(len(units))]
                                wv = templates[wi].copy()  
                                cb = chan_best[wi]
                                nsp = (clu==wi).sum()
                                
                                ax = fig.add_subplot(grid[k//20, k%20])
                                n_chan = wv.shape[-1]
                                ic0 = max(0, cb-nc//2)
                                ic1 = min(n_chan, cb+nc//2)
                                wv = wv[:, ic0:ic1]
                                x0, y0 = xc[ic0:ic1], yc[ic0:ic1]

                                amp = 4
                                for ii, (xi,yi) in enumerate(zip(x0,y0)):
                                    t = np.arange(-wv.shape[0]//2,wv.shape[0]//2,1,'float32')
                                    t /= wv.shape[0] / 20
                                    ax.plot(xi + t, yi + wv[:,ii]*amp, lw=0.5, color='k')

                                ax.set_title(f'{nsp}', fontsize='small')
                                ax.axis('off')
                            plt.savefig(os.path.join(self.SAVE_DIRECTORY, 'figures', f'spikes_{gstr[j]}_{self.RAT_ID}_{os.path.basename(folder)}.png'), dpi=150)
                            plt.show()
                        except:
                            print(f'ERROR: could not plot units for {os.path.basename(folder)}, {gstr[j]}. skipping plot...')
                except ValueError as e:
                    print(f'ERROR: could not plot {os.path.basename(folder)}')
                    raise e

            except ValueError as e:
                error_message = str(e)
                print(f"\nerror processing data in folder: {os.path.basename(folder)}")

                # Custom handling for specific errors
                if "Unrecognized extension for probe" in error_message:

                    print("- check your probe file directory - is it a prb json?")
                    # Optionally re-raise the exception if necessary
                    pass
                else:
                    print("\nAn unexpected error occurred:")
                    print(error_message)
                    # raise e
                    pass

                