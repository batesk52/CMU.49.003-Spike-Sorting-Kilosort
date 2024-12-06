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

class Rat: # for this class, you pass the folder containing the rat metadata in excel and matlab files, as well as the intan data in rhd format
    """
    The `Rat` class is designed to process and store data from experiments involving rats. It takes in three parameters: `DATA_DIRECTORY`, `PROBE_DIRECTORY`, and `RAT_ID`.
    Here is a list explaining what each class method does:
    * `__init__`: Initializes the `Rat` object by setting its attributes and importing metadata, MATLAB files, and Intan data.
    * `import_metadata`: Imports metadata from an Excel file and stores it in the object's attributes.
    * `import_matlab_files`: Imports MATLAB files from a directory and stores them in a dictionary.
    * `import_intan_data`: Imports Intan data from RHD files and stores it in a dictionary.
    * `get_sc_data`: Extracts spinal cord data from the Intan recordings and stores it in a dictionary.
    * `get_nerve_cuff_data`: Extracts nerve cuff data from the Intan recordings and stores it in a dictionary.
    * `get_emg_data`: Extracts EMG data from the Intan recordings and stores it in a dictionary.

    Note that the `Rat` class assumes a specific directory structure and file naming convention, which is not explicitly documented in the code.
    
    """
    def __init__(self, DATA_DIRECTORY, PROBE_DIRECTORY, RAT_ID, **kwargs):
        
        self.RAT_ID = RAT_ID
        self.PROBE_DIRECTORY = PROBE_DIRECTORY
        self.DATA_DIRECTORY = DATA_DIRECTORY
        self.intan_recordings_stream0 = {}
        self.intan_recordings_stream1 = {}
        self.intan_recordings_stream2 = {}
        self.intan_recordings_stream3 = {}
        self.intan_recordings_stream4 = {}
        self.emg_data,self.nerve_cuff_data,self.sc_data = {},{},{}
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

    def get_analog_data(self):
        for recording in self.intan_recordings_stream0:
            try:
                self.analog_data[recording] = self.intan_recordings_stream3[recording]
            except:
                self.analog_data[recording] = None
                print(f'Error recording von frey data for {recording}')
                pass
    
    
    def slice_and_concatenate_recording(self, base_recording, first_window=(0, 1050000), last_window=None):
        """
        Slices the base recording to keep only the specified first and last window frames,
        concatenates them, and returns the combined recording.

        Parameters:
        - base_recording: The recording to slice (e.g., sc_data, emg_data, etc.).
        - first_window: A tuple (start_frame, end_frame) for the first window.
        - last_window: A tuple (start_frame, end_frame) for the last window.
                      If None, the last window will be from (total_frames - 1050000) to total_frames.

        Returns:
        - recording_combined: The concatenated recording of the two slices.
        """

        # Slice the first window
        recording_first = base_recording.frame_slice(start_frame=first_window[0], end_frame=first_window[1])

        # Determine the last window frames
        total_frames = base_recording.get_num_frames()
        if last_window is None:
            # Default to the last 1,050,000 frames (approximately 35 seconds)
            last_window = (total_frames - 1050000, total_frames)
        else:
            # Ensure the end_frame does not exceed total_frames
            last_window = (last_window[0], min(last_window[1], total_frames))

        # Slice the last window
        recording_last = base_recording.frame_slice(start_frame=last_window[0], end_frame=last_window[1])

        # Concatenate the two recordings
        recording_combined = concatenate_recordings([recording_first, recording_last])

        return recording_combined

    def remove_drg_stim_window_sc(self, first_window=(0, 1050000), last_window=None):
        """
        Processes all spinal cord data recordings by slicing and concatenating the specified windows.

        Parameters:
        - first_window: Tuple specifying the first window frames.
        - last_window: Tuple specifying the last window frames.

         the default is to preserve the first & last 1,050,000 frames (approximately 35 seconds) from the recording.
        """
        processed_sc_data = {}
        for recording_name, recording in self.sc_data.items():
            if recording is not None:
                try:
                    combined_recording = self.slice_and_concatenate_recording(
                        base_recording=recording,
                        first_window=first_window,
                        last_window=last_window
                    )
                    processed_sc_data[recording_name] = combined_recording
                except Exception as e:
                    print(f"Error processing {recording_name}: {e}")
                    processed_sc_data[recording_name] = None
            else:
                print(f"No recording found for {recording_name}")
                processed_sc_data[recording_name] = None
        # Optionally, update self.sc_data with the processed data
        self.sc_data = processed_sc_data

    def remove_drg_stim_window_analog(self, first_window=(0, 1050000), last_window=None):
        """
        Processes all analog recordings by slicing and concatenating the specified windows.

        Parameters:
        - first_window: Tuple specifying the first window frames.
        - last_window: Tuple specifying the last window frames.

         the default is to preserve the first & last 1,050,000 frames (approximately 35 seconds) from the recording.
        """
        processed_analog_data = {}
        for recording_name, recording in self.analog_data.items():
            if recording is not None:
                try:
                    combined_recording = self.slice_and_concatenate_recording(
                        base_recording=recording,
                        first_window=first_window,
                        last_window=last_window
                    )
                    processed_analog_data[recording_name] = combined_recording
                except Exception as e:
                    print(f"Error processing {recording_name}: {e}")
                    processed_analog_data[recording_name] = None
            else:
                print(f"No recording found for {recording_name}")
                processed_analog_data[recording_name] = None
        # Optionally, update self.sc_data with the processed data
        self.analog_data = processed_analog_data


class SpikeInterface_wrapper: # for this class, you will pass an instance of a rat class as an argument
    def __init__(self, rat_class_instance, SAVE_DIRECTORY): # for this class, you will pass the class instance of a rat object as an argument
        self.RAT_ID = rat_class_instance.RAT_ID
        self.data = rat_class_instance
        self.PROBE_DIRECTORY = rat_class_instance.PROBE_DIRECTORY
        self.DATA_DIRECTORY = rat_class_instance.DATA_DIRECTORY
        self.SAVE_DIRECTORY = SAVE_DIRECTORY
        self.kilosort_results = {}  # Initialize the dictionary to store results
        print(f"Preparing SpikeInterface wrapper for rat {self.RAT_ID}")


    def save_spinalcord_data_to_binary(self, TRIAL_NAMES=None):
        # If no trial names are supplied, automatically process all trials
        trial_list = TRIAL_NAMES if TRIAL_NAMES is not None else self.data.sc_data.keys()

        for recording in trial_list:
            try:
                # Apply bandpass filtering and common average referencing using the preprocessing functions
                recording_filtered = spre.bandpass_filter(self.data.sc_data[recording], freq_min=300, freq_max=14000)

                # Apply a common reference to achieve a CAR-like effect
                # Setting `reference='global'` and `operator='average'` is akin to a CAR (this replace the old version , common_average_referece)
                recording_preprocessed = spre.common_reference(recording_filtered, reference='global', operator='average')

                dtype = np.int16
                filename, N, c, s, fs, probe_path = io.spikeinterface_to_binary(
                    recording_preprocessed, 
                    os.path.join(self.SAVE_DIRECTORY, 'binary', f"{recording}"), 
                    data_name=f'{self.RAT_ID}_{recording}_data.bin', 
                    dtype=dtype,
                    chunksize=60000, 
                    export_probe=False, 
                    probe_name=self.PROBE_DIRECTORY
                )
                print(f'Data saved to {filename}')

            except ValueError as e:
                print(f'ERROR: issue importing data for {recording}',"\n",e)
                pass


class Kilosort_wrapper: # for this class, you will pass the directory containing the binary files and kilosort results (if available)

    def __init__(self, SAVE_DIRECTORY,PROBE_DIRECTORY): # for this class, you will pass the class instance of a rat object as an argument
        self.PROBE_DIRECTORY = PROBE_DIRECTORY
        self.SAVE_DIRECTORY = SAVE_DIRECTORY
        self.kilosort_results = {}  # Initialize the dictionary to store results
        print(f"Preparing Kilosort wrapper...")

    def run_or_load_kilosort(self, trial_name, run_kilosort=False):
        if run_kilosort:
            # Code to run Kilosort
            self.run_kilosort_trial_summary()
        else:
            # Load existing Kilosort outputs
            self.extract_kilosort_outputs(trial_name)

    def apply_custom_labels_to_trial(self, trial_name, custom_criteria=None):
        try:
            results_dir = Path(self.SAVE_DIRECTORY) / 'binary' / trial_name / 'kilosort4'

            if not results_dir.exists():
                print(f"Kilosort directory not found for trial: {trial_name}")
                return

            # Load sorting results
            ops, st, clu, similar_templates, is_ref, est_contam_rate, kept_spikes = load_sorting(results_dir)

            cluster_labels = np.unique(clu)
            fs = ops['fs']  # Sampling rate

            # Option 1: Use existing labels as a starting point
            label_good = is_ref.copy()

            # Apply custom criteria
            if custom_criteria is not None:
                label_good = np.logical_and(label_good, custom_criteria(cluster_labels, st, clu, est_contam_rate, fs))
            else:
                # Default criteria: Contamination rate < 0.2 and firing rate >= 1 Hz
                contam_good = est_contam_rate < 0.2
                fr_good = np.zeros(cluster_labels.size, dtype=bool)
                for i, c in enumerate(cluster_labels):
                    spikes = st[clu == c]
                    fr = spikes.size / ((spikes.max() - spikes.min()) / fs)
                    if fr >= 1:
                        fr_good[i] = True
                label_good = np.logical_and(label_good, contam_good, fr_good)

            # Save the updated labels
            ks_labels = ['good' if b else 'mua' for b in label_good]

            # Paths to save the labels
            save_1 = results_dir / 'cluster_KSLabel.tsv'
            save_2 = results_dir / 'cluster_group.tsv'

            # Backup existing labels
            if not (results_dir / 'cluster_KSLabel_backup.tsv').exists():
                shutil.copyfile(save_1, results_dir / 'cluster_KSLabel_backup.tsv')

            # Write to .tsv files
            with open(save_1, 'w') as f:
                f.write('cluster_id\tKSLabel\n')
                for i, p in enumerate(ks_labels):
                    f.write(f'{i}\t{p}\n')
            shutil.copyfile(save_1, save_2)

            print(f"Custom labels applied and saved for trial: {trial_name}")

        except Exception as e:
            print(f"Error applying custom labels to trial {trial_name}: {e}")

    def run_kilosort_trial_summary(self, new_settings = None, custom_criteria=None,**kwargs): # runs kilosort in a loop through all of the binary files saved to the SAVE_DIRECTORY/binary folder

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
                
                elif new_settings == "vf_settings": # use these settings to run kilosort on trials with Von Frey. these seems to produce the best spike clusters.
                    settings = DEFAULT_SETTINGS
                    settings = {'data_dir': os.path.join(self.SAVE_DIRECTORY, 'binary', folder), 'n_chan_bin': 32,'nblocks':0,"batch_size":1500000}

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
                except Exception as e:
                    print(f'Error saving kilosort outputs for {folder}: {e}')
                    pass

                try: # plotting the kilosort outputs
                    rcParams['axes.spines.top'] = False
                    rcParams['axes.spines.right'] = False
                    gray = .5 * np.ones(3)

                    fig = plt.figure(figsize=(10,10), dpi=100)
                    fig.suptitle(os.path.basename(folder))
                    grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.5)

                    try:  # Attempt to plot the drift estimate
                        ax = fig.add_subplot(grid[0, 0])
                        ax.plot(np.arange(0, ops['Nbatches']) * 2, dshift)
                        ax.set_xlabel('time (sec.)')
                        ax.set_ylabel('drift (um)')
                    except Exception as e:  # Handle any errors in plotting the drift estimate
                        ax = fig.add_subplot(grid[0, 0])
                        ax.text(0.5, 0.5, "(no drift estimate,\n nblocks=0)", 
                                ha='center', va='center', fontsize=12, color='gray')
                        ax.set_axis_off()  # Optionally remove axes if you want just the text


                    # Define the specific time window in seconds
                    time_window_start = 10  # Start time in seconds
                    time_window_end = 20    # End time in seconds

                    # Convert the time window to indices
                    t_start_index = np.searchsorted(st, time_window_start * 30000)  # Convert to sample index
                    t_end_index = np.searchsorted(st, time_window_end * 30000)

                    # Plot the spikes within the specific time window
                    ax = fig.add_subplot(grid[0,1:])
                    ax.scatter(st[t_start_index:t_end_index]/30000., 
                            chan_best[clu[t_start_index:t_end_index]], 
                            s=0.5, color='k', alpha=0.25)

                    # Set x-axis limits to the specified time window
                    ax.set_xlim([time_window_start, time_window_end])
                    ax.set_ylim([chan_map.max(), 0])
                    ax.set_xlabel('time (sec.)')
                    ax.set_ylabel('channel')
                    ax.set_title(f'spikes from units ({time_window_start}-{time_window_end} sec)')

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
                    plt.savefig(os.path.join(self.SAVE_DIRECTORY, 'figures', f'spike_summary_{os.path.basename(folder)}.png'), dpi=150)
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

            # apply my own criteria for good & bad units
            trial_name = os.path.basename(folder)
            self.apply_custom_labels_to_trial(trial_name,custom_criteria=custom_criteria)
    

    def extract_kilosort_outputs(self):
        """
        Load specific Kilosort output files from all trial folders into self.kilosort_results.
        """
        try:
            # Assuming that the results are saved in self.SAVE_DIRECTORY/binary/trial_folder/kilosort4
            binary_dir = Path(self.SAVE_DIRECTORY) / 'binary'
            trial_folders = [f for f in binary_dir.iterdir() if f.is_dir()]
            
            if not trial_folders:
                print("No trial folders found.")

                return
            
            # Initialize the main results dictionary if not already done
            if not hasattr(self, 'kilosort_results'):
                self.kilosort_results = {}
            
            for trial_folder in trial_folders:
                try:
                    trial_name = trial_folder.name
                    results_dir = trial_folder / 'kilosort4'
                    
                    if not results_dir.exists():
                        print(f"Kilosort directory not found for trial: {trial_name}")
                        continue

                    # Load the Kilosort output files
                    ops = np.load(results_dir / 'ops.npy', allow_pickle=True).item()
                    camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values
                    contam_pct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values
                    chan_map = np.load(results_dir / 'channel_map.npy')
                    templates = np.load(results_dir / 'templates.npy')
                    chan_best = (templates**2).sum(axis=1).argmax(axis=-1)
                    chan_best = chan_map[chan_best]
                    amplitudes = np.load(results_dir / 'amplitudes.npy')
                    st = np.load(results_dir / 'spike_times.npy')
                    clu = np.load(results_dir / 'spike_clusters.npy')
                    firing_rates = np.unique(clu, return_counts=True)[1] * 30000 / st.max()
                    dshift = ops['dshift']

                    # Store the loaded data into the results dictionary
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
                    print(f"Error loading Kilosort outputs for trial {trial_folder.name}: {trial_error}: has kilosort been run?")
                    continue

        except Exception as e:
            print(f"Error processing Kilosort outputs: {e}")