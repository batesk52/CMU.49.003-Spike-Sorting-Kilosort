import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import pytest

# Import your modules from automations
from automations import RM1, analysis_functions, SpikeInterface_wrapper, Kilosort_wrapper

# ----------------------------
# Fixture: Create Dummy File Structure
# ----------------------------

@pytest.fixture
def dummy_file_structure(tmp_path):
    """
    Create a temporary file structure mimicking:
    
    tmp_path/
      Sample_Rat_Input/
          RatID_TermExpSchedule.xlsx   <-- metadata file (with at least 7 rows)
          RatID/
              RatID_TermExpSchedule.xlsx  <-- copy for RM1.Rat to find
              Trial_1/
                  VF_1_240911_164342.rhd
              Trial_2/
                  VF_2_240911_165039.rhd
      Sample_Rat_output/
          binary/
          figures/
          tables/
    """
    # Create input and output directories inside tmp_path
    input_dir = tmp_path / "Sample_Rat_Input"
    output_dir = tmp_path / "Sample_Rat_output"
    input_dir.mkdir()
    output_dir.mkdir()
    (output_dir / "binary").mkdir()
    (output_dir / "figures").mkdir()
    (output_dir / "tables").mkdir()
    
    # Define rat ID and create the metadata Excel file at the input root.
    rat_id = "RatID"
    metadata_file = input_dir / f"{rat_id}_TermExpSchedule.xlsx"
    
    # Create a dummy DataFrame with 7 rows.
    # Row index 4 will contain the experiment note; row 5 will be the header.
    dummy_data = [
        ["", "", ""],
        ["", "", ""],
        ["", "", ""],
        ["", "", ""],
        ["st_experiment_note", "Note_Value", ""],  # Row index 4: experiment note
        ["Trial Number", "Freq. (Hz)", "Other"],     # Row index 5: header row
        [1, 5, "dummy"]                              # Row index 6: data row
    ]
    dummy_df = pd.DataFrame(dummy_data)
    
    # Write the same dummy metadata to all required sheets.
    with pd.ExcelWriter(metadata_file) as writer:
        dummy_df.to_excel(writer, sheet_name="ST", index=False, header=False)
        dummy_df.to_excel(writer, sheet_name="DRGS", index=False, header=False)
        dummy_df.to_excel(writer, sheet_name="SC", index=False, header=False)
        dummy_df.to_excel(writer, sheet_name="QST", index=False, header=False)
    
    # Create the RatID folder inside the input directory and copy the metadata file there.
    rat_folder = input_dir / rat_id
    rat_folder.mkdir()
    shutil.copy(metadata_file, rat_folder / f"{rat_id}_TermExpSchedule.xlsx")
    
    # Create trial folders and dummy .rhd files.
    trial1 = rat_folder / "Trial_1"
    trial1.mkdir()
    trial2 = rat_folder / "Trial_2"
    trial2.mkdir()
    
    dummy_rhd_content = b"dummy"  # Content isn't parsed due to monkeypatching.
    (trial1 / "VF_1_240911_164342.rhd").write_bytes(dummy_rhd_content)
    (trial2 / "VF_2_240911_165039.rhd").write_bytes(dummy_rhd_content)
    
    return input_dir, output_dir, rat_id

# ----------------------------
# Dummy Recording Class for RM1 Module
# ----------------------------

class DummyRecordingForRM1:
    """A minimal dummy recording to simulate Intan recordings."""
    def __init__(self, data, sampling_rate=30000, channel_ids=None):
        self._data = np.array(data)
        self._sampling_rate = sampling_rate
        self._channel_ids = channel_ids if channel_ids is not None else ['ANALOG-IN-2']
        self._properties = {
            'gain_to_uV': [1.0] * len(self._channel_ids),
            'offset_to_uV': [0.0] * len(self._channel_ids)
        }
    def get_sampling_frequency(self):
        return self._sampling_rate
    def get_channel_ids(self):
        return self._channel_ids
    def get_traces(self, channel_ids, return_scaled=False):
        return self._data
    def get_property(self, prop):
        return self._properties[prop]
    def frame_slice(self, start_frame, end_frame):
        sliced_data = self._data[start_frame:end_frame]
        return DummyRecordingForRM1(sliced_data, self._sampling_rate, self._channel_ids)
    def get_num_frames(self):
        return len(self._data)
    def remove_channels(self, channels):
        return self
    def select_channels(self, channels):
        return self

# ----------------------------
# Dummy read_intan Function
# ----------------------------

def dummy_read_intan(rhd_file_path, stream_id='0'):
    """
    Dummy function to simulate reading an Intan .rhd file.
    Returns a DummyRecordingForRM1 with a signal of 2,100,000 samples:
    first 1,050,000 samples at 100, last 1,050,000 samples at 300.
    """
    low = np.full(1050000, 100, dtype=float)
    high = np.full(1050000, 300, dtype=float)
    data = np.concatenate([low, high])
    return DummyRecordingForRM1(data, sampling_rate=30000, channel_ids=['ANALOG-IN-2'])

# ----------------------------
# Test Function for the Full Pipeline
# ----------------------------

def test_full_pipeline(dummy_file_structure, monkeypatch):
    # Retrieve temporary input and output directories and rat_id.
    input_dir, output_dir, rat_id = dummy_file_structure
    
    # Monkeypatch RM1.read_intan to use our dummy function.
    monkeypatch.setattr(RM1, "read_intan", dummy_read_intan)
    
    # Define the probe file path using your permanent location.
    probe_path = r"D:\SynologyDrive\CMU.80 Data\88 Analyzed Data\88.001 A1x32-Edge-5mm-20-177-A32\A1x32-Edge-5mm-20-177-A32.prb"
    
    # Instantiate the Rat object.
    # RM1.Rat expects metadata at: DATA_DIRECTORY / rat_id / f"{rat_id}_TermExpSchedule.xlsx"
    rat = RM1.Rat(str(input_dir), probe_path, rat_id)
    
    # For testing, manually assign a dummy analog recording for trial "VF_1_TEST".
    dummy_recording = DummyRecordingForRM1(
        np.concatenate([np.full(1050000, 100, dtype=float), np.full(1050000, 300, dtype=float)]),
        sampling_rate=30000,
        channel_ids=['ANALOG-IN-2']
    )
    rat.analog_data = {"VF_1_TEST": dummy_recording}
    
    # Simulate qst_trial_notes with index 1 (extracted from trial name "VF_1_TEST").
    rat.qst_trial_notes = pd.DataFrame({'Freq. (Hz)': [5]}, index=[1])
    
    # Create dummy wrappers using the output directory.
    si = SpikeInterface_wrapper(rat, output_dir)
    ks = Kilosort_wrapper(output_dir, probe_path)
    # Simulate a dummy kilosort result for trial "VF_1_TEST".
    ks.kilosort_results = {
        "VF_1_TEST": {
            'spike_times': np.array([100, 200, 300, 400, 500]),  # in samples
            'spike_clusters': np.array([1, 1, 2, 2, 1]),
            'ops': {'fs': 30000}
        }
    }
    
    # Instantiate the VonFreyAnalysis object.
    vfa = analysis_functions.VonFreyAnalysis(rat, si, ks)
    
    # Run the analysis for trial "VF_1_TEST".
    results = vfa.analyze_subwindows(TRIAL_NAMES=["VF_1_TEST"], subwindow_width=0.5, corr_threshold=0.1)
    
    # Assert that results contain key "VF_1_TEST" and that the DataFrames are not empty.
    assert isinstance(results, dict)
    assert "VF_1_TEST" in results
    trial_results = results["VF_1_TEST"]
    assert "avg_voltage_df" in trial_results
    assert "firing_rates_df" in trial_results
    assert not trial_results["avg_voltage_df"].empty
    assert not trial_results["firing_rates_df"].empty
    
    print("Full pipeline test passed with dummy file structure and probe from specified file.")

# ----------------------------
# Run tests if script is executed directly.
# ----------------------------
if __name__ == "__main__":
    pytest.main([__file__])
