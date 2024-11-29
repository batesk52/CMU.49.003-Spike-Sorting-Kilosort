# Standard imports
from pathlib import Path
import os
import pandas as pd
import numpy as np
from kilosort import io
import matplotlib.pyplot as plt

# Custom imports
from automations import RM1
from automations import Kilosort_wrapper

PROBE_DIRECTORY = Path(r'D:\Data\CMU.80 Data\88 Analyzed Data\88.001 A1x32-Edge-5mm-20-177-A32\A1x32-Edge-5mm-20-177-A32.prb')

SAVE_DIRECTORY = Path(r'D:\Data\CMU.80 Data\88 Analyzed Data\88.003 Initial Analysis, DW323')
DATA_DIRECTORY = Path(r'D:\Data\CMU.80 Data\82 External Data\82.002 Sample Rat Data from RM1 Project')

# Create path if it doesn't exist
DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)

DW323 = RM1.Rat(DATA_DIRECTORY, PROBE_DIRECTORY, "DW323")
DW323.get_sc_data()

# Select the trial
TRIAL_DRGS = "DRGS_11_240911_160638"
recording = DW323.sc_data[TRIAL_DRGS]

# Get channel IDs and find the index of 'A-000'
channel_ids = recording.get_channel_ids()
try:
    channel_idx = list(channel_ids).index('A-000')
except ValueError:
    raise ValueError("Channel 'A-000' not found in the recording.")

# Retrieve gain and offset for 'A-000'
gain_to_uV_array = recording.get_property('gain_to_uV')
offset_to_uV_array = recording.get_property('offset_to_uV')
gain_to_uV = gain_to_uV_array[channel_idx]
offset_to_uV = offset_to_uV_array[channel_idx]
print(gain_to_uV, offset_to_uV)

# Retrieve the trace for channel 'A-000'
trace = recording.get_traces(channel_ids=['A-000'])

# Convert raw data to voltage
voltage = trace * gain_to_uV + offset_to_uV

# Get sampling frequency and calculate time axis
sampling_rate = recording.get_sampling_frequency()
n_samples = len(voltage)
time = np.arange(n_samples) / sampling_rate  # Time in seconds

# Flatten the arrays
time = time.flatten()
voltage = voltage.flatten()

# Optional: Clean data to remove NaNs or Infs
mask = ~np.isnan(time) & ~np.isnan(voltage) & ~np.isinf(time) & ~np.isinf(voltage)
time = time[mask]
voltage = voltage[mask]

# Optional: Downsample data
downsample_factor = 10  # Adjust as needed
time = time[::downsample_factor]
voltage = voltage[::downsample_factor]

# Import Plotly and create the plot
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scattergl(
    x=time,
    y=voltage,
    mode='lines',
    name='Voltage Trace',
    line=dict(color='blue')
))

# Set plot layout
fig.update_layout(
    title='Voltage vs Time',
    xaxis_title='Time (s)',
    yaxis_title='Voltage (µV)',
    showlegend=True,
    width=1900,
    height=800
)

# Show the interactive plot
fig.show()
