# plots.py

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def interactive_trace(recording, channel_id='A-000', downsample_factor=10, title=None):
    """
    Plots the voltage trace using Plotly.

    Parameters:
    - recording: the recording object from which to extract data
    - channel_id: string, the ID of the channel to plot
    - downsample_factor: int, factor by which to downsample the data
    - title: string, optional custom title for the plot
    """
    # Get channel IDs and find the index of the specified channel
    channel_ids = recording.get_channel_ids()
    try:
        channel_idx = list(channel_ids).index(channel_id)
    except ValueError:
        raise ValueError(f"Channel '{channel_id}' not found in the recording.")

    # Retrieve gain and offset for the specified channel
    gain_to_uV_array = recording.get_property('gain_to_uV')
    offset_to_uV_array = recording.get_property('offset_to_uV')
    gain_to_uV = gain_to_uV_array[channel_idx]
    offset_to_uV = offset_to_uV_array[channel_idx]
    print(f"gain_to_uV: {gain_to_uV}, offset_to_uV: {offset_to_uV}")

    # Retrieve the trace for the specified channel
    trace = recording.get_traces(channel_ids=[channel_id])

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
    time = time[::downsample_factor]
    voltage = voltage[::downsample_factor]

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=time,
        y=voltage,
        mode='lines',
        name='Voltage Trace',
        line=dict(color='blue')
    ))

    # Determine the title
    if title is None:
        plot_title = f'Voltage vs Time for Channel {channel_id}'
    else:
        plot_title = title

    # Set plot layout
    fig.update_layout(
        title=plot_title,
        xaxis_title='Time (s)',
        yaxis_title='Voltage (µV)',
        showlegend=True,
        # width=1900,
        # height=800
    )

    # Show the interactive plot
    fig.show()


def static_trace(recording, channel_id='A-000', start_time=0, end_time=10, title=None):
    """
    Plots the voltage trace using Matplotlib.

    Parameters:
    - recording: the recording object from which to extract data
    - channel_id: string, the ID of the channel to plot
    - start_time: float, start time in seconds for the plot
    - end_time: float, end time in seconds for the plot
    - title: string, optional custom title for the plot
    """
    # Get channel IDs and find the index of the specified channel
    channel_ids = recording.get_channel_ids()
    try:
        channel_idx = list(channel_ids).index(channel_id)
    except ValueError:
        raise ValueError(f"Channel '{channel_id}' not found in the recording.")

    # Retrieve gain and offset for the specified channel
    gain_to_uV_array = recording.get_property('gain_to_uV')
    offset_to_uV_array = recording.get_property('offset_to_uV')
    gain_to_uV = gain_to_uV_array[channel_idx]
    offset_to_uV = offset_to_uV_array[channel_idx]
    print(f"gain_to_uV: {gain_to_uV}, offset_to_uV: {offset_to_uV}")

    # Retrieve the trace for the specified channel
    trace = recording.get_traces(channel_ids=[channel_id], return_scaled=True)

    # Convert raw data to voltage (if needed)
    voltage = trace  # Assuming return_scaled=True already gives voltage

    # Get sampling frequency and calculate time axis
    sampling_rate = recording.get_sampling_frequency()
    n_samples = len(voltage)
    time = np.arange(n_samples) / sampling_rate  # Time in seconds

    # Flatten the arrays    
    time = time.flatten()
    voltage = voltage.flatten()  

    # Plotting a time window
    start_idx = int(start_time * sampling_rate)
    end_idx = int(end_time * sampling_rate)

    time_window = time[start_idx:end_idx]
    voltage_window = voltage[start_idx:end_idx]

    plt.figure(figsize=(15, 5))
    plt.plot(time_window, voltage_window, label=f'Channel {channel_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (µV)')
    
    # Determine the title
    if title is None:
        plot_title = f'Voltage Trace for Channel {channel_id} ({start_time}-{end_time} seconds)'
    else:
        plot_title = title
    
    plt.title(plot_title)
    plt.legend()
    plt.tight_layout()
    plt.show()
