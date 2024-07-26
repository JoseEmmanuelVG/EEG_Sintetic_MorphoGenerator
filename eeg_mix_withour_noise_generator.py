import mne
import numpy as np
import random  
import matplotlib.pyplot as plt
from scipy import signal
from mix_generator import generate_spike_wave_group
import pyedflib

sfreq = 200  # Sampling frequency in Hz
times = np.arange(0, 10, 1/sfreq)  # Time vector in seconds
n_channels = 10  # Number of channels

def pink_noise(n_samples):
    """Generates pink noise using the IIR filter method."""
    x = np.random.randn(n_samples)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    n_taps = max(len(a), len(b))

    zi = signal.lfilter_zi(b, a)
    noise = signal.lfilter(b, a, x, zi=zi*x[0])[0]
    noise = noise/np.max(np.abs(noise)) * random.uniform(15, 17.5)  # Normalize and scale to 20-60 µV
    return noise

def transition_noise(n_samples, transition_length=50):
    """Generates pink noise with a smooth transition at the start."""
    noise = pink_noise(n_samples + transition_length)
    window = signal.windows.hann(transition_length * 10)
    noise[:transition_length] *= window[:transition_length]  # Attenuate the start of the noise
    return noise[transition_length:]

# Generate the data for all the channels
data = np.zeros((n_channels+1, len(times)))  # One extra channel for the noise
for i in range(n_channels):
    n_groups = 1  # Generate the same number of spike-wave groups for all the channels
    start_indices = []
    for _ in range(n_groups):
        group_data = generate_spike_wave_group(sfreq)
        # Add pink noise to the spike-wave group data
        group_data += 0.1 * pink_noise(len(group_data))  
        start_index = random.randint(0, len(times) - len(group_data) - 1)
        start_indices.append(start_index)
        data[i, start_index:start_index + len(group_data)] += group_data

    # Add pink noise in sections of the signal where there are no spike-wave groups
    start_indices.sort()  # Ensure the start indices are in ascending order
    if start_indices[0] > 0:
        data[i, :start_indices[0]] += 10*pink_noise(start_indices[0])  # Add noise at the beginning
    for j in range(len(start_indices) - 1):
        if start_indices[j + 1] - start_indices[j] > len(group_data):
            # If there's a gap between spike-wave groups larger than the group length, add noise there
            data[i, start_indices[j] + len(group_data):start_indices[j + 1]] += 7*pink_noise(start_indices[j + 1] - start_indices[j] - len(group_data))
    if len(times) - start_indices[-1] > len(group_data):
        # If there's a space at the end after the last spike-wave group, add noise there
        data[i, start_indices[-1] + len(group_data):] += 10*transition_noise(len(times) - start_indices[-1] - len(group_data))

# Add a channel of just noise
data[n_channels, :] = 10*pink_noise(len(times))

# Define channel names and add the noise channel
ch_names = ['EEG ' + str(i + 1) for i in range(n_channels)]
ch_names.append('NOISE')

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * (n_channels+1))
raw = mne.io.RawArray(data, info)

scalings = {'eeg': 90}  # Set the range of amplitude for the EEG channels in 100 µV

# Create a matrix for bipolar data
bipolar_data = np.zeros((n_channels, len(times))) 

# Create a list for bipolar channel names
bipolar_ch_names = []

# Calculating bipolar data and channel names
for i in range(n_channels):
    bipolar_data[i, :] = data[i, :] - data[(i+1) % n_channels, :]  # operator % causes channel 10 to be subtracted from channel 1
    bipolar_ch_names.append('BIP ' + str(i+1) + '-' + str((i+1) % n_channels + 1))  # generate channel names

# Create a new Info for bipolar data
bipolar_info = mne.create_info(ch_names=bipolar_ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)

# Create a new Raw object for bipolar data
bipolar_raw = mne.io.RawArray(bipolar_data, bipolar_info)

# Plot
raw.plot(duration=10, n_channels=n_channels, scalings=scalings)
bipolar_raw.plot(duration=10, n_channels=n_channels, scalings=scalings)
plt.show(block=True)

raw.plot(duration=10, n_channels=n_channels, scalings=scalings)
bipolar_raw.plot(duration=10, n_channels=n_channels, scalings=scalings)
plt.show(block=True)

# Write normal EEG to EDF
f = pyedflib.EdfWriter('eeg_normal.edf', n_channels=n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)
channel_info = []
data_list = []
for i in range(n_channels):
    ch_dict = {'label': ch_names[i], 'dimension': 'uV', 'sample_rate': sfreq, 'physical_max': 100, 'physical_min': -100, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
    channel_info.append(ch_dict)
    data_list.append(data[i])
f.setSignalHeaders(channel_info)
f.writeSamples(data_list)
f.close()

# Write bipolar EEG to EDF
f = pyedflib.EdfWriter('eeg_bipolar.edf', n_channels=n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)
channel_info = []
data_list = []
for i in range(n_channels):
    ch_dict = {'label': bipolar_ch_names[i], 'dimension': 'uV', 'sample_rate': sfreq, 'physical_max': 100, 'physical_min': -100, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
    channel_info.append(ch_dict)
    data_list.append(bipolar_data[i])
f.setSignalHeaders(channel_info)
f.writeSamples(data_list)
f.close()