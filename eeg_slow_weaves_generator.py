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
    noise = noise/np.max(np.abs(noise)) * random.uniform(20, 60)  # Normalize and scale to 20-60 µV
    return noise


# Generate the data for all the channels
data = np.zeros((n_channels+1, len(times)))  # One extra channel for the noise
for i in range(n_channels):
    data[i, :] += 10*pink_noise(len(times))
    n_groups = 1  # Generate the same number of spike-wave groups for all the channels
    for _ in range(n_groups):
        group_data = generate_spike_wave_group(sfreq)
        start_index = random.randint(0, len(times) - len(group_data) - 1)
        data[i, start_index:start_index + len(group_data)] += group_data

# Add a channel of just noise
data[n_channels, :] = 10*pink_noise(len(times))

# Define channel names and add the noise channel
ch_names = ['EEG ' + str(i + 1) for i in range(n_channels)]
ch_names.append('NOISE')

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * (n_channels+1))
raw = mne.io.RawArray(data, info)

scalings = {'eeg': 90}  # Set the range of amplitude for the EEG channels in 100 µV

# Crear una matriz para los datos bipolares
bipolar_data = np.zeros((n_channels, len(times))) 

# Crear una lista para los nombres de los canales bipolares
bipolar_ch_names = []

# Calcular los datos bipolares y los nombres de los canales
for i in range(n_channels):
    bipolar_data[i, :] = data[i, :] - data[(i+1) % n_channels, :]  # el operador % hace que el canal 10 se reste con el canal 1
    bipolar_ch_names.append('BIP ' + str(i+1) + '-' + str((i+1) % n_channels + 1))  # generar los nombres de los canales

# Crear una nueva Info para los datos bipolares
bipolar_info = mne.create_info(ch_names=bipolar_ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)

# Crear un nuevo objeto Raw para los datos bipolares
bipolar_raw = mne.io.RawArray(bipolar_data, bipolar_info)

# Plot
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
