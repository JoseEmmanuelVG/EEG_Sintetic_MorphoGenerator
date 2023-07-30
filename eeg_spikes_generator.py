import mne
import numpy as np
import random  
import matplotlib.pyplot as plt
from scipy import signal
from spike_generator import generate_spike, generate_spikes_channel
import pyedflib

sfreq = 200  # Frecuencia de muestreo en Hz
times = np.arange(0, 10, 1/sfreq)  # Vector de tiempo en segundos
n_channels = 10  # Número de canales

# Generar los datos para todos los canales
data = np.zeros((n_channels, len(times)))
for i in range(n_channels):
    # Generate pink noise baseline EEG signal
    baseline = np.random.normal(size=len(times))
    baseline = signal.lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786], 
                              [1, -2.494956002, 2.017265875, -0.522189400], 
                              baseline)
    baseline = baseline/np.max(np.abs(baseline)) * random.uniform(20, 60)  # Normalization and scaling
    n_spikes = random.randint(5, 20)
    data[i, :] = generate_spikes_channel(n_spikes, times, sfreq, baseline)

# Create an MNE RawArray object with the synthetic EEG data
ch_names = ['EEG ' + str(i + 1) for i in range(n_channels)]
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)
raw = mne.io.RawArray(data, info)

# Plotting of synthetic EEG data
scalings = {'eeg': 100}  # Establece el rango de amplitud para los canales EEG en 100 µV

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
