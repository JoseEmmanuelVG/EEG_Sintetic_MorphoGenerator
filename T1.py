import mne
import numpy as np
import matplotlib.pyplot as plt
import random

from scipy.signal import gaussian

sfreq = 1000  # Frecuencia de muestreo en Hz
times = np.arange(0, 10, 1/sfreq)  # Vector de tiempo en segundos
n_channels = 10  # Número de canales

# Función para generar una onda punta
def generate_spike(amplitude, duration, sfreq):
    n_samples = int(duration * sfreq)
    spike = gaussian(n_samples, std=n_samples/7)  # Genera una gaussiana
    return amplitude * spike / np.max(spike)  # Normaliza a la amplitud deseada

# Función para generar ondas puntuales en un canal
def generate_spikes_channel(n_spikes, times):
    channel_data = np.zeros_like(times)
    for _ in range(n_spikes):
        amplitude = random.uniform(0.5, 1.0) * 100  # Amplitud en microvoltios
        duration = random.uniform(0.02, 0.07)  # Duración en segundos
        spike = generate_spike(amplitude, duration, sfreq)
        start_index = random.randint(0, len(times) - len(spike) - 1)
        channel_data[start_index:start_index + len(spike)] += spike
    return channel_data

# Generar los datos para todos los canales
data = np.zeros((n_channels, len(times)))
for i in range(n_channels):
    n_spikes = random.randint(5, 20)
    data[i, :] = generate_spikes_channel(n_spikes, times)

# Crear un objeto RawArray de MNE con los datos EEG sintéticos
ch_names = ['EEG ' + str(i + 1) for i in range(n_channels)]
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)
raw = mne.io.RawArray(data, info)

# Plotear los datos EEG sintéticos
raw.plot(duration=10, n_channels=n_channels)
plt.show(block=True)