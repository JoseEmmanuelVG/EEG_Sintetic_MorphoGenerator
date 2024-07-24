# Generación de señales EEG sintéticas con puntas
import numpy as np
import random
from scipy.signal import gaussian
import datetime
# Importar librerías de MNE y Matplotlib
import mne
import matplotlib.pyplot as plt
# Crear carpeta con índice incrementable para el guardado de señales
import os
from pathlib import Path
import pyedflib

# Funciones para la generación de ondas
def generate_spike(amplitude, duration_range, sfreq):
    duration = random.uniform(*duration_range)
    if isinstance(amplitude, (list, tuple)):
        amplitude = random.uniform(*amplitude)
    n_samples = int(duration * sfreq)
    std_dev = n_samples / 5  # Ajustar para un pico más agudo
    spike = gaussian(n_samples, std=std_dev)
    return amplitude * spike / np.max(spike)


# Definir las bandas de frecuencia del EEG
delta_band = [0, 4]  # Delta rhythm: 0-4 Hz
theta_band = [4, 8]  # Theta rhythm: 4-8 Hz
alpha_band = [8, 12]  # Alpha rhythm: 8-12 Hz
beta_band = [12, 30]  # Beta rhythm: 12-30 Hz
gamma_band = [30, 70]  # Gamma rhythm: 30-70 Hz

# Definir la duración y la frecuencia de muestreo de la señal EEG
duration = 10  # seconds
sampling_freq = 200  # Hz
num_samples = duration * sampling_freq
time = np.arange(0, duration, 1 / sampling_freq)

# Crear señal EEG vacía
eeg_signal = np.zeros(num_samples)

# Generar cada banda de frecuencia
def generate_band(freq_range, amplitude, duration, sampling_freq):
    frequency = np.random.uniform(freq_range[0], freq_range[1])
    phase = np.random.uniform(0, 2 * np.pi)
    time = np.arange(0, duration, 1 / sampling_freq)
    return amplitude * np.sin(2 * np.pi * frequency * time + phase)


eeg_signal += generate_band(delta_band, amplitude=50, duration=duration, sampling_freq=sampling_freq)
eeg_signal += generate_band(delta_band, amplitude=30, duration=duration, sampling_freq=sampling_freq)
eeg_signal += generate_band(delta_band, amplitude=20, duration=duration, sampling_freq=sampling_freq)
eeg_signal += generate_band(delta_band, amplitude=10, duration=duration, sampling_freq=sampling_freq)
eeg_signal += generate_band(delta_band, amplitude=5, duration=duration, sampling_freq=sampling_freq)

# Generar ruido rosa
pink_noise = np.random.randn(num_samples)
pink_noise = np.cumsum(pink_noise)
pink_noise -= np.mean(pink_noise)
pink_noise /= np.std(pink_noise)
eeg_signal += pink_noise

# Generar ruido blanco
white_noise = np.random.randn(num_samples)
white_noise /= np.std(white_noise)
eeg_signal += white_noise

# Generar ruido marrón
brown_noise = np.random.randn(num_samples)
brown_noise = np.cumsum(brown_noise)
brown_noise -= np.mean(brown_noise)
brown_noise /= np.std(brown_noise)
eeg_signal += brown_noise

# Normalizar la señal al rango de amplitud deseado
eeg_signal /= np.max(np.abs(eeg_signal))
eeg_signal *= 100  # Ajustar la escala de amplitud al rango deseado


def generate_eeg_signal(freq_bands, amplitudes, duration=10, sampling_freq=1000, noise_amplitude=1.0):
    """
    Genera una señal de EEG sintética basada en las bandas de frecuencia dadas.

    Args:
        freq_bands (list): Una lista de tuplas que definen las bandas de frecuencia.
        amplitudes (list): Una lista de amplitudes para cada banda de frecuencia.
        duration (int, optional): Duración de la señal en segundos. Por defecto es 10.
        sampling_freq (int, optional): Frecuencia de muestreo en Hz. Por defecto es 1000.
        noise_amplitude (float, optional): Amplitud del ruido aditivo. Por defecto es 1.0.

    Returns:
        numpy.ndarray: La señal de EEG sintética generada.
    """
    num_samples = duration * sampling_freq
    time = np.arange(0, duration, 1 / sampling_freq)

    # Crear señal EEG vacía
    eeg_signal = np.zeros(num_samples)

    for band, amplitude in zip(freq_bands, amplitudes):
        eeg_signal += generate_band(band, amplitude, duration, sampling_freq)

    # Generar ruido rosa
    pink_noise = np.random.randn(num_samples) * noise_amplitude
    pink_noise = np.cumsum(pink_noise)
    pink_noise -= np.mean(pink_noise)
    pink_noise /= np.std(pink_noise)
    eeg_signal += pink_noise

    # Normalizar la señal al rango de amplitud deseado
    eeg_signal /= np.max(np.abs(eeg_signal))
    eeg_signal *= 20  # Ajusta la escala de amplitud al rango deseado

    return eeg_signal






############# Generar señales EEG para cada canal #############
channels = [
    "Fp1", "F7", "T3", "T5", "Fp2", "F8", "T4", "T6", "F3", "C3", 
    "P3", "O1", "F4", "C4", "P4", "O2", "Fz", "Cz", "Pz"
]
n_channels = len(channels)
total_duration = 100  # total duration en segundos
sampling_freq = 200  # Hz

# Crear señal EEG vacía para toda la duración
eeg_signals = np.zeros((n_channels, total_duration * sampling_freq))

# Generar señal para cada canal en intervalos de 10 segundos
interval_duration = 10  # seconds
for i in range(n_channels):
    for start in range(0, total_duration, interval_duration):
        eeg_segment = generate_eeg_signal(
            [delta_band, theta_band, alpha_band, beta_band, gamma_band],
            [random.uniform(75, 100), random.uniform(50, 75), random.uniform(25, 50), random.uniform(10, 25), 5],
            duration=interval_duration,
            sampling_freq=sampling_freq
        )
        eeg_signals[i, start * sampling_freq:(start + interval_duration) * sampling_freq] = eeg_segment

# Introducir eventos en los canales especificados en intervalos de 10 segundos
event_channels = [
    ("Fp2", "F8"), ("Fp2", "F8"), ("P4", "O2"), ("P4", "O2"),
    ("F3", "C3", "Fp2", "F4"), ("F3", "C3", "Fp2", "F4"), ("F3", "C3", "Fp2", "F4"),
    ("F8", "T4"), ("Fp2", "F8"), ("Fp2", "F8")
]

channel_indices = {ch: idx for idx, ch in enumerate(channels)}

for case, start in enumerate(range(0, total_duration, interval_duration)):
    event_start = start * sampling_freq + random.randint(0, interval_duration - 3) * sampling_freq
    event_duration = 3 * sampling_freq
    event_channels_case = event_channels[case]

    num_spikes = random.randint(3, 21)  # Número de puntas (rango) en el evento
    spike_intervals = np.linspace(0, event_duration, num_spikes, endpoint=False)
    
    for t in spike_intervals:
        spike = generate_spike(amplitude=(40, 80), duration_range=(0.02, 0.07), sfreq=sampling_freq)
        start_idx = int(event_start + t)
        end_idx = start_idx + len(spike)
        
        for ch in event_channels_case:
            channel_idx = channel_indices[ch]
            eeg_signals[channel_idx, start_idx:end_idx] += spike

# Crear el objeto Raw de MNE
info = mne.create_info(channels, sfreq=sampling_freq, ch_types='eeg')
raw = mne.io.RawArray(eeg_signals * 1e-6, info)  # Convertir a Voltios


############################# Guardar señales #############################

# Crear carpeta con índice incrementable
base_folder = Path("MonoEEG_Patient_2_Spikes_JEVG")
index = 1

while (base_folder / f"{index:03d}").exists():
    index += 1

current_folder = base_folder / f"{index:03d}"
current_folder.mkdir(parents=True)

# Guardar a un archivo .fif en la carpeta creada
fif_filename = current_folder / "MonoEEG_Patient_2_Spikes_JEVG_raw.fif"
raw.save(fif_filename, overwrite=True)

# Configuración de escalado para la visualización
scalings = {'eeg': 60}

# Guardar las señales en un archivo .txt
txt_filename = current_folder / "MonoEEG_Patient_2_Spikes_JEVG_text.txt"
np.savetxt(txt_filename, (eeg_signals * 1e-6).T, delimiter="\t", header="\t".join(channels))  # Convertir a Voltios

# Guardar las señales en un archivo .edf
edf_filename = current_folder / "MonoEEG_Patient_2_Spikes_JEVG_EegFormat.edf"
edf_file = pyedflib.EdfWriter(str(edf_filename), n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)

channel_info = [{'label': ch, 'dimension': 'uV', 'sample_rate': sampling_freq, 'physical_min': -100.0, 'physical_max': 100.0, 'digital_min': -32768, 'digital_max': 32767} for ch in channels]
edf_file.setSignalHeaders(channel_info)
edf_file.writeSamples(eeg_signals)
edf_file.close()

# Guardar las imágenes correspondientes a cada segmento de 10 segundos
# Crear el objeto Raw de MNE
rawImg = mne.io.RawArray(eeg_signals, info)
for i in range(10):
    start_time = i * 10  # Tiempo de inicio del segmento en segundos
    fig = rawImg.plot(start=start_time, duration=10, n_channels=19, title=f'Synthetic EEG Data Segment {i+1}', scalings=scalings, show=False)
    fig.savefig(current_folder / f'MonoEEG_Pat_2_Sp_JEVG_segment_{i+1}.jpg')
    plt.close(fig)