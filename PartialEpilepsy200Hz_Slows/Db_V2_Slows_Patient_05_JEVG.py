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


# Funciones para la generación de olas
def generate_slow_wave(amplitude_wave, duration, sfreq, frequency=1.0, cutoff_range=(-5, 2)):
    period = 1 / frequency
    samples_per_cycle = int(period * sfreq)
    samples_needed = max(samples_per_cycle // 2, int(duration * sfreq))
    t = np.linspace(0, samples_needed / sfreq, samples_needed, endpoint=False)
    full_wave = amplitude_wave * np.sin(2 * np.pi * frequency * t)
    peak_index = np.argmax(full_wave)
    cross_lower = np.where((full_wave[peak_index:] <= cutoff_range[1]) & (full_wave[peak_index:] >= cutoff_range[0]))[0]
    if len(cross_lower) == 0:
        cutoff_index = len(full_wave) - 1
    else:
        cutoff_index = peak_index + cross_lower[0]
    return full_wave[:cutoff_index+1]


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
eeg_signal *= 100  # Ajusta la escala de amplitud al rango deseado


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
# Definición de los canales
channels = [
    "Fp1", "F7", "T3", "T5", "Fp2", "F8", "T4", "T6", "F3", "C3", 
    "P3", "O1", "F4", "C4", "P4", "O2", "Fz", "Cz", "Pz"
]
n_channels = len(channels)
total_duration = 100  # total duration in seconds
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
event_channels_list = [
    ["F8", "T4", "T6", "O2"],
    ["F8", "T4", "T6", "O2"],
    ["F8", "T4", "T6", "O2"],
    ["F8", "T4", "T6", "O2"],
    ["F8", "T4", "T6", "O2"],
    ["F8", "T4", "T6", "O2"],
    ["F8", "T4", "T6", "O2"],
    ["F8", "T4", "T6", "O2"],
    ["F8", "T4", "T6", "O2"],
    ["F8", "T4", "T6", "O2"]
]

for case, start in enumerate(range(0, total_duration, interval_duration)):
    event_start = start * sampling_freq + random.randint(0, interval_duration - 3) * sampling_freq
    event_duration = 3 * sampling_freq
    event_channels = event_channels_list[case]

    num_slow_waves = random.randint(1, 6)  # Número de ondas lentas entre 1 y 6
    slow_wave_intervals = np.linspace(0, event_duration, num_slow_waves, endpoint=False)
    for channel_name in event_channels:
        channel_idx = channels.index(channel_name)
        for t in slow_wave_intervals:
            slow_wave = generate_slow_wave(amplitude_wave=random.uniform(40, 80), duration=random.uniform(-0.2, 0.5), sfreq=sampling_freq)
            start_idx = int(event_start + t)
            end_idx = start_idx + len(slow_wave)
            eeg_signals[channel_idx, start_idx:end_idx] += slow_wave

# Crear el objeto Raw de MNE
info = mne.create_info(channels, sfreq=sampling_freq, ch_types='eeg')
raw = mne.io.RawArray(eeg_signals * 1e-6, info)  # Convertir a Voltios

############################# Guardar señales #############################

# Crear carpeta con índice incrementable
base_folder = Path("MonoEEG_Patient_5_Slows_JEVG")
index = 1

while (base_folder / f"{index:03d}").exists():
    index += 1

current_folder = base_folder / f"{index:03d}"
current_folder.mkdir(parents=True)

# Guardar a un archivo .fif en la carpeta creada
fif_filename = current_folder / "MonoEEG_Patient_5_Slows_JEVG_raw.fif"
raw.save(fif_filename, overwrite=True)

# Configuración de escalado para la visualización
scalings = {'eeg': 60}

# Guardar las señales en un archivo .txt
txt_filename = current_folder / "MonoEEG_Patient_5_Slows_JEVG_text.txt"
np.savetxt(txt_filename, (eeg_signals * 1e-6).T, delimiter="\t", header="\t".join(channels))  # Convertir a Voltios

# Guardar las señales en un archivo .edf
edf_filename = current_folder / "MonoEEG_Patient_5_Slows_JEVG_EegFormat.edf"
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
    fig.savefig(current_folder / f'MonoEEG_Pat_5_Sl_JEVG_segment_{i+1}.jpg')
    plt.close(fig)
