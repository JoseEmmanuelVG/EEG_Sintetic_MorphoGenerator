import numpy as np
import random
from scipy.signal import gaussian
import datetime
# edf
import pyedflib
import mne



# Funciones para la generación de olas
def generate_spike(amplitude, duration, sfreq):
    if isinstance(amplitude, (list, tuple)):
        amplitude = random.uniform(*amplitude)
    n_samples = int(duration * sfreq)
    std_dev = n_samples / 5  # Ajustar para un pico más agudo
    spike = gaussian(n_samples, std=std_dev)
    return amplitude * spike / np.max(spike)

# Funciones para la generación de ondas lemtas
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

# Funciones para la generación de puntas-onda lenta
def generate_spike_slow_wave(sfreq, amplitude_spike_range, duration_spike_range, amplitude_slow_range, duration_slow_range):
    amplitude_spike = random.uniform(*amplitude_spike_range)
    duration_spike = random.uniform(*duration_spike_range)
    spike_wave = generate_spike(amplitude_spike, duration_spike, sfreq)
    amplitude_slow = random.uniform(*amplitude_slow_range)
    duration_slow = random.uniform(*duration_slow_range)
    slow_wave = generate_slow_wave(amplitude_slow, duration_slow, sfreq)
    combined_wave = np.concatenate((spike_wave, slow_wave))
    return combined_wave


# Definir las bandas de frecuencia del EEG
delta_band = [0, 4]  # Delta rhythm: 0-4 Hz
theta_band = [4, 8]  # Theta rhythm: 4-8 Hz
alpha_band = [8, 12]  # Alpha rhythm: 8-12 Hz
beta_band = [12, 30]  # Beta rhythm: 12-30 Hz
gamma_band = [30, 70]  # Gamma rhythm: 30-70 Hz

# Definir la duración y la frecuencia de muestreo de la señal EEG
duration = 10  # seconds
sampling_freq = 500  # Hz
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


def save_to_edf(data, sfreq, channel_names, filename='output.edf'):
    """
    Save the given data to an EDF file.

    Args:
        data (list of arrays): EEG channel data.
        sfreq (int): Sampling frequency.
        channel_names (list of str): Names of EEG channels.
        filename (str): Name of the EDF file to be created.
    """
    f = pyedflib.EdfWriter(filename, len(channel_names), file_type=pyedflib.FILETYPE_EDFPLUS)

    header = {
        'technician': 'Simulador de ondas epilépticas',
        'recording_additional': '',
        'patientname': 'Simulated',
        'patient_additional': '',
        'patientcode': '',
        'equipment': 'Simulator',
        'admincode': '',
        'gender': '',  # 
        'sex': '',  # 
        'startdate': datetime.datetime.now(),
        'birthdate': ''
    }

    f.setHeader(header)

    for i, ch_name in enumerate(channel_names):
        f.setSignalHeader(i, {'label': ch_name, 'dimension': 'uV', 'sample_rate': sfreq, 'physical_max': 1000, 'physical_min': -1000, 'digital_max': 1000, 'digital_min': -1000, 'transducer': 'Simulated EEG', 'prefilter': ''})

    f.writeSamples(data)
    f.close()

# Funciones para la visualización de señales
def create_mne_visualization(data, sfreq, n_channels, output_filename):
    # Crear objeto Raw de MNE a partir de datos
    info = mne.create_info(ch_names=[f'ch_{i}' for i in range(n_channels)], sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)

    # Ajustar el escalado para EEG en Volts (por ejemplo, 200 microvoltios = 200e-6 Volts)
    scalings = {'eeg': 900e-1}

    # Crear la figura con el escalado ajustado
    fig_mne = raw.plot(n_channels=n_channels, scalings=scalings, show=False)
    fig_mne.set_size_inches(10, 8)
    fig_mne.savefig(output_filename, dpi=600)

# Funciones para guardar en formato TXT
def save_to_txt(data, channel_names, filename):
        """
        Save the given data to a TXT file with channels in columns.

        Args:
            data (numpy.ndarray): The EEG data to save, expected shape is (n_channels, n_samples).
            channel_names (list of str): The names of the EEG channels.
            filename (str): The output filename for the TXT file.
        """
        with open(filename, "w") as file:
            # Escribir la cabecera con los nombres de los canales
            file.write("\t".join(channel_names) + "\n")
            
            # Escribir los datos
            for i in range(data.shape[1]):  # Suponiendo que los datos están en forma (n_canales, n_muestras)
                values = [str(data[ch, i]) for ch in range(data.shape[0])]
                file.write("\t".join(values) + "\n")

