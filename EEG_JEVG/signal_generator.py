import numpy as np
import random
from scipy.signal import gaussian
import datetime


# edf
import pyedflib

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
        f.setSignalHeader(i, {'label': ch_name, 'dimension': 'uV', 'sample_rate': sfreq, 'physical_max': 1000, 'physical_min': -1000, 'digital_max': 32767, 'digital_min': -32768, 'transducer': 'Simulated EEG', 'prefilter': ''})

    f.writeSamples(data)
    f.close()


# Funciones para generación de ondas
def generate_spike(amplitude, duration, sfreq):
    n_samples = int(duration * sfreq)
    spike = gaussian(n_samples, std=n_samples/7)
    return amplitude * spike / np.max(spike)


def generate_spikes_channel(n_spikes, times, sfreq, baseline, amplitude, duration, mode='transient', white_noise_amplitude=0, pink_noise_amplitude=0):
    channel_data = np.copy(baseline)
    prev_spike_end = 0
    for _ in range(n_spikes):
        amplitude_val = random.uniform(*amplitude)
        duration_val = random.uniform(*duration)
        spike = generate_spike(amplitude_val, duration_val, sfreq)
        
        if mode == 'transient':
            start_index = random.randint(0, len(times) - len(spike) - 1)
        elif mode == 'complex':
            start_index = prev_spike_end
            if start_index + len(spike) >= len(times):
                break
        else:
            raise ValueError("Invalid mode. Use 'transient' or 'complex'.")
        
        channel_data[start_index:start_index + len(spike)] += spike
        prev_spike_end = start_index + len(spike)
        # Agregar ruido blanco y rosa
    white_noise = white_noise_amplitude * np.random.randn(len(channel_data))
    pink_noise = pink_noise_amplitude * np.cumsum(np.random.randn(len(channel_data)))
    channel_data += white_noise + pink_noise

    return channel_data

# Define the EEG frequency bands
delta_band = [0, 4]  # Delta rhythm: 0-4 Hz
theta_band = [4, 8]  # Theta rhythm: 4-8 Hz
alpha_band = [8, 12]  # Alpha rhythm: 8-12 Hz
beta_band = [12, 30]  # Beta rhythm: 12-30 Hz
gamma_band = [30, 70]  # Gamma rhythm: 30-70 Hz

# Define the duration and sampling frequency of the EEG signal
duration = 10  # seconds
sampling_freq = 500  # Hz
num_samples = duration * sampling_freq
time = np.arange(0, duration, 1 / sampling_freq)

# Create empty EEG signal
eeg_signal = np.zeros(num_samples)

# Generate each frequency band
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

# Generate pink noise
pink_noise = np.random.randn(num_samples)
pink_noise = np.cumsum(pink_noise)
pink_noise -= np.mean(pink_noise)
pink_noise /= np.std(pink_noise)
eeg_signal += pink_noise

# Generate white noise
white_noise = np.random.randn(num_samples)
white_noise /= np.std(white_noise)
eeg_signal += white_noise

# Generate brown noise
brown_noise = np.random.randn(num_samples)
brown_noise = np.cumsum(brown_noise)
brown_noise -= np.mean(brown_noise)
brown_noise /= np.std(brown_noise)
eeg_signal += brown_noise

# Normalize the signal to the desired amplitude range
eeg_signal /= np.max(np.abs(eeg_signal))
eeg_signal *= 100  # Adjust the amplitude scale to your desired range


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

    # Create empty EEG signal
    eeg_signal = np.zeros(num_samples)

    for band, amplitude in zip(freq_bands, amplitudes):
        eeg_signal += generate_band(band, amplitude, duration, sampling_freq)

    # Generate pink noise
    pink_noise = np.random.randn(num_samples) * noise_amplitude
    pink_noise = np.cumsum(pink_noise)
    pink_noise -= np.mean(pink_noise)
    pink_noise /= np.std(pink_noise)
    eeg_signal += pink_noise

    # Normalize the signal to the desired amplitude range
    eeg_signal /= np.max(np.abs(eeg_signal))
    eeg_signal *= 100  # Adjust the amplitude scale to your desired range

    return eeg_signal

