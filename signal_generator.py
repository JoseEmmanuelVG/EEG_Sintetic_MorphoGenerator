import numpy as np
import random
from scipy.signal import gaussian
import datetime


import matplotlib.pyplot as plt
from scipy.stats import norm

# edf
import pyedflib



# Functions for wave generation
# Assuming gaussian function is defined or imported from scipy.stats
from scipy.stats import norm


def generate_spike(amplitude, duration, sfreq):
    if isinstance(amplitude, (list, tuple)):
        amplitude = random.uniform(*amplitude)
    n_samples = int(duration * sfreq)
    std_dev = n_samples / 5  # Ajustado para un pico más agudo
    spike = gaussian(n_samples, std=std_dev)
    return amplitude * spike / np.max(spike)


def generate_slow_wave(amplitude_wave, duration, sfreq, frequency=1.0):
    """
    Generate a sinusoidal slow wave with a specific frequency.
    
    Args:
    amplitude_wave (float): The amplitude of the slow wave.
    duration (float): The duration in seconds over which to generate the wave.
    sfreq (int): The sampling frequency in Hz.
    frequency (float, optional): The frequency of the slow wave in Hz. Defaults to 1.0 Hz.
    
    Returns:
    numpy.ndarray: The generated slow wave signal.
    """
    n_samples = int(duration * sfreq)
    t = np.arange(n_samples) / sfreq
    slow_wave = np.sin(2 * np.pi * frequency * t)
    return amplitude_wave * slow_wave


def generate_spike_slow_wave(sfreq, amplitude_spike_range, duration_spike_range, amplitude_slow_range, duration_slow_range):
    """
    Generate a combined spike-slow wave sequence with specified amplitude and duration ranges.
    
    Args:
    sfreq (int): Sampling frequency.
    amplitude_spike_range (tuple): Amplitude range for spikes.
    duration_spike_range (tuple): Duration range for spikes in seconds.
    amplitude_slow_range (tuple): Amplitude range for slow waves.
    duration_slow_range (tuple): Duration range for slow waves in seconds.

    Returns:
    numpy.ndarray: The generated spike-slow wave group data.
    """
    # Generate the spike wave
    amplitude_spike = random.uniform(*amplitude_spike_range)
    duration_spike = random.uniform(*duration_spike_range)
    spike_wave = generate_spike(amplitude_spike, duration_spike, sfreq)
    
    # Generate the slow wave
    amplitude_slow = random.uniform(*amplitude_slow_range)
    duration_slow = random.uniform(*duration_slow_range)
    slow_wave = generate_slow_wave(amplitude_slow, duration_slow, sfreq)
    
    # Combine the spike and slow wave
    combined_wave = np.concatenate((spike_wave, slow_wave))
    return combined_wave





def generate_channel(n_waves, times, sfreq, baseline, amplitude_spike, duration_spike_range, amplitude_slow, duration_slow_range, wave_type='spike', mode='transient', white_noise_amplitude=0, pink_noise_amplitude=0, frequency=1.0):
    channel_data = np.copy(baseline)
    eeg_length = len(times)

    # Define a function to generate the appropriate wave type
    def generate_wave(wave_type):
        if wave_type == 'spike':
            amplitude_val = random.uniform(*amplitude_spike)
            duration_val = random.uniform(*duration_spike_range)
            return generate_spike(amplitude_val, duration_val, sfreq)
        elif wave_type == 'slow_wave':
            amplitude_val = random.uniform(*amplitude_slow)
            duration_val = random.uniform(*duration_slow_range)
            return generate_slow_wave(amplitude_val, duration_val, sfreq, frequency)
        elif wave_type == 'spike_slow_wave':
            return generate_spike_slow_wave(sfreq, amplitude_spike, duration_spike_range, amplitude_slow, duration_slow_range)

    # Generate waves and insert them into the channel data
    last_end_index = 0
    for _ in range(n_waves):
        wave = generate_wave(wave_type)
        wave_length = len(wave)

        # Ensure there's enough space for the wave
        if last_end_index + wave_length > eeg_length:
            break  # Not enough space

        # Find a start index for the wave within a valid range
        start_index = random.randint(last_end_index, eeg_length - wave_length) if mode == 'transient' else (eeg_length - wave_length) // 2
        channel_data[start_index:start_index + wave_length] += wave
        last_end_index = start_index + wave_length if mode == 'transient' else eeg_length

    # Add white and pink noise to channel_data if necessary
    if white_noise_amplitude:
        white_noise = white_noise_amplitude * np.random.randn(eeg_length)
        channel_data += white_noise
    if pink_noise_amplitude:
        pink_noise = pink_noise_amplitude * np.cumsum(np.random.randn(eeg_length))
        channel_data += pink_noise

    return channel_data





def generate_spikes_channel(n_spikes, times, sfreq, baseline, amplitude, duration, mode='transient', white_noise_amplitude=0, pink_noise_amplitude=0):
    channel_data = np.copy(baseline)
    eeg_length = len(times)

    # Manejar duration como un número
    if isinstance(duration, (list, tuple)):
        duration_val = sum(duration) / len(duration)  # Promedio si es lista
    else:
        duration_val = duration
    spike_length = int(sfreq * duration_val)

    if mode == 'transient':
        for _ in range(n_spikes):
            # Generar e insertar una onda
            spike = generate_spike(amplitude, duration_val, sfreq)
            start_index = random.randint(0, eeg_length - spike_length)
            channel_data[start_index:start_index + spike_length] += spike

            # Insertar una parte de la señal EEG después de la onda
            eeg_part_length = int(sfreq * random.uniform(0.1, 0.5))  # Duración aleatoria para la parte EEG
            start_index += spike_length
            if start_index + eeg_part_length < eeg_length:
                channel_data[start_index:start_index + eeg_part_length] = baseline[start_index:start_index + eeg_part_length]

    elif mode == 'complex':
        # Calculamos la longitud total de las ondas
        total_spikes_length = n_spikes * spike_length

        # Verificamos si hay suficiente espacio para las ondas y la señal EEG
        if total_spikes_length < eeg_length:
            # Insertamos las ondas juntas en el medio
            start_index = (eeg_length - total_spikes_length) // 2
            for _ in range(n_spikes):
                spike = generate_spike(amplitude, duration_val, sfreq)
                channel_data[start_index:start_index + spike_length] += spike
                start_index += spike_length

    # Agregar ruido blanco y rosa a channel_data
    white_noise = white_noise_amplitude * np.random.randn(len(channel_data))
    pink_noise = pink_noise_amplitude * np.cumsum(np.random.randn(len(channel_data)))
    channel_data += white_noise + pink_noise

    return channel_data






def get_wave_values(amplitude, duration):
    if isinstance(amplitude, (list, tuple)):
        amplitude_val = random.uniform(*amplitude)
    else:
        amplitude_val = amplitude

    if isinstance(duration, (list, tuple)):
        duration_val = random.uniform(*duration)
    else:
        duration_val = duration

    return amplitude_val, duration_val


def generate_spike_wave_group(sfreq, amplitude_spike, duration_spike, amplitude_slow, duration_slow, eeg_signal, group_duration, mode='transient'):
    n_samples = int(group_duration * sfreq)
    group_data = np.zeros(n_samples)
    current_start_index = 0

    # Asegurarse de que duration_spike sea un iterable antes de usar max().
    if not isinstance(duration_spike, (list, tuple)):
        duration_spike = (duration_spike, duration_spike)
    
    # Utilizar siempre la duración máxima para la onda punta para asegurar su completa generación antes de la onda lenta.
    max_duration_spike = max(duration_spike)

    while current_start_index < n_samples:
        # Generar la onda tipo "spike"
        amplitude_spike_val, _ = get_wave_values(amplitude_spike, duration_spike)
        spike = generate_spike(amplitude_spike_val, max_duration_spike, sfreq)
        n_duration_spike = len(spike)

        # Generar la onda lenta
        amplitude_slow_val, duration_slow_val = get_wave_values(amplitude_slow, duration_slow)
        slow_wave = generate_slow_wave(amplitude_slow_val, duration_slow_val, sfreq)
        n_duration_slow = len(slow_wave)

        # Asegurarse que hay espacio suficiente para la inserción de las ondas
        if current_start_index + n_duration_spike + n_duration_slow <= n_samples:
            # Modo transitorio: las ondas se insertan una tras otra con espacio EEG aleatorio entre ellas
            if mode == 'transient':
                group_data[current_start_index:current_start_index + n_duration_spike] = spike
                current_start_index += n_duration_spike
                group_data[current_start_index:current_start_index + n_duration_slow] = slow_wave
                current_start_index += n_duration_slow

                # Espacio EEG entre ondas punta-lenta
                eeg_part_length = int(sfreq * random.uniform(1, 2))
                if current_start_index + eeg_part_length < n_samples:
                    group_data[current_start_index:current_start_index + eeg_part_length] = eeg_signal[current_start_index:current_start_index + eeg_part_length]
                    current_start_index += eeg_part_length
            # Modo complejo: las ondas se insertan juntas en el centro del grupo
            elif mode == 'complex':
                center_index = n_samples // 2
                spike_start_index = max(center_index - n_duration_spike, 0)
                slow_start_index = spike_start_index + n_duration_spike
                group_data[spike_start_index:spike_start_index + n_duration_spike] = spike
                group_data[slow_start_index:slow_start_index + n_duration_slow] = slow_wave
                break  # Only one centered group for complex mode
        else:
            # No hay suficiente espacio para insertar la siguiente onda lenta
            break

    return group_data




















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


eeg_signal += generate_band(gamma_band, amplitude=50, duration=duration, sampling_freq=sampling_freq)
eeg_signal += generate_band(beta_band, amplitude=30, duration=duration, sampling_freq=sampling_freq)
eeg_signal += generate_band(alpha_band, amplitude=20, duration=duration, sampling_freq=sampling_freq)
eeg_signal += generate_band(theta_band, amplitude=10, duration=duration, sampling_freq=sampling_freq)
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

def save_to_txt(data, sfreq, file_name="output_file.txt"):
    with open(file_name, "w") as f:
        # Encabezado del archivo
        f.write("Numero de datos\tFrecuencia de muestreo\tNumero de columnas\tInicial\tFinal\n")

        # Información del archivo
        num_datos = len(data[0])  # Asumiendo que todas las señales tienen la misma longitud
        num_columnas = len(data)
        f.write(f"{num_datos}\t{sfreq}\t{num_columnas}\t0\t{num_columnas - 1}\n")

        # Escribir los datos de cada canal en líneas separadas
        for channel_data in data:
            line = "\t".join(map(str, channel_data))
            f.write(line + "\n")






# Función para plotear la señal generada
def plot_spike_slow_wave(sfreq, amplitude_spike, duration_spike, amplitude_slow, duration_slow, group_duration):
    # Generar la señal EEG con ruido para el fondo
    eeg_signal = generate_eeg_signal([delta_band, theta_band, alpha_band, beta_band, gamma_band], [50, 30, 20, 10, 5], group_duration, sfreq)
    
    # Generar el grupo de ondas punta-lenta
    spike_slow_group = generate_spike_wave_group(sfreq, amplitude_spike, duration_spike, amplitude_slow, duration_slow, eeg_signal, group_duration)
    
    # Crear un vector de tiempo para plotear
    time_vector = np.linspace(0, group_duration, int(sfreq * group_duration))
    
    # Plotear la señal
    plt.figure(figsize=(10, 4))
    plt.plot(time_vector, spike_slow_group, label='Spike-Slow Wave Group')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Spike-Slow Wave Group Signal')
    plt.legend()
    plt.show()

# Parámetros de ejemplo
sfreq = 500  # Frecuencia de muestreo
amplitude_spike = (40, 60)  # Amplitud de las ondas puntas
duration_spike = (0.02, 0.07)  # Duración de las ondas puntas
amplitude_slow = (30, 60)  # Amplitud de las ondas lentas
duration_slow = (0.2, 0.5)  # Duración de las ondas lentas
group_duration = 3  # Duración total del grupo en segundos

# Llamar a la función para plotear la señal
plot_spike_slow_wave(sfreq, amplitude_spike, duration_spike, amplitude_slow, duration_slow, group_duration)