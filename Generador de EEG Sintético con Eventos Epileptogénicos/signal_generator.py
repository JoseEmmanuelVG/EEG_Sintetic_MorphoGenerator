import numpy as np
import random
from scipy.signal import gaussian
import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm
# edf
import pyedflib

# Funciones para la generación de ondas
def generate_spike(amplitude, duration, sfreq):
    if isinstance(amplitude, (list, tuple)):
        amplitude = random.uniform(*amplitude)
    n_samples = int(duration * sfreq)
    std_dev = n_samples / 5  # Ajustar para un pico más agudo
    spike = gaussian(n_samples, std=std_dev)
    return amplitude * spike / np.max(spike)


def generate_slow_wave(amplitude_wave, duration, sfreq, frequency=1.0, cutoff_range=(-5, 2)):
    """
    Genera una onda lenta sinusoidal con una frecuencia específica que completa al menos medio ciclo y termina entre un rango de corte.
    
    Args:
        amplitud_onda (float): La amplitud de la onda lenta.
        duración (float): La duración en segundos sobre la que generar la onda.
        sfreq (int): La frecuencia de muestreo en Hz.
        frequency (float, opcional): La frecuencia de la onda lenta en Hz. Por defecto 1.0 Hz.
        cutoff_range (tupla): El rango de amplitud dentro del cual puede terminar la onda lenta.
    
    Devuelve:
        numpy.ndarray: La señal de onda lenta generada.
    """
    # Calcular el periodo de un ciclo
    period = 1 / frequency
    # Calcular cuántas muestras por ciclo
    samples_per_cycle = int(period * sfreq)
    # Garantizar que se genera al menos medio ciclo
    samples_needed = max(samples_per_cycle // 2, int(duration * sfreq))
    t = np.linspace(0, samples_needed / sfreq, samples_needed, endpoint=False)
    # Generar la onda completa
    full_wave = amplitude_wave * np.sin(2 * np.pi * frequency * t)
    # Encuentra donde la onda cruza por primera vez el corte inferior después del pico.
    peak_index = np.argmax(full_wave)
    cross_lower = np.where((full_wave[peak_index:] <= cutoff_range[1]) & 
                           (full_wave[peak_index:] >= cutoff_range[0]))[0]
    # Si no se encuentra ningún cruce, termina la onda en el último punto
    if len(cross_lower) == 0:
        cutoff_index = len(full_wave) - 1
    else:
        cutoff_index = peak_index + cross_lower[0]
    # Devuelve la onda hasta el corte
    return full_wave[:cutoff_index+1]


def generate_spike_slow_wave(sfreq, amplitude_spike_range, duration_spike_range, amplitude_slow_range, duration_slow_range):
    """
    Genera una secuencia combinada de picos y ondas lentas con los rangos de amplitud y duración especificados.
    
    Args:
    sfreq (int): Frecuencia de muestreo.
    amplitude_spike_range (tupla): Rango de amplitud de los picos.
    duration_spike_range (tupla): Intervalo de duración de los picos en segundos.
    amplitude_slow_range (tupla): Rango de amplitud de las ondas lentas.
    duration_slow_range (tupla): Intervalo de duración de las ondas lentas en segundos.

    Devuelve:
    numpy.ndarray: Los datos generados del grupo de ondas pico-lentas.
    """
    # Generar la onda de punta
    amplitude_spike = random.uniform(*amplitude_spike_range)
    duration_spike = random.uniform(*duration_spike_range)
    spike_wave = generate_spike(amplitude_spike, duration_spike, sfreq)
    # Generar la onda lenta
    amplitude_slow = random.uniform(*amplitude_slow_range)
    duration_slow = random.uniform(*duration_slow_range)
    slow_wave = generate_slow_wave(amplitude_slow, duration_slow, sfreq)
    # Combina el pico y la onda lenta
    combined_wave = np.concatenate((spike_wave, slow_wave))
    return combined_wave


# Función para generar un canal de EEG
def generate_channel(n_waves, times, sfreq, baseline, amplitude_spike, duration_spike_range, amplitude_slow, duration_slow_range, wave_type='spike', mode='transient', white_noise_amplitude=0, pink_noise_amplitude=0, frequency=1.0):
    channel_data = np.copy(baseline)
    eeg_length = len(times)

    # Define una función para generar el tipo de onda apropiado
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

    if mode == 'complex':
        # Define el intervalo centrado específico para la generación de ondas
        central_start_time = np.random.uniform(2, 4)  # Inicio del intervalo central
        central_end_time = np.random.uniform(6, 8)    # Fin del intervalo central
        
        # Convertir tiempos a índices
        central_start_index = int(central_start_time * sfreq)
        central_end_index = int(central_end_time * sfreq)
        
        # Ajusta el espacio disponible para la inserción de ondas
        available_space = central_end_index - central_start_index
    else:
        available_space = eeg_length

    last_end_index = central_start_index if mode == 'complex' else 0
    for _ in range(n_waves):
        wave = generate_wave(wave_type)
        wave_length = len(wave)

        # Para el modo complejo, ajusta el cálculo del índice de inicio
        if mode == 'complex':
            if last_end_index + wave_length > central_end_index:
                break  # No hay espacio suficiente en el intervalo central
            start_index = last_end_index
        else:
            if last_end_index + wave_length > eeg_length:
                break  # No hay espacio suficiente
            start_index = random.randint(last_end_index, eeg_length - wave_length)

        channel_data[start_index:start_index + wave_length] += wave
        last_end_index = start_index + wave_length
        
    # Añade ruido blanco y rosa a channel_data si es necesario
    if white_noise_amplitude:
        white_noise = white_noise_amplitude * np.random.randn(eeg_length)
        channel_data += white_noise
    if pink_noise_amplitude:
        pink_noise = pink_noise_amplitude * np.cumsum(np.random.randn(eeg_length))
        channel_data += pink_noise

    return channel_data

# Definir las bandas de frecuencia del EEG
delta_band = [0, 4]  # Delta rhythm: 0-4 Hz
theta_band = [4, 8]  # Theta rhythm: 4-8 Hz
alpha_band = [8, 12]  # Alpha rhythm: 8-12 Hz
beta_band = [12, 30]  # Beta rhythm: 12-30 Hz
gamma_band = [30, 70]  # Gamma rhythm: 30-70 Hz

# Definir la duración y la frecuencia de muestreo de la señal EEG
duration = 10  # segundos
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


eeg_signal += generate_band(gamma_band, amplitude=50, duration=duration, sampling_freq=sampling_freq)
eeg_signal += generate_band(beta_band, amplitude=30, duration=duration, sampling_freq=sampling_freq)
eeg_signal += generate_band(alpha_band, amplitude=20, duration=duration, sampling_freq=sampling_freq)
eeg_signal += generate_band(theta_band, amplitude=10, duration=duration, sampling_freq=sampling_freq)
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
    eeg_signal *= 100  # Ajusta la escala de amplitud al rango deseado

    return eeg_signal


def save_to_edf(data, sfreq, channel_names, filename='output.edf'):
    """
    Guarda los datos dados en un archivo EDF.

    Args:
        data (lista de matrices): Datos del canal de EEG.
        sfreq (int): Frecuencia de muestreo.
        channel_names (lista de cadenas): Nombres de los canales de EEG.
        filename (str): Nombre del archivo EDF que se va a crear.
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


# Función para guardar los datos en un archivo de texto
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




