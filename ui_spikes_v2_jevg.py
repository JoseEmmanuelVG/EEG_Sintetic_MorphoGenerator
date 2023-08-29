import io
import base64
import numpy as np
import random
from scipy.signal import gaussian
import pandas as pd
import matplotlib.pyplot as plt
import dash
import plotly.graph_objs as go
from dash import dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

app = dash.Dash(__name__)

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



# Definición de app.layout
app.layout = html.Div([
    html.H1("Simulador de ondas epilépticas"),
    dcc.Graph(id="graph"),

    html.Label('Amplitud (µV)'),
    dcc.RangeSlider(id='amplitude-slider', min=-10, max=100, step=2, value=[5, 50]),  

    html.Label('Duración'),
    dcc.RangeSlider(id='duration-slider', min=0.1, max=2, step=0.1, value=[1, 1.5]),

    html.Label('Frecuencia de muestreo'),
    dcc.Slider(id='sfreq-slider', min=100, max=1000, step=50, value=500),

    html.Label('Número de Canales'),
    dcc.Input(id='channels-input', type='number', min=1, value=1),

    html.Label('Número de Puntas'),
    dcc.Input(id='spikes-input', type='number', min=1, value=10),

    html.Label('Tiempo total (s)'),
    dcc.Input(id='time-input', type='number', min=1, value=10),

    html.Label('Modo de generación de ondas pico'),
dcc.RadioItems(
    id='spike-mode',
    options=[
        {'label': 'Transitorio', 'value': 'transient'},
        {'label': 'Complejo', 'value': 'complex'}
    ],
    value='transient'),

#Ruido para las ondas
html.Label('Amplitud de Ruido en Ondas Pico (Blanco y Rosa)'),
dcc.Slider(id='spike-noise-amplitude', min=0, max=2, step=0.1, value=0),



    html.Label('Banda de Delta'),
    dcc.RangeSlider(id='delta-band', min=0, max=4, step=0.1, value=[0, 4]),

    html.Label('Banda de Theta'),
    dcc.RangeSlider(id='theta-band', min=4, max=8, step=0.1, value=[4, 8]),

    html.Label('Banda de Alpha'),
    dcc.RangeSlider(id='alpha-band', min=8, max=12, step=0.1, value=[8, 12]),

    html.Label('Banda de Beta'),
    dcc.RangeSlider(id='beta-band', min=12, max=30, step=0.5, value=[12, 30]),

    html.Label('Banda de Gamma'),
    dcc.RangeSlider(id='gamma-band', min=30, max=70, step=0.1, value=[30, 70]),

    html.Label('Amplitud del ruido'),
    dcc.Slider(id='noise-amplitude', min=0, max=2, step=0.1, value=1),

    html.Label('Amplitud de EEG (µV)'),
    dcc.Slider(id='eeg-amplitude-slider', min=0, max=100, step=1, value=50),  # Cambiamos el rango para permitir valores entre 0 y 100
])

@app.callback(
    Output("graph", "figure"), 
    [Input("amplitude-slider", "value"),
     Input("duration-slider", "value"),
     Input("sfreq-slider", "value"),
     Input("spike-noise-amplitude", "value"),
     Input("channels-input", "value"),
     Input("spikes-input", "value"),
     Input("time-input", "value"),
     Input("spike-mode", "value"),
     Input("delta-band", "value"),
     Input("theta-band", "value"),
     Input("alpha-band", "value"),
     Input("beta-band", "value"),
     Input("gamma-band", "value"),
     Input("noise-amplitude", "value"),
     Input("eeg-amplitude-slider", "value")]
)
def update_graph(amplitude, duration, sfreq, spike_noise_amplitude, n_channels, n_spikes, total_time, spike_mode,
                 delta_band, theta_band, alpha_band, beta_band, gamma_band, noise_amplitude, eeg_amplitude):
    num_samples = int(total_time * sfreq)
    times = np.linspace(0, total_time, num_samples, endpoint=False)

    baseline = np.zeros(len(times))

    data = []

    # Generar la señal EEG con la amplitud actualizada desde el slider
    eeg_signal = generate_eeg_signal(
        freq_bands=[delta_band, theta_band, alpha_band, beta_band, gamma_band],
        amplitudes=[50, 30, 20, 10, 5],
        duration=total_time,
        sampling_freq=sfreq,
        noise_amplitude=noise_amplitude
    ) * (eeg_amplitude / 100)

    for ch in range(n_channels):
        channel_data = generate_spikes_channel(
        n_spikes=n_spikes,
        times=times,
        sfreq=sfreq,
        baseline=baseline,
        amplitude=amplitude,
        duration=duration,
        mode=spike_mode,
        white_noise_amplitude=spike_noise_amplitude, # Agregar esta línea
        pink_noise_amplitude=spike_noise_amplitude   # Agregar esta línea
    )

        # Si estamos en modo complejo, queremos que las puntas estén juntas
        if spike_mode == 'complex':
            # Encuentra los índices donde comienzan y terminan las puntas
            spike_start_idx = np.where(channel_data != 0)[0][0]
            spike_end_idx = np.where(channel_data != 0)[0][-1]

            # Extrae las partes de la señal EEG antes, en medio y después de las puntas
            eeg_start = eeg_signal[:spike_start_idx]
            eeg_middle = eeg_signal[spike_start_idx:spike_end_idx + 1]
            eeg_end = eeg_signal[spike_end_idx + 1:]

            # Combinar la señal EEG y las puntas
            combined_data = np.concatenate([eeg_start, channel_data[spike_start_idx:spike_end_idx + 1], eeg_middle, channel_data[spike_start_idx:spike_end_idx + 1], eeg_end])
        else:
            # En el modo transitorio, simplemente combine donde hay puntas
            combined_data = np.where(channel_data != 0, channel_data, eeg_signal)


        trace = go.Scattergl(x=times[:len(combined_data)], y=combined_data, mode='lines', name=f'Canal {ch + 1}')
        data.append(trace)

    layout = go.Layout(yaxis=dict(range=[-10, max(amplitude)]), xaxis=dict(range=[0, total_time]))

    return go.Figure(data=data, layout=layout)


from waitress import serve
serve(app.server, host="0.0.0.0", port=8080)
