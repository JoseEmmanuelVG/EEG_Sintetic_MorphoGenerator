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

from signal_conectors_eeg import generate_eeg_signal


# Funciones para generación de ondas
def generate_spike(amplitude, duration, sfreq):
    n_samples = int(duration * sfreq)
    spike = gaussian(n_samples, std=n_samples/7)
    return amplitude * spike / np.max(spike)

def generate_spikes_channel(n_spikes, times, sfreq, baseline, amplitude, duration, mode='transient'):
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
    
    return channel_data

app = dash.Dash(__name__)

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

    html.Label('Amplitud de EEG (µV)'),  # Nueva etiqueta para el slider de amplitud de EEG
    dcc.Slider(id='eeg-amplitude-slider', min=-0, max=1, step=0.1, value=1),  # Slider de amplitud de EEG
])

@app.callback(
    Output("graph", "figure"), 
    [Input("amplitude-slider", "value"),
     Input("duration-slider", "value"),
     Input("sfreq-slider", "value"),
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
     Input("eeg-amplitude-slider", "value")]  # Agregamos el Input del slider de amplitud de EEG
)
def update_graph(amplitude, duration, sfreq, n_channels, n_spikes, total_time, spike_mode,
                delta_band, theta_band, alpha_band, beta_band, gamma_band, noise_amplitude,
                eeg_amplitude):  # Agregamos el argumento para la amplitud de EEG
    times = np.arange(0, total_time, 1/sfreq)
    baseline = np.zeros(len(times))

    data = []

    # Generar la señal EEG con la amplitud actualizada desde el slider
    eeg_signal = generate_eeg_signal(
        freq_bands=[delta_band, theta_band, alpha_band, beta_band, gamma_band],
        amplitudes=[50, 30, 20, 10, 5],
        duration=total_time,
        sampling_freq=sfreq,
        noise_amplitude=noise_amplitude
    ) * eeg_amplitude  # Multiplicar por el valor del slider para controlar la amplitud

    for ch in range(n_channels):
        channel_data = generate_spikes_channel(
            n_spikes=n_spikes,
            times=times,
            sfreq=sfreq,
            baseline=baseline,
            amplitude=amplitude,
            duration=duration,
            mode=spike_mode
        )

        # Unir la señal EEG con las puntas
        if spike_mode == 'transient':
            start_idx = int((ch + 1) * len(times) / (n_channels + 1))
            end_idx = int((ch + 2) * len(times) / (n_channels + 1))
            combined_data = np.concatenate([eeg_signal[start_idx:end_idx], channel_data])
        else:
            if ch == 0:
                start_idx = 0
                end_idx = int(len(times) / (n_channels + 1))
                combined_data = np.concatenate([eeg_signal[start_idx:end_idx], channel_data])
            elif ch == n_channels - 1:
                start_idx = int((ch + 1) * len(times) / (n_channels + 1))
                end_idx = len(times)
                combined_data = np.concatenate([channel_data, eeg_signal[start_idx:end_idx]])
            else:
                start_idx = int((ch + 1) * len(times) / (n_channels + 1))
                end_idx = int((ch + 2) * len(times) / (n_channels + 1))
                combined_data = np.concatenate([channel_data, eeg_signal[start_idx:end_idx]])

        trace = go.Scattergl(x=times, y=combined_data, mode='lines', name=f'Canal {ch + 1}')
        data.append(trace)

    layout = go.Layout(yaxis=dict(range=[0, max(amplitude)]), xaxis=dict(range=[0, total_time]))

    return go.Figure(data=data, layout=layout)
from waitress import serve
serve(app.server, host="0.0.0.0", port=6060)
