import dash
import matplotlib
matplotlib.use('Agg')

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from signal_generator import generate_channel, generate_eeg_signal, save_to_edf, save_to_txt
import numpy as np 
import mne
from plotly.subplots import make_subplots
import time

from dash import dcc
from dash import html
import os
from signal_generator import delta_band
import plotly.io as pio
import random

import glob
import zipfile



def register_callbacks_detailed(app):
    

# Callback para actualizar la UI basada en el tipo de onda seleccionado
    @app.callback(
        [Output("amplitude-slider-spike", "style"),
        Output("amplitude-slider-slow", "style"),
        Output("duration-slider-spike", "style"),
        Output("duration-slider-slow", "style"),
        Output("amplitude-slider-spike-slow", "style"),
        Output("duration-slider-spike-slow", "style")],
        [Input("wave-selector", "value")]
    )
    def update_ui_based_on_wave(wave_selector):
        if wave_selector == "spike":
            return {}, {'display': 'none'}, {}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        elif wave_selector == "slow":
            return {'display': 'none'}, {}, {'display': 'none'}, {}, {'display': 'none'}, {'display': 'none'}
        elif wave_selector == "spike-slow":
            return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {}, {}
        else:
            return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}





    @app.callback(
        Output("graph", "figure"),
        [Input("amplitude-slider-spike", "value"),
        Input("amplitude-slider-slow", "value"),
        Input("duration-slider-spike", "value"),
        Input("duration-slider-slow", "value"),
        Input("amplitude-slider-spike-slow", "value"),
        Input("duration-slider-spike-slow", "value"),
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
        Input("eeg-amplitude-slider", "value"),
        Input("wave-selector", "value")]
    )
    def update_graph(amplitude_spike, amplitude_slow, duration_spike, duration_slow, amplitude_spike_slow, duration_spike_slow,
                    sfreq, spike_noise_amplitude, n_channels, n_waves, total_time, 
                    spike_mode, delta_band, theta_band, alpha_band, beta_band, 
                    gamma_band, noise_amplitude, eeg_amplitude, wave_selector):

        num_samples = int(total_time * sfreq)
        times = np.linspace(0, total_time, num_samples, endpoint=False)
        data = []

        eeg_signal = generate_eeg_signal(
            freq_bands=[delta_band, theta_band, alpha_band, beta_band, gamma_band],
            amplitudes=[50, 30, 20, 10, 5],
            duration=total_time,
            sampling_freq=sfreq,
            noise_amplitude=noise_amplitude
        ) * (eeg_amplitude / 100)

        for ch in range(n_channels):
            if wave_selector == "spike-slow":
                channel_data = generate_channel(
                    n_waves=n_waves,
                    times=times,
                    sfreq=sfreq,
                    baseline=np.zeros(len(times)),
                    amplitude_spike=amplitude_spike,
                    duration_spike_range=(duration_spike[0], duration_spike[1]),
                    amplitude_slow=amplitude_slow,
                    duration_slow_range=(duration_slow[0], duration_slow[1]),
                    wave_type='spike_slow_wave',
                    mode=spike_mode,
                    white_noise_amplitude=spike_noise_amplitude,
                    pink_noise_amplitude=spike_noise_amplitude
                )
            elif wave_selector == "slow":
                # Para ondas lentas, utiliza sólo los parámetros relevantes
                channel_data = generate_channel(
                    n_waves=n_waves,
                    times=times,
                    sfreq=sfreq,
                    baseline=np.zeros(len(times)),
                    amplitude_spike=None,  # o un valor predeterminado si es necesario
                    duration_spike_range=None,  # o un valor predeterminado si es necesario
                    amplitude_slow=amplitude_slow,
                    duration_slow_range=(duration_slow[0], duration_slow[1]),
                    wave_type='slow_wave',
                    mode=spike_mode,
                    white_noise_amplitude=spike_noise_amplitude,
                    pink_noise_amplitude=spike_noise_amplitude
                )
            else:  # 'spike' o cualquier otro valor
                # Para ondas de pico, utiliza sólo los parámetros relevantes
                channel_data = generate_channel(
                    n_waves=n_waves,
                    times=times,
                    sfreq=sfreq,
                    baseline=np.zeros(len(times)),
                    amplitude_spike=amplitude_spike,
                    duration_spike_range=(duration_spike[0], duration_spike[1]),
                    amplitude_slow=None,  # o un valor predeterminado si es necesario
                    duration_slow_range=None,  # o un valor predeterminado si es necesario
                    wave_type='spike',
                    mode=spike_mode,
                    white_noise_amplitude=spike_noise_amplitude,
                    pink_noise_amplitude=spike_noise_amplitude
                )

            # Mezcla la señal generada con la señal EEG
            combined_data = np.where(channel_data != 0, channel_data, eeg_signal)
            
            trace = go.Scattergl(x=times, y=combined_data, mode='lines', name=f'Canal {ch + 1}')
            data.append(trace)

        return {"data": data, "layout": go.Layout(title="EEG Signal")}






    @app.callback(
        Output('save-message', 'children'),
        [Input('save-button', 'n_clicks')],
        [State("amplitude-slider-spike", "value"),
        State("amplitude-slider-slow", "value"),
        State("duration-slider-spike", "value"),
        State("duration-slider-slow", "value"),
        State("amplitude-slider-spike-slow", "value"),
        State("duration-slider-spike-slow", "value"),
        State("wave-selector", "value"),
        State("sfreq-slider", "value"),
        State("spike-noise-amplitude", "value"),
        State("channels-input", "value"),
        State("spikes-input", "value"),
        State("time-input", "value"),
        State("spike-mode", "value"),
        State("delta-band", "value"),
        State("theta-band", "value"),
        State("alpha-band", "value"),
        State("beta-band", "value"),
        State("gamma-band", "value"),
        State("noise-amplitude", "value"),
        State("eeg-amplitude-slider", "value")]
    )
    def save_data(n_clicks, amplitude_spike, amplitude_slow, duration_spike, duration_slow, amplitude_spike_slow, duration_spike_slow,
                wave_selector, sfreq, spike_noise_amplitude, n_channels, n_spikes, total_time, spike_mode,
                delta_band, theta_band, alpha_band, beta_band, gamma_band, noise_amplitude, eeg_amplitude):

        if not n_clicks:
            raise PreventUpdate

        num_samples = int(total_time * sfreq)
        times = np.linspace(0, total_time, num_samples, endpoint=False)

        eeg_signal = generate_eeg_signal(
            freq_bands=[delta_band, theta_band, alpha_band, beta_band, gamma_band],
            amplitudes=[50, 30, 20, 10, 5],
            duration=total_time,
            sampling_freq=sfreq,
            noise_amplitude=noise_amplitude
        ) * (eeg_amplitude / 100)

        data_for_edf = []

        for ch in range(n_channels):
            if wave_selector == "spike":
                channel_data = generate_channel(n_spikes, times, sfreq, np.zeros(len(times)), amplitude_spike, duration_spike, None, None, 'spike', spike_mode, spike_noise_amplitude, spike_noise_amplitude)
            elif wave_selector == "slow":
                channel_data = generate_channel(n_spikes, times, sfreq, np.zeros(len(times)), None, None, amplitude_slow, duration_slow, 'slow_wave', spike_mode, spike_noise_amplitude, spike_noise_amplitude)
            elif wave_selector == "spike-slow":
                channel_data = generate_channel(n_spikes, times, sfreq, np.zeros(len(times)), amplitude_spike_slow, duration_spike_slow, amplitude_spike_slow, duration_spike_slow, 'spike_slow_wave', spike_mode, spike_noise_amplitude, spike_noise_amplitude)
            else:
                # Default case, you might want to handle this differently
                channel_data = np.zeros(len(times))

            combined_data = np.where(channel_data != 0, channel_data, eeg_signal)
            data_for_edf.append(combined_data)

        channel_names = [f"Channel_{i}" for i in range(1, n_channels + 1)]
        save_to_edf(data_for_edf, sfreq, channel_names, "assets/output_file.edf")
        save_to_txt(data_for_edf, channel_names, "assets/output_file.txt")

        return "Data saved successfully!"




    @app.callback(
        [Output('eeg-plot', 'figure'), Output('eeg-image', 'src'), Output('download-image', 'href'), Output('download-edf', 'href')],
        [Input('show-button', 'n_clicks')]
        )
    def update_graph(n_clicks):
        if n_clicks == 0:
            return go.Figure(), None, '', ''

        raw = mne.io.read_raw_edf("assets/output_file.edf", preload=True)

        fig_mne = raw.plot(n_channels=15, scalings={"eeg": 100e-6}, show=False)
        fig_mne.savefig('assets/EEG_plot.png')

        timestamp = time.time()
        image_path = f'/assets/EEG_plot.png?{timestamp}'
        edf_path = f'/assets/output_file.edf?{timestamp}'

        data, times = raw[:, :]
        fig = make_subplots(rows=len(data), cols=1, shared_xaxes=True, vertical_spacing=0.02)

        for index, channel_data in enumerate(data):
            # Verifica si channel_data es un array 2D y selecciona la primera fila si es necesario
            if isinstance(channel_data, np.ndarray) and channel_data.ndim > 1:
                channel_data = channel_data[0]  # Asume que los datos están en la primera fila
                trace = go.Scattergl(x=times, y=channel_data, mode='lines', name=f'Channel {index + 1}')
                fig.add_trace(trace, row=index + 1, col=1)

            elif isinstance(channel_data, np.ndarray) and channel_data.ndim > 1:
                y_data = channel_data[0]  # Asume que los datos están en la primera fila
            else:
                y_data = np.zeros_like(times)  # En caso de que channel_data no sea un array

            trace = go.Scattergl(x=times, y=y_data, mode='lines', name=f'Channel {index + 1}')
            fig.add_trace(trace, row=index + 1, col=1)

        fig.update_layout(height=300 * len(data), title_text="EEG Signal", showlegend=False)

        return fig, image_path, image_path, edf_path
