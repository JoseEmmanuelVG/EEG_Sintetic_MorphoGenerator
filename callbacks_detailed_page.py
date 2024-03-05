import dash
import matplotlib
matplotlib.use('Agg')

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from signal_generator import generate_spike, generate_spikes_channel, generate_eeg_signal, save_to_edf, save_to_txt, generate_spike_wave_group
import numpy as np 
import mne
from plotly.subplots import make_subplots
import time
from ui_definition import generacion_rapida_layout, generacion_detallada_layout, homepage_layout

from dash import dcc
from dash import html
import os
from signal_generator import delta_band
import plotly.io as pio
import random

import glob
import zipfile



def register_callbacks_detailed(app):
    

# Página de Generación detallada 
    @app.callback(
        [Output("amplitude-slider-spike", "style"),
        Output("amplitude-slider-slow", "style"),
        Output("duration-slider-spike", "style"),
        Output("duration-slider-slow", "style")],
        [Input("wave-selector", "value")]
    )
    def update_ui_based_on_wave(wave_selector):
        if wave_selector == "slow":
            return {'display': 'none'}, {}, {'display': 'none'}, {}
        elif wave_selector == "mix_wave":
            return {}, {}, {}, {}
        else:  # Assume any other value is a spike
            return {}, {'display': 'none'}, {}, {'display': 'none'}


    @app.callback(
        Output("graph", "figure"),
        [Input("amplitude-slider-spike", "value"),
         Input("amplitude-slider-slow", "value"),
         Input("duration-slider-spike", "value"),
         Input("duration-slider-slow", "value"),
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
    def update_graph(amplitude_spike, amplitude_slow, duration_spike, duration_slow,
                    sfreq, spike_noise_amplitude, n_channels, n_spikes, total_time, 
                    spike_mode, delta_band, theta_band, alpha_band, beta_band, 
                    gamma_band, noise_amplitude, eeg_amplitude, wave_selector):

        # Asigna amplitude y duration basándose en wave_selector
        if wave_selector == "slow":
            amplitude = amplitude_slow
            duration = duration_slow
        elif wave_selector == "mix_wave":
            combined_data = generate_spike_wave_group(sfreq, group_duration=3)            

        else:  # Asumimos que cualquier otro valor es un spike
            amplitude = amplitude_spike
            duration = duration_spike

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
            channel_data = generate_spikes_channel(
                n_spikes=n_spikes,
                times=times,
                sfreq=sfreq,
                baseline=np.zeros(len(times)),
                amplitude=amplitude,
                duration=duration,
                mode=spike_mode,
                white_noise_amplitude=spike_noise_amplitude, 
                pink_noise_amplitude=spike_noise_amplitude
            )

            if spike_mode == 'complex':
                spike_start_idx = np.where(channel_data != 0)[0][0]
                spike_end_idx = np.where(channel_data != 0)[0][-1]
                combined_data = np.concatenate([
                    eeg_signal[:spike_start_idx],
                    channel_data[spike_start_idx:spike_end_idx + 1],
                    eeg_signal[spike_end_idx + 1:]
            ])
            else:
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
    def save_data(n_clicks, amplitude_spike, amplitude_slow, duration_spike, duration_slow, wave_selector, 
                sfreq, spike_noise_amplitude, n_channels, n_spikes, total_time, spike_mode,
                delta_band, theta_band, alpha_band, beta_band, gamma_band, noise_amplitude, eeg_amplitude):

        if not n_clicks:
            raise PreventUpdate

        # Asigna amplitude y duration basándose en wave_selector
        if wave_selector == "slow":
            amplitude = amplitude_slow
            duration = duration_slow
        else:  # Asumimos que cualquier otro valor es un spike
            amplitude = amplitude_spike
            duration = duration_spike


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
            channel_data = generate_spikes_channel(
                n_spikes=n_spikes,
                times=times,
                sfreq=sfreq,
                baseline=np.zeros(len(times)),
                amplitude=amplitude,
                duration=duration,
                mode=spike_mode,
                white_noise_amplitude=spike_noise_amplitude, 
                pink_noise_amplitude=spike_noise_amplitude
            )

            if spike_mode == 'complex':
                spike_start_idx = np.where(channel_data != 0)[0][0]
                spike_end_idx = np.where(channel_data != 0)[0][-1]
                combined_data = np.concatenate([
                    eeg_signal[:spike_start_idx],
                    channel_data[spike_start_idx:spike_end_idx + 1],
                    eeg_signal[spike_end_idx + 1:]
                ])
            else:
                combined_data = np.where(channel_data != 0, channel_data, eeg_signal)

            data_for_edf.append(combined_data)

        channel_names = [f"Channel_{i}" for i in range(1, n_channels + 1)]
        save_to_edf(data_for_edf, sfreq, channel_names, "assets/output_file.edf")
        save_to_txt(data_for_edf, channel_names, "assets/output_file.txt")  # Aquí añadimos la línea para guardar en .txt

        return "Data saved successfully!"  # Este mensaje puede ser personalizado


    @app.callback(
        [Output('eeg-plot', 'figure'), Output('eeg-image', 'src'), Output('download-image', 'href'), Output('download-edf', 'href')],
        [Input('show-button', 'n_clicks')]
    )
    def update_graph(n_clicks):
        raw = mne.io.read_raw_edf("assets/output_file.edf", preload=True) # Añade 'assets/' antes de 'output_file.edf'
        data, times = raw[:, :] 

        if n_clicks == 0:
            return go.Figure(), None, '', ''

        # Utiliza MNE para visualizar y guardar la figura
        fig_mne = raw.plot(n_channels=15, scalings={"eeg": 100e-6}, show=False)  
        fig_mne.savefig('assets/EEG_plot.png')
        # Actualiza el timestamp para que el navegador no use la imagen en caché
        timestamp = time.time()
        image_path = f'/assets/EEG_plot.png?{timestamp}'
        edf_path = f'/assets/output_file.edf?{timestamp}'
        txt_path = f'/assets/output_file.txt?{timestamp}'


        # Crea un subplot para cada canal
        fig = make_subplots(rows=len(data), cols=1, shared_xaxes=True, vertical_spacing=0.02)

        for index, channel_data in enumerate(data):
            trace = go.Scattergl(x=times, y=channel_data, mode='lines', name=f'Channel {index + 1}')
            fig.add_trace(trace, row=index + 1, col=1)
    
        # Estilizar los ejes
        fig.update_layout(height=300 * len(data), title_text="EEG Signal", showlegend=False)

        return fig, image_path, image_path, edf_path  # Devuelve cuatro valores, uno para cada Output








