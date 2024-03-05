from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from signal_generator import generate_spikes_channel, generate_eeg_signal
import numpy as np 

def register_callbacks(app):

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
     Input("eeg-amplitude-slider", "value")]    )
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