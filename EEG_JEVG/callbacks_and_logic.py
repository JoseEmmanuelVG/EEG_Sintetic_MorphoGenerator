from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from signal_generator import generate_spikes_channel, generate_eeg_signal, save_to_edf
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
         Input("eeg-amplitude-slider", "value")]
    )
    def update_graph(amplitude, duration, sfreq, spike_noise_amplitude, n_channels, n_spikes, total_time, spike_mode,
                     delta_band, theta_band, alpha_band, beta_band, gamma_band, noise_amplitude, eeg_amplitude):

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
        [State("amplitude-slider", "value"),
         State("duration-slider", "value"),
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
    def save_data(n_clicks, amplitude, duration, sfreq, spike_noise_amplitude, n_channels, n_spikes, total_time, spike_mode,
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
        save_to_edf(data_for_edf, sfreq, channel_names, "output_file.edf")  # Asumo que la función `save_to_edf` maneja el nombre del archivo y su ubicación

        return "Data saved successfully!"  # Este mensaje puede ser personalizado
