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
from signal_generator import delta_band
import plotly.io as pio



# Callback para actualizar la UI basada en el tipo de onda seleccionado
def register_callbacks_detailed(app):
    @app.callback(
        [Output("amplitude-slider-spike", "style"),
        Output("amplitude-slider-slow", "style"),
        Output("duration-slider-spike", "style"),
        Output("duration-slider-slow", "style"),
        Output("amplitude-slider-spike-slow", "style"),
        Output("duration-slider-spike-slow", "style")],
        [Input("wave-selector", "value")]
    )
    ## Esta función actualiza la UI basada en el tipo de onda seleccionado
    def update_ui_based_on_wave(wave_selector):
        if wave_selector == "spike":
            return {}, {'display': 'none'}, {}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        elif wave_selector == "slow":
            return {'display': 'none'}, {}, {'display': 'none'}, {}, {'display': 'none'}, {'display': 'none'}
        elif wave_selector == "spike-slow":
            return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {}, {}
        else:
            return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

    ## Esta función actualiza el gráfico y almacena los datos generados
    @app.callback(
        [Output("graph", "figure"),
        Output('store-data', 'data')],
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
        figure = go.Figure()
        data_for_storage = [] 

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
            elif wave_selector == "spike":
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
            else:
                channel_data = np.zeros(len(times))
            # Mezcla la señal generada con la señal EEG
            combined_data = np.where(channel_data != 0, channel_data, eeg_signal)
            # Agrega la señal combinada como trazo, etiquetada como "Señal EEG"
            figure.add_trace(go.Scattergl(x=times, y=combined_data, mode='lines', name=f'Señal EEG {ch + 1}'))
            # Almacena la señal combinada
            data_for_storage.append(combined_data.tolist())

        stored_data = {'data': data_for_storage, 'sfreq': sfreq}
        return figure, stored_data

    ## Esta función convierte los datos almacenados en formato EDF
    def convert_stored_data_to_edf_format(stored_data):
        data_for_edf = stored_data['data']  # Suponiendo que 'data' es una lista de señales
        return data_for_edf

    ## Esta función guarda los datos almacenados en un archivo EDF
    @app.callback(
        Output('save-message', 'children'),
        [Input('save-button', 'n_clicks')],
        [State('store-data', 'data')]
    )
    def save_data(n_clicks, stored_data):
        if not n_clicks or stored_data is None:
            raise PreventUpdate
        # Convertimos stored_data['data'] a arrays de NumPy
        data_for_edf = [np.array(d) for d in stored_data['data']]
        sfreq = stored_data['sfreq']
        channel_names = [f"Channel_{i+1}" for i in range(len(data_for_edf))]
        save_to_edf(data_for_edf, sfreq, channel_names, "assets/output_file.edf")
        # Guardar también en formato TXT
        save_to_txt(data_for_edf, sfreq, "assets/output_file.txt")
        
        return "Data saved successfully!"

    ## Esta función actualiza el gráfico y almacena los datos generados
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
            # Si channel_data es un array 2D, selecciona la primera fila.
            if isinstance(channel_data, np.ndarray) and channel_data.ndim > 1:
                channel_data = channel_data[0]
            trace = go.Scattergl(x=times, y=channel_data, mode='lines', name=f'Channel {index + 1}')
            fig.add_trace(trace, row=index + 1, col=1)
        fig.update_layout(height=300 * len(data), title_text="EEG Signal", showlegend=False)

        return fig, image_path, image_path, edf_path