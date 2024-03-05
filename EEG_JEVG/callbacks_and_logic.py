import dash
import matplotlib
matplotlib.use('Agg')

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from signal_generator import generate_spike, generate_spikes_channel, generate_eeg_signal, save_to_edf, generate_spike_wave_group
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



def register_callbacks(app):
    
    @app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
    def display_page(pathname):
        if pathname == '/generacion-rapida':
            return generacion_rapida_layout()
        elif pathname == '/generacion-detallada':
            return generacion_detallada_layout()
        else:
            return homepage_layout()



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
        elif wave_selector == "mix-wave":
            return {}, {}, {}, {}
        else:  # Asumimos que cualquier otro valor es un spike
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
        elif wave_selector == "mix-wave":
            generated_wave = generate_spike_wave_group(sfreq)
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
        elif wave_selector == "mix-wave":
            generated_wave = generate_spike_wave_group(sfreq)
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


        # Crea un subplot para cada canal
        fig = make_subplots(rows=len(data), cols=1, shared_xaxes=True, vertical_spacing=0.02)

        for index, channel_data in enumerate(data):
            trace = go.Scattergl(x=times, y=channel_data, mode='lines', name=f'Channel {index + 1}')
            fig.add_trace(trace, row=index + 1, col=1)
    
        # Estilizar los ejes
        fig.update_layout(height=300 * len(data), title_text="EEG Signal", showlegend=False)

        return fig, image_path, image_path, edf_path  # Devuelve cuatro valores, uno para cada Output








# Página de Generación rápida (placeholder)

    type_to_prefix = {
        "puntas": "Puntas",
        "ondas_lentas": "Lentas",
        "punta_onda_lenta": "Mixta"
    }

    @app.callback(
        [Output("rapid-graph", "figure"),
        Output("onda-output", "children")],
        [Input("generate-button", "n_clicks"),
        Input("onda-selector", "value")],
        [State("ondas-slider", "value"),
        State("canales-input", "value"),
        State("hojas-input", "value"),
        State("onda-selector", "value")]
    )
    def combined_callback(n_clicks, onda_selector_value, num_ondas_range, num_canales, num_hojas, onda_type):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        else:
            prop_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if prop_id == 'generate-button':
            # If the trigger was the generate-button
            return generate_rapid_data(n_clicks, num_ondas_range, num_canales, num_hojas, onda_type)
        elif prop_id == 'onda-selector':
            # If the trigger was the onda-selector
            return dash.no_update, display_selected_onda(onda_selector_value)  # The dash.no_update indicates no change to the output
        else:
            raise PreventUpdate

    def generate_eeg_signal(freq_bands, freq_weights, duration, sampling_freq, noise_amplitude=5.0):
        """
        Generar una señal de EEG basada en bandas de frecuencia.
        """
        t = np.arange(0, duration, 1.0/sampling_freq)
        signal = np.zeros(t.shape)

        for f, w in zip(freq_bands, freq_weights):
            signal += w * np.sin(2 * np.pi * f * t)

        noise = noise_amplitude * np.random.randn(signal.shape[0])
        signal += noise

        return signal


    def generate_rapid_data(n_clicks, num_ondas_range, num_canales, num_hojas, onda_type):
        if not n_clicks:
            raise PreventUpdate
        
        # Borrar archivos antiguos
        old_files = glob.glob('assets/Puntas_*') + glob.glob('assets/sheet_*.edf')
        for f in old_files:
            os.remove(f)


        min_onda, max_onda = num_ondas_range

        all_figures = []
        all_edf_files = []

        # Asegurar que num_hojas es un número par
        if num_hojas % 2 != 0:
            num_hojas += 1

        for sheet_num in range(1, num_hojas + 1):
            all_channels_data = []
            
            # Decide el modo basado en el número de hoja (transitorio o complejo)
            spike_mode = "transitory" if sheet_num % 2 == 0 else "complex"

            for _ in range(num_canales):
                eeg_data = generate_eeg_signal([10], [30], duration=10, sampling_freq=1000, noise_amplitude=5.0)

                if onda_type == "puntas":
                    # Aquí generamos la señal base
                    eeg_data = generate_eeg_signal([10], [30], duration=10, sampling_freq=1000, noise_amplitude=5.0)
                    
                    # Añadir spikes
                    num_spikes = random.randint(min_onda, max_onda)  # Numero de spikes entre min y max
                    spike_amplitude_range = (20, 100)  # Definir el rango de amplitud para las puntas
                    spike_duration_range = (0.1, 0.3)  # Duración de las puntas

                    for _ in range(num_spikes):
                        amplitude_val = random.uniform(*spike_amplitude_range)
                        duration_val = random.uniform(*spike_duration_range)
                        spike = generate_spike(amplitude_val, duration_val, 1000)
                        
                        # Decidir dónde colocar el spike en la señal EEG
                        start_index = random.randint(0, len(eeg_data) - len(spike) - 1)
                        eeg_data[start_index:start_index + len(spike)] += spike
                        
                
                
                elif onda_type == "ondas_lentas":
                    # Implementa lógica similar para generar ondas lentas
                    pass
                elif onda_type == "punta_onda_lenta":
                    # Implementa lógica similar para generar punta-onda lenta
                    pass
                else:
                    raise ValueError(f"Unknown onda_type: {onda_type}")

                all_channels_data.append(eeg_data)

            # Saving the EDF files directly in the 'assets' folder without subfolder
            output_filename = os.path.join('assets', f"sheet_{sheet_num}.edf")
            all_edf_files.append(output_filename)

            channel_names = [f"Channel_{i+1}" for i in range(num_canales)]
            save_to_edf(np.array(all_channels_data), 1000, channel_names, output_filename)

            # Creación de figuras usando Plotly para cada hoja
            fig = make_subplots(rows=num_canales, cols=1, shared_xaxes=True)

            for i, data in enumerate(all_channels_data):
                fig.add_trace(go.Scatter(y=data, mode='lines', name=f'Channel {i+1}'), row=i+1, col=1)

            all_figures.append(fig)

            # Guardar la figura como imagen PNG
            image_filename = os.path.join('assets', f"{type_to_prefix[onda_type]}_{sheet_num}.png")
            pio.write_image(fig, image_filename)

        # Crear un archivo zip para las imágenes
        with zipfile.ZipFile('assets/images.zip', 'w') as zipf:
            for sheet_num in range(1, num_hojas + 1):
                image_filename = os.path.join('assets', f"{type_to_prefix[onda_type]}_{sheet_num}.png")
                zipf.write(image_filename)

        # Crear un archivo zip para los EDFs (opcional, como mencioné)
        with zipfile.ZipFile('assets/edfs.zip', 'w') as zipf:
            for edf_file in all_edf_files:
                zipf.write(edf_file)

        download_links = [
            html.A('Descargar Todas las Imágenes', href='/assets/images.zip', download='images.zip'),
            html.A('Descargar Todos los EDFs', href='/assets/edfs.zip', download='edfs.zip')
        ]

        return all_figures[0], html.Div(download_links)

    def display_selected_onda(onda_type):
        if onda_type not in type_to_prefix:
            raise ValueError(f"Unknown onda_type: {onda_type}")

        prefix = type_to_prefix[onda_type]
        folder_path = 'assets'  

        max_images = 100  
        components = []

        for i in range(1, max_images + 1):
            img_file_name = f"{prefix}_{i}.png"  # Nota el cambio en el nombre del archivo
            img_path = os.path.join(folder_path, img_file_name)

            if os.path.exists(img_path):
                components.append(html.Img(src=img_path, style={"width": "100%", "height": "auto"}))
                components.append(html.Br())

        return html.Div(components)
