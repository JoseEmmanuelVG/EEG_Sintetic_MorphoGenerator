import dash
import matplotlib
matplotlib.use('Agg')

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from signal_generator_default import generate_spike, generate_spikes_channel, generate_eeg_signal, save_to_edf, generate_spike_wave_group
import numpy as np 
import mne
from plotly.subplots import make_subplots
import time
from ui_definition import generacion_rapida_layout, generacion_detallada_layout, homepage_layout

from dash import dcc
from dash import html
import os
from signal_generator_default import delta_band
import plotly.io as pio
import random

import glob
import zipfile

def generate_slow_wave(amplitude, duration, sampling_freq):
    """Genera una onda lenta (por ejemplo, onda delta) con la amplitud y duración dadas."""
    t = np.arange(0, duration, 1.0/sampling_freq)
    freq = random.uniform(0.5, 4)  # Frecuencia delta típica
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    return wave


def register_callbacks_fast(app):
    
       
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

    def generate_eeg_signal(freq_bands, freq_weights, duration=10, sampling_freq=1000, noise_amplitude=5.0):
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

    def create_mne_visualization(data, sfreq, n_channels, output_filename):
        # Crear objeto Raw de MNE a partir de datos
        info = mne.create_info(ch_names=[f'ch_{i}' for i in range(n_channels)], sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)

        # Ajustar el escalado para EEG en Volts (por ejemplo, 200 microvoltios = 200e-6 Volts)
        scalings = {'eeg': 400e-24}

        # Crear la figura con el escalado ajustado
        fig_mne = raw.plot(n_channels=n_channels, scalings=scalings, show=False)
        fig_mne.set_size_inches(10, 8)
        fig_mne.savefig(output_filename, dpi=600)




    def save_to_txt(data, filename):
        np.savetxt(filename, data)


    def generate_rapid_data(n_clicks, num_ondas_range, num_canales, num_hojas, onda_type):
        sfreq = 1000
        if not n_clicks:
            raise PreventUpdate
        
        # Borrar archivos antiguos
        old_files = glob.glob('assets/Puntas_*') + glob.glob('assets/sheet_*.edf') + glob.glob('assets/MNE_Puntas_*')
        for f in old_files:
            os.remove(f)


        min_onda, max_onda = num_ondas_range

        all_figures = []
        all_edf_files = []
        all_txt_files = []

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
                    eeg_data = generate_eeg_signal([10], [30], duration=10, sampling_freq=1000, noise_amplitude=1.0)
                    
                    # Añadir spikes
                    num_spikes = random.randint(min_onda, max_onda)  # Numero de spikes entre min y max
                    spike_amplitude_range = (40, 100)  # Definir el rango de amplitud para las puntas
                    spike_duration_range = (0.02, 0.07)  # Duración de las puntas

                    for _ in range(num_spikes):
                        amplitude_val = random.uniform(*spike_amplitude_range)
                        duration_val = random.uniform(*spike_duration_range)
                        spike = generate_spike(amplitude_val, duration_val, 1000)
                        
                        # Decidir dónde colocar el spike en la señal EEG
                        start_index = random.randint(0, len(eeg_data) - len(spike) - 1)
                        eeg_data[start_index:start_index + len(spike)] += spike
                        
                
                
                elif onda_type == "ondas_lentas":
                    # Aquí generamos la señal base
                    eeg_data = generate_eeg_signal([10], [30], duration=10, sampling_freq=1000, noise_amplitude=1.0)
                    
                    # Añadir ondas lentas
                    num_slow_waves = random.randint(min_onda, max_onda)  # Número de ondas lentas entre min y max
                    slow_wave_amplitude_range = (20, 60)  # Definir el rango de amplitud para las ondas lentas
                    slow_wave_duration_range = (0.2, 0.5)  # Duración de las ondas lentas (generalmente entre 0.2 y 0.5 segundos)

                    for _ in range(num_slow_waves):
                        amplitude_val = random.uniform(*slow_wave_amplitude_range)
                        duration_val = random.uniform(*slow_wave_duration_range)
                        slow_wave = generate_slow_wave(amplitude_val, duration_val, 1000)
                        
                        # Decidir dónde colocar la onda lenta en la señal EEG
                        start_index = random.randint(0, len(eeg_data) - len(slow_wave) - 1)
                        eeg_data[start_index:start_index + len(slow_wave)] += slow_wave

                elif onda_type == "punta_onda_lenta":
                    # Aquí generamos la señal base
                    eeg_data = generate_eeg_signal([10], [30], duration=10, sampling_freq=1000, noise_amplitude=1.0)

                    # Añadir grupos de punta-onda lenta
                    num_groups = random.randint(min_onda, max_onda)  # Número de grupos entre min y max
                    group_duration = 3  # Duración aproximada de un grupo punta-onda lenta
                                    
                    for _ in range(num_groups):
                        spike_wave_group = generate_spike_wave_group(sfreq, group_duration)
                                        
                        # Decidir dónde colocar el grupo punta-onda lenta en la señal EEG
                        start_index = random.randint(0, len(eeg_data) - len(spike_wave_group) - 1)
                        eeg_data[start_index:start_index + len(spike_wave_group)] += spike_wave_group

                     
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

            # Llamada a la función de visualización de MNE
            mne_image_filename = os.path.join('assets', f"MNE_{type_to_prefix[onda_type]}_{sheet_num}.png")
            create_mne_visualization(np.array(all_channels_data), 1000, num_canales, mne_image_filename)


            # Guardar la figura como imagen PNG
            image_filename = os.path.join('assets', f"{type_to_prefix[onda_type]}_{sheet_num}.png")
            pio.write_image(fig, image_filename)

            # Crear un archivo zip para las imágenes
            with zipfile.ZipFile('assets/images.zip', 'w') as zipf:
                for sheet_num in range(1, num_hojas + 1):
                    image_filename = os.path.join('assets', f"{type_to_prefix[onda_type]}_{sheet_num}.png")
                    mne_image_filename = os.path.join('assets', f"MNE_{type_to_prefix[onda_type]}_{sheet_num}.png")
                    
                    if os.path.exists(image_filename):
                        zipf.write(image_filename)
                    else:
                        print(f"Error: El archivo {image_filename} no se encontró.")

                    if os.path.exists(mne_image_filename):  # Agregar las imágenes de MNE al ZIP
                        zipf.write(mne_image_filename)
                    else:
                        print(f"Error: El archivo {mne_image_filename} no se encontró.")
        
        # Crear un archivo zip para los archivos .TXT
        with zipfile.ZipFile('assets/txts.zip', 'w') as zipf:
            for txt_file in all_txt_files:
                zipf.write(txt_file)


        # Crear un archivo zip para los EDFs (opcional, como mencioné)
        with zipfile.ZipFile('assets/edfs.zip', 'w') as zipf:
            for edf_file in all_edf_files:
                zipf.write(edf_file)

        download_links = [
            html.A('Download All Images', id="download-images-btn", href='/assets/images.zip', download='images.zip', style={"display": "none"}),
            html.A('Download All EDFs', id="download-edfs-btn", href='/assets/edfs.zip', download='edfs.zip', style={"display": "none"}),
            html.A('Download All TXTs', id="download-txts-btn", href='/assets/txts.zip', download='txts.zip', style={"display": "block"}),
        ]
        fig.update_layout(
            xaxis=dict(
                tickvals=[2, 4, 6, 8, 1],
                ticktext=['2s', '4s', '6s', '8s', '10s']
            )
        )
        return all_figures[0], html.Div(download_links)

    def display_selected_onda(onda_type):
        if onda_type not in type_to_prefix:
            raise ValueError(f"Unknown onda_type: {onda_type}")

        prefix = type_to_prefix[onda_type]
        folder_path = 'assets'  

        max_images = 100  
        components = []

        for i in range(1, max_images + 1):
            img_file_name = f"{prefix}_{i}.png"  
            img_path = os.path.join(folder_path, img_file_name)
            
            mne_img_file_name = f"MNE_{prefix}_{i}.png"  # Nombre de archivo para imágenes de MNE
            mne_img_path = os.path.join(folder_path, mne_img_file_name)

            if os.path.exists(img_path):
                timestamp = time.time()
                components.append(html.Img(src=f"{img_path}?_={timestamp}", style={"width": "100%", "height": "auto"}))
                components.append(html.Br())

            if os.path.exists(mne_img_path):  # Agregar la visualización de las imágenes de MNE
                components.append(html.Img(src=mne_img_path, style={"width": "100%", "height": "auto"}))
                components.append(html.Br())

        return html.Div(components)


        # Añadimos una función para verificar la existencia de archivos
    def check_files(file_type):
        """
        Comprueba si existen ciertos tipos de archivos en el directorio 'assets'.
        
        Parámetros:
        - file_type: Un string que indica el tipo de archivo a verificar. Puede ser 'images' o 'edfs'.

        Devuelve:
        - True si los archivos del tipo especificado existen, False en caso contrario.
        """
        
        # Directorio donde verificar los archivos
        directory = 'assets'
        
        if file_type == 'images':
            # Cambia esta línea si la estructura o el formato de nombre de tus imágenes es diferente
            file_pattern = os.path.join(directory, 'Puntas_*.png')
        elif file_type == 'edfs':
            # Cambia esta línea si la estructura o el formato de nombre de tus archivos EDF es diferente
            file_pattern = os.path.join(directory, 'sheet_*.edf')
        else:
            raise ValueError(f"Unknown file_type: {file_type}")

        # Verificar si existen archivos que coincidan con el patrón
        files = glob.glob(file_pattern)
        
        return len(files) > 0


    @app.callback(
        [Output("download-images-btn", "style"),
        Output("download-edfs-btn", "style")],
        [Input("rapid-graph", "figure")]
    )
    def update_download_buttons(fig):
        # Si hay imágenes en assets, muestra el botón de descarga de imágenes
        if check_files('images'):
            images_btn_style = {"display": "block"}
        else:
            images_btn_style = {"display": "none"}

        # Si hay archivos EDF en assets, muestra el botón de descarga de EDFs
        if check_files('edfs'):
            edfs_btn_style = {"display": "block"}
        else:
            edfs_btn_style = {"display": "none"}

        return images_btn_style, edfs_btn_style