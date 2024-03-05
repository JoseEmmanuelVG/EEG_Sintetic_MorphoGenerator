import dash
import matplotlib
matplotlib.use('Agg')
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from signal_generator_default import generate_spike, generate_slow_wave, generate_eeg_signal, save_to_edf, create_mne_visualization, save_to_txt, generate_spike_slow_wave
import numpy as np 
from plotly.subplots import make_subplots
import time
from ui_definition import generacion_rapida_layout
from dash import html
import os
from signal_generator_default import delta_band, theta_band, alpha_band, beta_band
import plotly.io as pio
import random
import glob
import zipfile

# Página de Generación predeterminada (placeholder)
def register_callbacks_fast(app):
    type_to_prefix = {
        "puntas": "Puntas",
        "ondas_lentas": "Lentas",
        "punta_onda_lenta": "Mixta"
    }

    ## Esta función actualiza el gráfico y almacena los datos generados
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
            # Si el disparador fue el botón generar
            return generate_rapid_data(n_clicks, num_ondas_range, num_canales, num_hojas, onda_type)
        elif prop_id == 'onda-selector':
            # Si el disparador fue el onda-selector
            return dash.no_update, display_selected_onda(onda_selector_value)  # El guión.no_update indica que no hay cambios en la salida
        else:
            raise PreventUpdate

    # Funciones para la generación de ondas
    def generate_events_transitory(eeg_data, onda_type, sfreq, num_ondas_range):
        # Asumimos que num_ondas_range es una tupla (min_onda, max_onda)
        num_events = random.randint(*num_ondas_range)
        for _ in range(num_events):
            if onda_type == "puntas":
                event = generate_spike(random.uniform(40, 100), random.uniform(0.02, 0.07), sfreq)
            elif onda_type == "ondas_lentas":
                event = generate_slow_wave(random.uniform(40, 80), random.uniform(0.2, 0.5), sfreq)
            elif onda_type == "punta_onda_lenta":
                event = generate_spike_slow_wave(sfreq, (40, 100), (0.02, 0.07), (40, 80), (0.2, 0.5))
            else:
                continue  # Si el tipo de onda no es reconocido, saltamos este ciclo

            # Añadir el evento a un punto aleatorio en eeg_data
            start_index = random.randint(0, len(eeg_data) - len(event))
            eeg_data[start_index:start_index + len(event)] += event
        return eeg_data

    # Funciones para la generación de ondas
    def generate_events_complex(eeg_data, onda_type, sfreq, num_ondas_range):
        num_events = random.randint(*num_ondas_range)

        # Estimar la duración promedio de los eventos basada en el tipo de onda
        if onda_type == "puntas":
            average_event_duration = int(0.045 * sfreq)  # Asumiendo duración promedio para 'puntas'
        elif onda_type == "ondas_lentas":
            average_event_duration = int(0.35 * sfreq)  # Asumiendo duración promedio para 'ondas_lentas'
        elif onda_type == "punta_onda_lenta":
            average_event_duration = int(0.57 * sfreq)  # Asumiendo duración promedio para 'punta_onda_lenta'
        else:
            raise ValueError("Tipo de onda no reconocido")
        total_event_space = num_events * average_event_duration
        total_length = len(eeg_data)

        if total_event_space > total_length:
            raise ValueError("No hay suficiente espacio en la señal para todos los eventos complejos")
        padding_space = (total_length - total_event_space) // 2
        current_index = padding_space

        for _ in range(num_events):
            if onda_type == "puntas":
                event = generate_spike(random.uniform(40, 100), random.uniform(0.02, 0.07), sfreq)
            elif onda_type == "ondas_lentas":
                event = generate_slow_wave(random.uniform(40, 80), random.uniform(0.2, 0.5), sfreq)
            elif onda_type == "punta_onda_lenta":
                event = generate_spike_slow_wave(sfreq, (40, 100), (0.02, 0.07), (40, 80), (0.2, 0.5))
            event_length = len(event)
            end_index = min(current_index + event_length, total_length)
            eeg_data[current_index:end_index] += event[:end_index-current_index]
            current_index += event_length  # Preparar el índice para el próximo evento

        return eeg_data

    # Función para la generación de datos rápidos
    def generate_rapid_data(n_clicks, num_ondas_range, num_canales, num_hojas, onda_type):
        sfreq = 200  # Frecuencia de muestreo para la generación de señales
        # Definir los rangos de amplitud y duración para picos y ondas lentas
        amplitude_spike_range = (40, 100)  # Ejemplo de rango de amplitud de los picos
        duration_spike_range = (0.02, 0.07)  # Ejemplo de rango de duración de los picos
        amplitude_slow_range = (40, 80)  # Ejemplo de rango de amplitud de las ondas lentas
        duration_slow_range = (0.2, 0.5)  # Ejemplo de rango de duración de las ondas lentas
        # Define las bandas de frecuencia y sus respectivas amplitudes como ejemplo
        freq_bands = [delta_band, theta_band, alpha_band, beta_band]
        amplitudes = [1, 0.5, 0.3, 0.2] 

        if not n_clicks:
            raise PreventUpdate

        # Limpieza de archivos antiguos
        old_files = glob.glob('assets/Puntas_*') + glob.glob('assets/sheet_*.edf') + glob.glob('assets/MNE_Puntas_*')
        for f in old_files:
            os.remove(f)
        min_onda, max_onda = num_ondas_range
        all_figures = []
        all_edf_files = []
        all_txt_files = []

        all_figures = []  # Lista para almacenar las figuras generadas
        # Ajustar el número de hojas para que sea par, en caso de que el usuario ingrese un número impar
        num_hojas_adj = num_hojas if num_hojas % 2 == 0 else num_hojas + 1

        for sheet_num in range(1, num_hojas_adj + 1):
            all_channels_data = []

            # Alternar entre eventos transitorios y complejos cada hoja
            mode = "transitory" if sheet_num % 2 == 1 else "complex"

            for _ in range(num_canales):
                # Generar la señal base EEG
                eeg_data = generate_eeg_signal(freq_bands, amplitudes, duration=10, sampling_freq=sfreq, noise_amplitude=1.0)

                # Aplicar eventos según el modo
                if mode == "transitory":
                    eeg_data = generate_events_transitory(eeg_data, onda_type, sfreq, num_ondas_range)
                else:
                    eeg_data = generate_events_complex(eeg_data, onda_type, sfreq, num_ondas_range)
                all_channels_data.append(eeg_data)


                if onda_type == "puntas":
                    # Aquí generamos la señal base
                    eeg_data = generate_eeg_signal(freq_bands, amplitudes, duration=10, sampling_freq=200, noise_amplitude=3.0)
                    # Añadir spikes
                    num_spikes = random.randint(min_onda, max_onda)  # Numero de spikes entre min y max
                    spike_amplitude_range = (40, 100)  # Definir el rango de amplitud para las puntas
                    spike_duration_range = (0.02, 0.07)  # Duración de las puntas

                    for _ in range(num_spikes):
                        amplitude_val = random.uniform(*spike_amplitude_range)
                        duration_val = random.uniform(*spike_duration_range)
                        spike = generate_spike(amplitude_val, duration_val, 200)
                        # Decidir dónde colocar el spike en la señal EEG
                        start_index = random.randint(0, len(eeg_data) - len(spike) - 1)
                        eeg_data[start_index:start_index + len(spike)] += spike
                                                        
                elif onda_type == "ondas_lentas":
                    # Aquí generamos la señal base
                    eeg_data = generate_eeg_signal(freq_bands, amplitudes, duration=10, sampling_freq=200, noise_amplitude=3.0)               
                    # Añadir ondas lentas
                    num_slow_waves = random.randint(min_onda, max_onda)  # Número de ondas lentas entre min y max
                    slow_wave_amplitude_range = (40, 80)  # Definir el rango de amplitud para las ondas lentas
                    slow_wave_duration_range = (0.2, 0.5)  # Duración de las ondas lentas (generalmente entre 0.2 y 0.5 segundos)

                    for _ in range(num_slow_waves):
                        amplitude_val = random.uniform(*slow_wave_amplitude_range)
                        duration_val = random.uniform(*slow_wave_duration_range)
                        slow_wave = generate_slow_wave(amplitude_val, duration_val, 200)
                        # Decidir dónde colocar la onda lenta en la señal EEG
                        start_index = random.randint(0, len(eeg_data) - len(slow_wave) - 1)
                        eeg_data[start_index:start_index + len(slow_wave)] += slow_wave


                elif onda_type == "punta_onda_lenta":
                    # Aquí generamos la señal base
                    eeg_data = generate_eeg_signal(freq_bands, amplitudes, duration=10, sampling_freq=200, noise_amplitude=3.0)
                    # Calcular el número de ondas punta-lenta para generar dentro del rango especificado
                    num_spike_slow_waves = random.randint(min_onda, max_onda)

                    for _ in range(num_spike_slow_waves):
                        spike_slow_wave = generate_spike_slow_wave(
                            sfreq, 
                            amplitude_spike_range, 
                            duration_spike_range, 
                            amplitude_slow_range, 
                            duration_slow_range
                        )
                        # Decidir dónde colocar la onda punta-lenta en la señal EEG
                        # Asegurarse de que hay espacio suficiente para la onda
                        start_index = np.random.randint(0, len(eeg_data) - len(spike_slow_wave))
                        # Añadir la onda punta-lenta a la señal EEG
                        eeg_data[start_index:start_index + len(spike_slow_wave)] += spike_slow_wave
                    # Añadir grupos de punta-onda lenta
                    num_groups = random.randint(min_onda, max_onda)  # Número de grupos entre min y max
                    group_duration = 3  # Duración aproximada de un grupo punta-onda lenta              
                    # Esta lista ayudará a asegurar que las ondas punta-lenta no se superpongan
                    occupied_indices = []

                    for _ in range(num_groups):
                        spike_wave_group = generate_spike_slow_wave(
                            sfreq, 
                            amplitude_spike_range, 
                            duration_spike_range, 
                            amplitude_slow_range, 
                            duration_slow_range
                    )
                    # Asegurarse de que el índice de inicio + la longitud de `spike_wave_group` no exceda la longitud de `eeg_data`
                    # y que no se superpongan las ondas
                    valid_start_index = False
                    while not valid_start_index:
                        start_index = np.random.randint(0, len(eeg_data) - len(spike_wave_group))
                        end_index = start_index + len(spike_wave_group)
                        # Verificar si el rango de índices está ocupado
                        if all(start_index >= i[1] or end_index <= i[0] for i in occupied_indices):
                            valid_start_index = True
                            occupied_indices.append((start_index, end_index))
                    eeg_data[start_index:start_index + len(spike_wave_group)] += spike_wave_group
                          
                    pass
                else:
                    raise ValueError(f"Unknown onda_type: {onda_type}")

                # Suponiendo que all_channels_data ya se ha llenado correctamente...
                all_channels_data_np = np.array(all_channels_data)  # Asegúrate de que esto tenga la forma correcta (n_canales, n_muestras)

                if all_channels_data_np.ndim == 2 and all_channels_data_np.shape[0] == num_canales:
                    # Guardar como EDF
                    output_filename_edf = os.path.join('assets', f"sheet_{sheet_num}.edf")
                    save_to_edf(all_channels_data_np, sfreq, [f"Channel_{i+1}" for i in range(num_canales)], output_filename_edf)
                    all_edf_files.append(output_filename_edf)  # Asegúrate de que esta lista se use correctamente para el ZIP
                    # Guardar como TXT (Si es necesario)
                    output_filename_txt = os.path.join('assets', f"sheet_{sheet_num}.txt")
                    save_to_txt(all_channels_data_np, [f"Channel_{i+1}" for i in range(num_canales)], output_filename_txt)
                    all_txt_files.append(output_filename_txt)  # Asegúrate de que esta lista se use correctamente para el ZIP
                else:
                    print("Error: Los datos de los canales no tienen la forma esperada.")
            # Guardar los archivos EDF directamente en la carpeta 'assets' sin subcarpeta.
            output_filename = os.path.join('assets', f"sheet_{sheet_num}.edf")
            save_to_edf(all_channels_data_np, sfreq, [f"Channel_{i+1}" for i in range(num_canales)], output_filename)

            # Creación de figuras usando Plotly para cada hoja
            fig = make_subplots(rows=num_canales, cols=1, shared_xaxes=True)
            for i, data in enumerate(all_channels_data):
                fig.add_trace(go.Scatter(y=data, mode='lines', name=f'Channel {i+1}'), row=i+1, col=1)
            all_figures.append(fig)

            # Llamada a la función de visualización de MNE
            mne_image_filename = os.path.join('assets', f"MNE_{type_to_prefix[onda_type]}_{sheet_num}.png")
            create_mne_visualization(np.array(all_channels_data), 200, num_canales, mne_image_filename)

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

        if not all_figures:  # Verificar si all_figures está vacío
            # Si all_figures está vacío, inicializa un fig predeterminado o maneja el caso adecuadamente
            fig = make_subplots(rows=1, cols=1)
            fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='Placeholder'))
            all_figures.append(fig)

        # Ahora puedes acceder de forma segura a all_figures[0] porque sabes que hay al menos un elemento
        return all_figures[0], html.Div(download_links)




    # Función para mostrar la onda seleccionada
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


    # Función para actualizar los botones de descarga
    @app.callback(
        [Output("download-images-btn", "style"),
        Output("download-edfs-btn", "style"),
        Output("download-txts-btn", "style")],  # Asegúrate de controlar también el botón de TXT
        [Input("page-content", "children")]  # Cambia esto para que se dispare cuando la página se cargue o actualice
    )
    def update_download_buttons(_):
        # No necesitas los datos de la figura o el tipo de onda para actualizar los botones
        images_btn_style = {"display": "none"}
        edfs_btn_style = {"display": "none"}
        txts_btn_style = {"display": "none"}  # Inicializa el estilo del botón de TXT también

        # Verifica si hay archivos de imagen disponibles
        if len(glob.glob('assets/*.png')) > 0:  # Busca cualquier archivo PNG en la carpeta 'assets'
            images_btn_style = {"display": "block"}

        # Verifica si hay archivos EDF disponibles
        if len(glob.glob('assets/*.edf')) > 0:
            edfs_btn_style = {"display": "block"}

        # Verifica si hay archivos TXT disponibles
        if len(glob.glob('assets/*.txt')) > 0:
            txts_btn_style = {"display": "block"}

        return images_btn_style, edfs_btn_style, txts_btn_style

    # Y asegúrate de que los botones estén inicialmente visibles en tu layout Dash
    download_links = [
        html.A('Download All Images', id="download-images-btn", href='/assets/images.zip', download='images.zip', style={"display": "block"}),
        html.A('Download All EDFs', id="download-edfs-btn", href='/assets/edfs.zip', download='edfs.zip', style={"display": "block"}),
        html.A('Download All TXTs', id="download-txts-btn", href='/assets/txts.zip', download='txts.zip', style={"display": "block"}),
    ]

