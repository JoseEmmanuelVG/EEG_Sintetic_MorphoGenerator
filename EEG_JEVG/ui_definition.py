from dash import dcc
from dash import html

def get_layout():
    return html.Div([
        dcc.Location(id='url', refresh=False),  # Manejo de URLs
        html.Div(id='page-content')  # Contenido basado en la URL
    ])


# Página de inicio
def homepage_layout():
    return html.Div([
        html.H1("Bienvenido al Simulador de ondas epilépticas"),
        html.Br(),
        html.Hr(),
        html.Div([
            html.A('Generación rápida', href='/generacion-rapida'),
            html.Br(),
            html.A('Generación detallada', href='/generacion-detallada')
        ], className='button-container')
    ])


# Página de Generación rápida (placeholder)
def generacion_rapida_layout():
    return html.Div([
        html.H1('Generación Rápida'),
        html.Br(),
        html.Hr(),
        html.A('Volver al inicio', href='/'),  
        html.Label('Número de Ondas:'),  
        dcc.RangeSlider(  # Modificado a RangeSlider para seleccionar un rango
            id='ondas-slider',
            min=0,
            max=100,
            value=[10, 50],  # Valor inicial del rango
            marks={i: str(i) for i in range(0, 101, 10)},
            step=1,
            tooltip={'always_visible': True},
            updatemode='drag',
        ),
        html.Br(),
        html.Label('Número de Canales:'),  # Etiqueta para el input de canales
        dcc.Input(  # Campo de texto para introducir el número de canales
            id='canales-input',
            type='number',  # Asegura que solo se puedan introducir números
            value=5,  # Valor inicial
        ),
        html.Br(),
        html.Label('Número de Hojas (Numero par: Se genera señal compleja y transitoria):'),  # Etiqueta modificada para el input de hojas
        dcc.Input(
            id='hojas-input',
            type='number',
            min=2,  # El mínimo ahora es 2 para asegurar que siempre haya al menos un par
            step=2,  # Se incrementa de dos en dos para asegurarse de que sea par
            value=2,  # Valor inicial
        ),
        html.Br(),
        html.Button('Generar', id='generate-button', n_clicks=0),
    
    
    dcc.Dropdown(
            id='onda-selector',
            options=[
                {'label': 'Puntas', 'value': 'puntas'},
                {'label': 'Ondas Lentas', 'value': 'ondas_lentas'},
                {'label': 'Punta-Onda Lenta', 'value': 'punta_onda_lenta'}
            ],
        ),
        dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0),
        html.Div(id='image-check-state', style={'display': 'none'}, children="not_generated"),
        dcc.Graph(id='rapid-graph'),
        html.Div(id='onda-output') # Placeholder for showing results
    ])















# Página de Generación detallada 
def generacion_detallada_layout():
    return html.Div([
        html.H1('Generación Detallada'),
        html.Br(),
        html.Hr(),
        html.A('Volver al inicio', href='/'),


        html.Label('Tipo de Onda:'),
        dcc.Dropdown(
            id='wave-selector',
            options=[
                {'label': 'Ondas Punta', 'value': 'spike'},
                {'label': 'Ondas Lentas', 'value': 'slow'},
                {'label': 'Ondas Punta-Lenta', 'value': 'spike-wave'},
            ],
            value='spike'  # Valor por defecto
        ),

    
    dcc.Graph(id="graph"),

html.Div(id='amplitude-div', children=[
    html.Label('Amplitud (µV)'),
    dcc.RangeSlider(id='amplitude-slider-spike', min=-10, max=100, step=2, value=[5, 50]),  
    dcc.RangeSlider(id='amplitude-slider-slow', min=-10, max=125, step=5, value=[5, 50]),  
]),
html.Div(id='duration-div', children=[
    html.Label('Duración'),
    dcc.RangeSlider(id='duration-slider-spike', min=0.01, max=0.1, step=0.01, value=[0.02, 0.07]),
    dcc.RangeSlider(id='duration-slider-slow', min=0.1, max=0.3, step=0.02, value=[0.15,0.25 ]),
]),


    html.Label('Frecuencia de muestreo'),
    dcc.Slider(id='sfreq-slider', min=100, max=1000, step=50, value=500),

    html.Label('Número de Canales'),
    dcc.Input(id='channels-input', type='number', min=1, value=1),

    html.Label('Número de Ondas'),
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
    dcc.Slider(id='eeg-amplitude-slider', min=0, max=100, step=1, value=25),  # Cambiamos el rango para permitir valores entre 0 y 100


    # Botón para guardar el archivo EDF
        html.Button('Guardar como EDF', id='save-button', n_clicks=0),
        html.Div(id='save-message'),
   
        # Visor Edf
        dcc.Graph(id='eeg-plot'),
        html.Button('Show EEG', id='show-button', n_clicks=0),
        html.Img(id='eeg-image'),
        
        # Botón para descargar la imagen EEG
        html.A(
            'Descargar Imagen EEG',
            id='download-image',
            download='EEG_plot.png',
            href='/assets/EEG_plot.png',
            target="_blank",
            className='btn'
        ),
        
        # Botón para descargar el archivo EDF
        html.A(
            'Descargar Archivo EDF',
            id='download-edf',
            download='output_file.edf',
            href='/assets/output_file.edf', # Asegúrate de tener 'assets/' antes de 'output_file.edf'
            target="_blank",
            className='btn'
        ),
    ])