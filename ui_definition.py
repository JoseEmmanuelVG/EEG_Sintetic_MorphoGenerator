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
        html.H1("Welcome to Epileptic Wave Simulator"),
        html.Br(),
        html.Hr(),
        html.Div([
            html.A('Default generation', href='/generacion-rapida'),
            html.Br(),
            html.A('Detailed Generation', href='/generacion-detallada')
        ], className='button-container')
    ])


# Página de Default generation (placeholder)
def generacion_rapida_layout():
    return html.Div([
        html.H1('Default generation'),
        html.Br(),
        html.Hr(),
        html.A('Back to Menu', href='/'),  
        html.Label('Number of Waves:'),  
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
        html.Label('Number of Channels:'),  # Etiqueta para el input de canales
        dcc.Input(  # Campo de texto para introducir el número de canales
            id='canales-input',
            type='number',  # Asegura que solo se puedan introducir números
            value=5,  # Valor inicial
        ),
        html.Br(),
        html.Label('Number of Leaves (even number: complex and transient signal is generated):'),  # Etiqueta modificada para el input de hojas
        dcc.Input(
            id='hojas-input',
            type='number',
            min=2,  # El mínimo ahora es 2 para asegurar que siempre haya al menos un par
            step=2,  # Se incrementa de dos en dos para asegurarse de que sea par
            value=2,  # Valor inicial
        ),
        html.Br(),
        html.Button('Generate', id='generate-button', n_clicks=0),
    
    
    dcc.Dropdown(
            id='onda-selector',
            options=[
                {'label': 'Spikes', 'value': 'puntas'},
                {'label': 'Slow Waves', 'value': 'ondas_lentas'},
                {'label': 'Spike-Slow Wave', 'value': 'punta_onda_lenta'}
            ],
        ),
        dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0),
        html.Div(id='image-check-state', style={'display': 'none'}, children="not_generated"),
        dcc.Graph(id='rapid-graph'),
        html.Div(id='onda-output') # Marcador de posición para mostrar resultados
    ])















# Página de Detailed Generation 
def generacion_detallada_layout():
    return html.Div([
        html.H1('Detailed Generation'),
        html.Br(),
        html.Hr(),
        html.A('Back to Main', href='/'),


        html.Label('Wave Type:'),
        dcc.Dropdown(
            id='wave-selector',
            options=[
                {'label': 'Spike Wave', 'value': 'spike'},
                {'label': 'Slow Waves', 'value': 'slow'},
                {'label': 'Spike-Slow Waves', 'value': 'spike-slow'},
            ],
            value='spike'  # Valor por defecto
        ),

    
    dcc.Graph(id="graph"),

html.Div(id='amplitude-div', children=[
    html.Label('Amplitud (µV)'),
    dcc.RangeSlider(id='amplitude-slider-spike', min=-10, max=100, step=5, value=[5, 50]),  
    dcc.RangeSlider(id='amplitude-slider-slow', min=-10, max=100, step=5, value=[5, 50]),  
    dcc.RangeSlider(id='amplitude-slider-spike-slow', min=-10, max=100, step=5, value=[5, 50]),  

]),
html.Div(id='duration-div', children=[
    html.Label('Duration'),
    dcc.RangeSlider(id='duration-slider-spike', min=0.02, max=0.07, step=0.005, value=[0.02, 0.07]),
    dcc.RangeSlider(id='duration-slider-slow', min=0.2, max=0.5, step=0.05, value=[0.2,0.5 ]),
    dcc.RangeSlider(id='duration-slider-spike-slow', min=0.22, max=0.57, step=0.05, value=[0.22,0.57]),

]),


    html.Label('Sampling frequency'),
    dcc.Slider(id='sfreq-slider', min=100, max=1000, step=50, value=500),

    #Ruido para las ondas   
    html.Label('Peak Noise Amplitude (White and Pink)'),
    dcc.Slider(id='spike-noise-amplitude', min=0, max=2, step=0.1, value=0),

    html.Label('Number of Channels'),
    dcc.Input(id='channels-input', type='number', min=1, value=1),

    html.Label('Number of Waves'),
    dcc.Input(id='spikes-input', type='number', min=1, value=10),

    html.Label('Total time (s)'),
    dcc.Input(id='time-input', type='number', min=1, value=10),

    html.Label('Peak wave generation mode'),
    dcc.RadioItems(
    id='spike-mode',
    options=[
        {'label': 'Transitory', 'value': 'transient'},
        {'label': 'Complex', 'value': 'complex'}
    ],
    value='transient'),

    html.Label('Delta Band'),
    dcc.RangeSlider(id='delta-band', min=0, max=4, step=0.1, value=[0, 4]),

    html.Label('Theta Band'),
    dcc.RangeSlider(id='theta-band', min=4, max=8, step=0.1, value=[4, 8]),

    html.Label('Alpha Band'),
    dcc.RangeSlider(id='alpha-band', min=8, max=12, step=0.1, value=[8, 12]),

    html.Label('Beta Band'),
    dcc.RangeSlider(id='beta-band', min=12, max=30, step=0.5, value=[12, 30]),

    html.Label('Gamma Band'),
    dcc.RangeSlider(id='gamma-band', min=30, max=70, step=1, value=[30, 70]),

    html.Label('Noise amplitude'),
    dcc.Slider(id='noise-amplitude', min=0, max=2, step=0.1, value=1),

    html.Label('Amplitude of the EEG junction signal (µV)'),
    dcc.Slider(id='eeg-amplitude-slider', min=0, max=100, step=1, value=25),  # Cambiamos el rango para permitir valores entre 0 y 100

    dcc.Store(id='store-data'),
    dcc.Store(id='graph-data-store', storage_type='memory'),

    # Botón para guardar el archivo EDF
        html.Button('Save Signal', id='save-button', n_clicks=0),
        html.Div(id='save-message'),
   
        # Visor Edf
        dcc.Graph(id='eeg-plot'),
        html.Button('Show EEG', id='show-button', n_clicks=0),
        html.Img(id='eeg-image'),
        
        # Botón para descargar la imagen EEG
        html.A(
            'Download EEG Image',
            id='download-image',
            download='EEG_plot.png',
            href='/assets/EEG_plot.png',
            target="_blank",
            className='btn'
        ),
        
        # Botón para descargar el archivo EDF
        html.A(
            'Download EDF File',
            id='download-edf',
            download='output_file.edf',
            href='/assets/output_file.edf', # Asegúrate de tener 'assets/' antes de 'output_file.edf'
            target="_blank",
            className='btn'
        ),

        # Botón para descargar el archivo TXT
        html.A(
            'Download TXT File',
            id='download-txt',
            download='output_file.txt',
            href='/assets/output_file.txt',
            target="_blank",
            className='btn'
        ),

    ])