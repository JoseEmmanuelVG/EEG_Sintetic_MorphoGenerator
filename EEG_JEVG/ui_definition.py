from dash import dcc
import dash_html_components as html

def get_layout():
    return html.Div([
    html.H1("Simulador de ondas epilépticas"),
    dcc.Graph(id="graph"),

    html.Label('Amplitud (µV)'),
    dcc.RangeSlider(id='amplitude-slider', min=-10, max=100, step=2, value=[5, 50]),  

    html.Label('Duración'),
    dcc.RangeSlider(id='duration-slider', min=0.1, max=2, step=0.1, value=[1, 1.5]),

    html.Label('Frecuencia de muestreo'),
    dcc.Slider(id='sfreq-slider', min=100, max=1000, step=50, value=500),

    html.Label('Número de Canales'),
    dcc.Input(id='channels-input', type='number', min=1, value=1),

    html.Label('Número de Puntas'),
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
    dcc.Slider(id='eeg-amplitude-slider', min=0, max=100, step=1, value=50),  # Cambiamos el rango para permitir valores entre 0 y 100
    ])