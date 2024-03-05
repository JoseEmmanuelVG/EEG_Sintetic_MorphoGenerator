import dash
from ui_definition import get_layout
from callbacks_default_page import register_callbacks_fast
from callbacks_detailed_page import register_callbacks_detailed  
from waitress import serve
from dash.dependencies import Output, Input
from ui_definition import generacion_rapida_layout, generacion_detallada_layout, homepage_layout

## Se crea la aplicación de Dash
app = dash.Dash(__name__)
app.layout = get_layout()
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])

## Se define la función que maneja el contenido de la página
def display_page(pathname):
    if pathname == '/generacion-rapida':
        return generacion_rapida_layout()
    elif pathname == '/generacion-detallada':
        return generacion_detallada_layout()
    else:
        return homepage_layout()

## Se registran los callbacks
register_callbacks_fast(app)
register_callbacks_detailed(app)

## Se inicia el servidor
from waitress import serve
serve(app.server, host="0.0.0.0", port=8080)

