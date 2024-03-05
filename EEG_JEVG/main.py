import dash
from ui_definition import get_layout
from callbacks_and_logic import register_callbacks
from waitress import serve

app = dash.Dash(__name__)
app.layout = get_layout()

register_callbacks(app)

from waitress import serve
serve(app.server, host="0.0.0.0", port=8080)
