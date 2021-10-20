import warnings
from dash import dcc, html
import dash_bootstrap_components as dbc
from src.utils import nav, title

warnings.filterwarnings("ignore")

def layout(children):
    return dbc.Container([
        nav,
        title,
        html.Div(children)
    ])