import warnings
from dash import dcc, html
import dash_bootstrap_components as dbc
from src.utils import sidebar,nav, title

warnings.filterwarnings("ignore")

def layout(children):
    return dbc.Container([
        # sidebar,
        nav,
        # dbc.Row(html.H2(html.B("Exploratory Data Analysis")),justify="center"),
        title,
        html.Div(children)
    ])