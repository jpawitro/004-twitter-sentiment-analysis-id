import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from server import app, User
from flask_login import login_user
from werkzeug.security import check_password_hash

def layout():
    return dbc.Container([
        html.Br(),
        dbc.Container([
            dcc.Location(id='urlLogin', refresh=True),
            dbc.Form([
                dbc.FormGroup([
                    dbc.Label("Username", className="mr-2"),
                    dbc.Input(type="text", id='usernameBox', placeholder="Enter your username", n_submit=0, className='form-control'),
                ], className="mr-3"),
                dbc.FormGroup([
                    dbc.Label("Password", className="mr-2"),
                    dbc.Input(type="password", id='passwordBox', placeholder="Enter your password", n_submit=0, className='form-control'),
                ], className="mr-3"),
                dbc.Button(html.B("Login", style={'padding-left': 20, 'padding-right': 20, 'font-size': 'large'}), id='loginButton', color="primary", n_clicks=0, className='mr-2')
            ], inline=False, className='form-group')
        ], className='jumbotron', style={'background-color': '#f8f9fa'}),
    ])

################################################################################
# LOGIN BUTTON CLICKED / ENTER PRESSED - REDIRECT TO PAGE1 IF LOGIN DETAILS ARE CORRECT
################################################################################
@app.callback(Output('urlLogin', 'pathname'),
              [Input('loginButton', 'n_clicks'),
              Input('usernameBox', 'n_submit'),
              Input('passwordBox', 'n_submit')],
              [State('usernameBox', 'value'),
               State('passwordBox', 'value')])
def sucess(n_clicks, usernameSubmit, passwordSubmit, username, password):
    user = User.query.filter_by(username=username).first()
    if user:
        if check_password_hash(user.password, password):
            login_user(user)
            return '/'
        else:
            pass
    else:
        pass


################################################################################
# LOGIN BUTTON CLICKED / ENTER PRESSED - RETURN RED BOXES IF LOGIN DETAILS INCORRECT
################################################################################
@app.callback(Output('usernameBox', 'className'),
              [Input('loginButton', 'n_clicks'),
              Input('usernameBox', 'n_submit'),
              Input('passwordBox', 'n_submit')],
              [State('usernameBox', 'value'),
               State('passwordBox', 'value')])
def update_output(n_clicks, usernameSubmit, passwordSubmit, username, password):
    if (n_clicks > 0) or (usernameSubmit > 0) or (passwordSubmit) > 0:
        user = User.query.filter_by(username=username).first()
        if user:
            if check_password_hash(user.password, password):
                return 'form-control'
            else:
                return 'form-control is-invalid'
        else:
            return 'form-control is-invalid'
    else:
        return 'form-control'


################################################################################
# LOGIN BUTTON CLICKED / ENTER PRESSED - RETURN RED BOXES IF LOGIN DETAILS INCORRECT
################################################################################
@app.callback(Output('passwordBox', 'className'),
              [Input('loginButton', 'n_clicks'),
              Input('usernameBox', 'n_submit'),
              Input('passwordBox', 'n_submit')],
              [State('usernameBox', 'value'),
               State('passwordBox', 'value')])
def update_output(n_clicks, usernameSubmit, passwordSubmit, username, password):
    if (n_clicks > 0) or (usernameSubmit > 0) or (passwordSubmit) > 0:
        user = User.query.filter_by(username=username).first()
        if user:
            if check_password_hash(user.password, password):
                return 'form-control'
            else:
                return 'form-control is-invalid'
        else:
            return 'form-control is-invalid'
    else:
        return 'form-control'