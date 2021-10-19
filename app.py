import warnings

import dash_table as dt
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
from flask_login import logout_user, current_user

from src.utils import CONTENT_STYLE, performance,eda, get_data, TABLE_STYLE, engine, keywords, go_get_it, home
from pages import scraping, explore, report,login
from server import app, server

DEV = False

warnings.filterwarnings("ignore")

app.layout = dbc.Container([
    dcc.Location(id='url'),
    dcc.Loading(html.Div(id='content'),type="default",)
], style=CONTENT_STYLE)

@app.callback(
    Output('content','children'),
    [
        Input('url','pathname'),
    ]
)
def pages(pathname):
    if DEV:
        if pathname == '/scraping':
            keys = pd.read_sql_query("select * from keywords", con=engine).values.reshape(1,-1)[0]
            keys = [{'label': n, 'value': n} for n in keys]
            return scraping.layout(keys,keywords())
        elif pathname.startswith("/addkeys"):
            newk = [n.replace("%20"," ") for n in pathname.split("+")[1].split(",")]
            keys = pd.read_sql_query("select * from keywords", con=engine).values.reshape(1,-1)[0]
            keys = sorted(set(keys.tolist() + newk))
            pd.DataFrame(keys,columns=["keywords"]).to_sql("keywords",con=engine,if_exists="replace",index=False)
            keys = [{'label': n, 'value': n} for n in keys]
            return scraping.layout(keys,keywords())
        elif pathname.startswith("/delkeys"):
            delk = [n.replace("%20"," ") for n in pathname.split("+")[1].split(",")]
            keys = pd.read_sql_query("select * from keywords", con=engine).values.reshape(1,-1)[0]
            keys = sorted(set([n for n in keys if n not in delk]))
            pd.DataFrame(keys,columns=["keywords"]).to_sql("keywords",con=engine,if_exists="replace",index=False)
            keys = [{'label': n, 'value': n} for n in keys]
            return scraping.layout(keys,keywords())
        elif pathname.startswith('/crawling'):
            ncrawl = int([n.replace("%20","") for n in pathname.split("+")[1].split(",")][0])
            go_get_it(ncrawl)
            keys = pd.read_sql_query("select * from keywords", con=engine).values.reshape(1,-1)[0]
            keys = [{'label': n, 'value': n} for n in keys]
            return scraping.layout(keys,keywords(),display="inline")
        elif pathname == '/explore':
            children = eda()
            return explore.layout(children)
        elif pathname == '/report':
            children = performance()
            return report.layout(children)
        elif pathname == '/logout':
            return login.layout()
        else:
            return home 
    else:    
        if pathname == '/scraping' or pathname == '/main':
            if current_user.is_authenticated:
                keys = pd.read_sql_query("select * from keywords", con=engine).values.reshape(1,-1)[0]
                keys = [{'label': n, 'value': n} for n in keys]
                return scraping.layout(keys,keywords())
            else:
                return login.layout()
        elif pathname.startswith("/addkeys"):
            if current_user.is_authenticated:
                newk = [n.replace("%20"," ") for n in pathname.split("+")[1].split(",")]
                keys = pd.read_sql_query("select * from keywords", con=engine).values.reshape(1,-1)[0]
                keys = sorted(set(keys.tolist() + newk))
                pd.DataFrame(keys,columns=["keywords"]).to_sql("keywords",con=engine,if_exists="replace",index=False)
                keys = [{'label': n, 'value': n} for n in keys]
                return scraping.layout(keys,keywords())
            else:
                return login.layout()
        elif pathname.startswith("/delkeys"):
            if current_user.is_authenticated:
                delk = [n.replace("%20"," ") for n in pathname.split("+")[1].split(",")]
                keys = pd.read_sql_query("select * from keywords", con=engine).values.reshape(1,-1)[0]
                keys = sorted(set([n for n in keys if n not in delk]))
                pd.DataFrame(keys,columns=["keywords"]).to_sql("keywords",con=engine,if_exists="replace",index=False)
                keys = [{'label': n, 'value': n} for n in keys]
                return scraping.layout(keys,keywords())
            else:
                return login.layout()
        elif pathname.startswith('/crawling'):
            if current_user.is_authenticated:
                ncrawl = int([n.replace("%20","") for n in pathname.split("+")[1].split(",")][0])
                go_get_it(ncrawl)
                keys = pd.read_sql_query("select * from keywords", con=engine).values.reshape(1,-1)[0]
                keys = [{'label': n, 'value': n} for n in keys]
                return scraping.layout(keys,keywords(),display="inline")
            else:
                return login.layout()
        elif pathname == '/explore':
            if current_user.is_authenticated:
                children = eda()
                return explore.layout(children)
            else:
                return login.layout()
        elif pathname == '/report':
            if current_user.is_authenticated:
                children = performance()
                return report.layout(children)
            else:
                return login.layout()
        elif pathname == '/logout':
            if current_user.is_authenticated:
                logout_user()
                return login.layout()
            else:
                return login.layout()
        else:
            return home

@app.callback(
    [
        Output("data","children"),
        Output("data-footer","children")
    ],
    Input("tabs","active_tab")
)
def data_render(tab):
    df = get_data()
    if tab == "tab-2":
        df = df[df.columns[[0,1,3,-2,-1]]]
        return dt.DataTable(
            data=df.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df.columns],
            **TABLE_STYLE
        ), "The data has been cleaned and case folded, by removing unused strings and punctuation."
    elif tab == "tab-3":
        df = df[df.columns[[0,1,4,-2,-1]]]
        return dt.DataTable(
            data=df.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df.columns],
            **TABLE_STYLE
        ), "The text has been tokenized using NLTK library and slang words has been substitued."
    elif tab == "tab-4":
        df = df[df.columns[[0,1,5,-2,-1]]]
        return dt.DataTable(
            data=df.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df.columns],
            **TABLE_STYLE
        ), "The stopwords and emojis were removed from the texts."
    elif tab == "tab-5":
        df = df[df.columns[[0,1,6,-2,-1]]]
        return dt.DataTable(
            data=df.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df.columns],
            **TABLE_STYLE
        ), "Stemming in Indonesian language using Nazief-and-Adriani-algorithm."
    else:
        df = df[df.columns[[0,1,2,-2,-1]]]
        return dt.DataTable(
            data=df.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df.columns],
            **TABLE_STYLE
        ), "The results of scraping using twitter API and tweepy library."
@app.callback(
    Output("del-button","disabled"),
    Input("del-field","value")
)
def del_disabled(val):
    if val is not None and val != []:
        return False
    else:
        return True

@app.callback(
    Output("url","href"),
    [
        Input("add-field","value"),
        Input("add-button","n_clicks"),
        Input("del-field","value"),
        Input("del-button","n_clicks"),
        Input("crawl-config","value"),
        Input("crawl-button","n_clicks"),
    ]
)
def add_keys(add,nadd,delete,ndel,cn,cb):
    if add is not None and add != [] and nadd is not None:
        return f"/addkeys+{add}"
    elif delete is not None and delete != [] and ndel is not None:
        delete = ", ".join(delete)
        return f"/delkeys+{delete}"
    elif cn is not None and cb is not None:
        return f"/crawling+{cn}"
    

if __name__ == '__main__':
    app.run_server(debug=True)