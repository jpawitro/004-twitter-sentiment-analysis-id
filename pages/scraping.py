import warnings
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Card import Card
from dash_bootstrap_components._components.CardBody import CardBody
from dash_bootstrap_components._components.CardHeader import CardHeader
from src.utils import sidebar,nav, title

warnings.filterwarnings("ignore")

def layout(keys,keywords, display="none"):
    return dbc.Container([
        # sidebar,
        nav,
        # dbc.Row(html.H2(html.B("Scraping")),justify="center"),
        title,
        html.Br(),
        html.Div(html.H4('Data')),
        dbc.Card(
            [
                dbc.CardHeader([
                    # html.H4('Data',className="card-title"),
                    dbc.Tabs(
                        [
                            dbc.Tab(label="Crawling", tab_id="tab-1"),
                            dbc.Tab(label="Cleaning & Casefolding", tab_id="tab-2"),
                            dbc.Tab(label="Tokenizing", tab_id="tab-3"),
                            dbc.Tab(label="Stopwords Removal", tab_id="tab-4"),
                            dbc.Tab(label="Stemming", tab_id="tab-5"),
                        ],
                        id="tabs",
                        active_tab="tab-1",
                    ),
                ]),
                dbc.CardBody(
                    [
                        dbc.Row(html.H5(html.I("Crawling Done"), style={"display":display}),justify="center"),
                        # html.H4("Card title", className="card-title"),
                        # html.P("This is some card text", className="card-text"),
                        html.Div(id="data"),
                    ]
                ),
                dbc.CardFooter(id="data-footer"),
            ],
            # style={"width": "18rem"},
        ),
        # html.Div(html.H4('Data')),
        # dbc.Row(html.H5(html.I("Crawling Done"), style={"display":display}),justify="center"),
        # html.Div(
        #     [
        #         dbc.Tabs(
        #             [
        #                 dbc.Tab(label="Crawling", tab_id="tab-1"),
        #                 dbc.Tab(label="Cleaning & Casefolding", tab_id="tab-2"),
        #                 dbc.Tab(label="Tokenizing & Slang Words Substitution", tab_id="tab-3"),
        #                 dbc.Tab(label="Stopwords & Emoji Removal", tab_id="tab-4"),
        #                 dbc.Tab(label="Stemming", tab_id="tab-5"),
        #             ],
        #             id="tabs",
        #             active_tab="tab-1",
        #         ),
        #         html.Div(id="data"),
        #     ]
        # ),
        html.Br(),
        html.Br(),
        html.Div(html.H4('Keywords')),
        dbc.Card([
            dbc.CardHeader(html.H5('Existing')),
            dbc.CardBody([
                dbc.Table.from_dataframe(keywords, dark=False, striped=True, bordered=True, hover=True),
            ])
        ]),
        html.Br(),
        dbc.Card([
            dbc.CardHeader(html.H5('Modify')),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(
                        dbc.Form(
                            [
                                dbc.FormGroup(
                                    [
                                        dbc.Label("Add new Keywords", className="mr-2"),
                                        dbc.Input(type="text", id="add-field", placeholder="with commas"),
                                    ],
                                    className="mr-3",
                                ),
                                dbc.Button("Submit", id="add-button", color="primary"),
                            ],
                            inline=True,
                        ),
                    md = 6),
                    dbc.Col([
                        dbc.Row([
                            dbc.Label("Delete Keywords", className="mr-2"),
                            dcc.Dropdown(className="mr-3", id="del-field", options=keys, style={'width':'60%','color':'black'},multi=True),
                            dbc.Button("Submit", id="del-button", color="primary"),
                        ])
                    ], style={"display":"inline-block"}, md = 6)
                ]),
            ])
        ]),
        # html.Div(html.H5('Existing')),
        # dbc.Table.from_dataframe(keywords, dark=True, striped=True, bordered=True, hover=True),
        # html.Div(html.H5('Modify')),
        # dbc.Row([
        #     dbc.Col(
        #         dbc.Form(
        #             [
        #                 dbc.FormGroup(
        #                     [
        #                         dbc.Label("Add new Keywords", className="mr-2"),
        #                         dbc.Input(type="text", id="add-field", placeholder="with commas"),
        #                     ],
        #                     className="mr-3",
        #                 ),
        #                 dbc.Button("Submit", id="add-button", color="primary"),
        #             ],
        #             inline=True,
        #         ),
        #     md = 6),
        #     dbc.Col([
        #         dbc.Row([
        #             dbc.Label("Delete Keywords", className="mr-2"),
        #             dcc.Dropdown(className="mr-3", id="del-field", options=keys, style={'width':'60%','color':'black'},multi=True),
        #             dbc.Button("Submit", id="del-button", color="primary"),
        #         ])
        #     ], style={"display":"inline-block"}, md = 6)
        # ]),
        html.Br(),
        html.Br(),
        dbc.Card(
            dbc.Row([
                dbc.Col(html.H4('Crawl New Data'), md=6),
                dbc.Col(
                    dbc.Row(
                        dbc.Form(
                            [
                                dbc.FormGroup(
                                    [
                                        dbc.Label("Number of Data", className="mr-2"),
                                        dbc.Input(type="number", id="crawl-config", placeholder="number of data",value=1000),
                                    ],
                                    className="mr-3",
                                ),
                                dbc.Button("Crawl", id="crawl-button", color="primary"),
                            ], inline=True,
                        )
                    , justify="center")
                ,md=6)
            ], style={'padding':'2rem'})
        ),
        # html.Div(html.H4('Crawl New Data')),
        # html.Div(html.H5('Configuration')),
        # dbc.Row(
        #     dbc.Form(
        #         [
        #             dbc.FormGroup(
        #                 [
        #                     dbc.Label("Number of Data", className="mr-2"),
        #                     dbc.Input(type="number", id="crawl-config", placeholder="number of data",value=1000),
        #                 ],
        #                 className="mr-3",
        #             ),
        #             dbc.Button("Crawl", id="crawl-button", color="primary"),
        #         ], inline=True,
        #     )
        # , justify="center")
    ])