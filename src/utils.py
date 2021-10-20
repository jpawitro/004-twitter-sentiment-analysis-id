import os
import json
import warnings
import pickle
from dash_bootstrap_components._components.Card import Card
from dash_bootstrap_components._components.CardBody import CardBody
import numpy as np
import ast
import re
import string

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import tweepy
import preprocessor as p
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize

from sklearn.metrics import classification_report,confusion_matrix#, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE

import plotly.express as px
import plotly.figure_factory as ff

from src.config import engine

pd.options.mode.chained_assignment = None
nltk.download('stopwords')
factory = StemmerFactory()
stemmer = factory.create_stemmer()
listStopword =  set(stopwords.words('indonesian'))
update = True  # Configuration for retrain only, without scraping. True to crawl, False for retrain only.

warnings.filterwarnings("ignore")

consumerKey = "CONSUMER KEY"
consumerSecret = "CONSUMER SECRET"
accessToken = "ACCESS TOKEN"
accessTokenSecret = "ACCESS TOKEN SECRET"

template = "plotly_dark"

auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

file  = open(os.path.join("data","cleaning_source","update_combined_slang_words.txt"), "r")
content = file.read()
slang_words = ast.literal_eval(content)

TABLE_STYLE = dict(
    style_cell={
        'backgroundColor': 'rgb(100, 100, 100)',
        'color': 'white',
        'font-family': "Lato",
        'textAlign': 'left'
    },
    style_table={'overflowX': 'auto'},
    page_size=15,
    style_header={
        'backgroundColor': 'black',
        'fontWeight': 'bold',
    },
    editable=True,
    filter_action="native",
    sort_action="native",
    sort_mode="multi",
    column_selectable="single",
    row_selectable="multi",
    selected_columns=[],
    selected_rows=[],
    page_action="native",
    page_current= 0,
)

CONTENT_STYLE = {
    "padding": "2rem 1rem",
}

LOGO = "https://upload.wikimedia.org/wikipedia/en/6/60/Twitter_Logo_as_of_2021.svg"

emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])

emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

emoticons = emoticons_happy.union(emoticons_sad)

listtoberemoved = ["halilintar"]

negasi = ['bukan','tidak','ga','gk']
lexicon = pd.read_csv(os.path.join('data','lexicon','modified_full_lexicon.csv'))
lexicon = lexicon.drop(lexicon[(lexicon['word'] == 'bukan')
                               |(lexicon['word'] == 'tidak')
                               |(lexicon['word'] == 'ga')|(lexicon['word'] == 'gk') ].index,axis=0)
lexicon = lexicon.reset_index(drop=True)
lexicon_word = lexicon['word'].to_list()

nav = dbc.Navbar([
    html.A(
        # Use row and col to control vertical alignment of logo / brand
        dbc.Row(
            [
                dbc.Col(html.Img(src=LOGO, height="30px")),
            ],
            align="center",
            no_gutters=True,
        ),
        href="https://landak-tech.github.io/",
    ),
    dbc.Nav(
        [
            dbc.NavItem(dbc.NavLink("Home", href="/", active="exact")),
            dbc.NavItem(dbc.NavLink("Scraping", href="/scraping", active="exact")),
            dbc.NavItem(dbc.NavLink("Exploratory Analysis", href="/explore", active="exact")),
            dbc.NavItem(dbc.NavLink("Model Performance", href="/report", active="exact")),
            dbc.NavItem(dbc.NavLink("Logout", href="/logout", active="exact")),
        ],
        pills=True,
    ),
],color="dark", dark=True)

home = [
    dbc.Jumbotron(
        [
            html.H1("Sentiment Analysis on Tweets", className="display-3"),
            html.P(
                "Using Naive Bayes Algorithm and "
                "utilizing GridSearchCV for hyperparameter optimization to find the best paramaters.",
                className="lead",
            ),
            html.Hr(className="my-2"),
            html.P(
                "Built using plotly, dash, and sklearn "
                "on top of flask micro-framework."
            ),
            html.P(dbc.Button("Begin", color="primary", href='/scraping'), className="lead"),
        ], id='welcome'
    , style={'display':'block'})
]

title = html.Div([
    dbc.Row(html.H2(html.B("Twitter Sentiment Analysis & Classification")),justify="center"),
    dbc.Row(html.H4(html.I("Using Naive-Bayes Algorithm")),justify="center")
])

def keywords():
    keysd = pd.read_sql_query("select * from crawling", con=engine)
    keys = pd.read_sql_query("select * from keywords", con=engine)
    df = keys.join(keysd.groupby("keywords").agg({
        "username":"nunique",
        "full_text":"count"
    }),on="keywords").fillna(0)
    return df

def get_data():
    df = pd.read_sql_query("select * from data", con=engine)
    return df

def get_sentiment():
    df = pd.read_sql_query("select * from sentiment", con=engine)
    return df

def performance():
    vectorizer = pickle.load(open(os.path.join('model','vectorizer.sav'),'rb'))
    tfidfconverter = pickle.load(open(os.path.join('model','tfidfconverter.sav'),'rb'))
    le = pickle.load(open(os.path.join('model','le.sav'),'rb'))
    sm = pickle.load(open(os.path.join('model','smote.sav'),'rb'))
    gs_NB = pickle.load(open(os.path.join('model','clf.sav'),'rb'))
    df = get_data()
    stemmed = [word_tokenize(n) for n in df['stemmed'].values]
    X = vectorizer.transform([" ".join(x) for x in stemmed]).toarray()
    X = tfidfconverter.transform(X).toarray()
    y = le.transform(df['label'])
    Xres,yres = sm.fit_resample(X,y)
    X_train, X_test, y_train, y_test = train_test_split(Xres, yres, test_size = 0.2, random_state=42, shuffle=True)
    y_pred = gs_NB.predict(X_train)
    y_pred_ = gs_NB.predict(X_test)
    report1 = classification_report(y_train,y_pred, target_names=le.classes_, output_dict=True)
    report1 = pd.DataFrame(report1).transpose()
    cf1 = confusion_matrix(y_train, y_pred)
    cf1 = px.imshow(cf1, x=le.classes_, y=le.classes_, color_continuous_scale=px.colors.diverging.Spectral)
    cf1.update_layout(
        title="Confusion Matrix Training Dataset",
        hovermode="x",
        template=template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    report2 = classification_report(y_test,y_pred_, target_names=le.classes_, output_dict=True)
    report2 = pd.DataFrame(report2).transpose()
    cf2 = confusion_matrix(y_test, y_pred_)
    cf2 = px.imshow(cf2, x=le.classes_, y=le.classes_, color_continuous_scale=px.colors.diverging.Spectral)
    cf2.update_layout(
        title="Confusion Matrix Test Dataset",
        hovermode="x",
        template=template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    best_params = json.loads(open(os.path.join("data","best_params.json")).read())
    return [
        html.Br(),
        dbc.Card(
            dbc.Row([
                dbc.Col(html.H4('Best Parameters'), md=6),
                dbc.Col(
                    dbc.Row(
                        html.P(str(best_params))
                    , justify="center")
                ,md=6)
            ], style={'padding':'2rem'})
        ),
        html.Br(),
        dbc.Row(html.H4(html.B("Training Classification Report"))),
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dbc.Table.from_dataframe(report1.reset_index(), dark=False, striped=True, bordered=True, hover=True, responsive=True)
                    )
                ),
            )
        ),
        html.Br(),
        dbc. Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dcc.Graph(
                            figure=cf1
                        )
                    )
                ),
            )
        ),
        html.Br(),
        html.Br(),
        dbc.Row(html.H4(html.B("Test Classification Report"))),
        dbc. Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dbc.Table.from_dataframe(report2.reset_index(), dark=False, striped=True, bordered=True, hover=True, responsive=True)
                    )
                )
            )
        ),
        html.Br(),
        dbc. Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dcc.Graph(
                            figure=cf2
                        )
                    )
                )
            )
        )
    ]

def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=10):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr

def top_ten_corr(au,df_sen):
    top10 = au[au<float(1)][0:10]
    label = top10.index
    label_list =[]
    for i in label:
        for j in i:
            if(j not in label_list):
                label_list.append(j)
                
    df_sen_corr = df_sen[label_list]
    corr = df_sen_corr.corr()
    for i in label_list:
        for j in label_list:
            if i!=j:
                corr[i][j] = round(corr[i][j],3)
    return corr

def eda():
    df = get_data()
    df_sen = get_sentiment()
    au = get_top_abs_correlations(df_sen, 15)
    corr = top_ten_corr(au,df_sen)
    fig1 = px.histogram(df['sentiment'],x='sentiment',marginal='box', color_discrete_sequence=['indianred'])
    fig1.update_layout(
        title="Sentiment Score Distribution",
        hovermode="x",
        template=template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig2 = px.pie(df.groupby('label').size().to_frame('count').reset_index(),names='label',values='count', color_discrete_sequence=px.colors.sequential.Darkmint)
    fig2.update_layout(
        title="Sentiment Label Composition",
        hovermode="x",
        template=template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig3 = ff.create_annotated_heatmap(corr.values.tolist(), x=corr.columns.values.tolist(), y=corr.index.values.tolist(), colorscale="Earth")
    fig3.update_layout(
        title="Top 10 Correlation",
        hovermode="x",
        template=template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    top15 = au[au<float(1)][0:15]
    fig4 = px.bar(top15, x=top15, y=[str(n) for n in top15.index])
    fig4.update_layout(
        title="Top 15 Not-perfect Correlation",
        hovermode="x",
        template=template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis = dict(autorange="reversed",title="Word Pairs"),
        xaxis = dict(
            tickformat=".0%",
            title="Correlation"
        )
    )
    top15_word = df_sen.drop(['sentiment'],axis=1).sum().sort_values(ascending=False)[0:15]
    fig5 = px.bar(top15_word, x=top15_word, y=top15_word.index)
    fig5.update_layout(
        title="Top 15 Most Often Occured Words",
        hovermode="x",
        template=template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis = dict(autorange="reversed",title="Words"),
        xaxis = dict(title="Count")
    )
    return [
        html.Br(),
        dbc.Row(html.H4(html.B("Sentiment Distribution"))),
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dcc.Graph(
                            figure=fig1
                        ),
                    )
                ), md = 6
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dcc.Graph(
                            figure=fig2
                        )       
                    )
                ), md = 6
            )
        ]),
        html.Br(),
        html.Br(),
        dbc.Row(html.H4(html.B("Features Correlation"))),
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dcc.Graph(
                            figure=fig3,
                            style={
                                "height":"100%",
                                "width":"100%",
                            }
                        )
                    )
                ), md = 12
            )
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dcc.Graph(
                            figure=fig4,
                            style={
                                "height":"100%",
                                "width":"100%",
                            }
                        )
                    )
                ), md = 12
            )
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dcc.Graph(
                            figure=fig5,
                            style={
                                "height":"100%",
                                "width":"100%",
                            }
                        )
                    )
                ), md = 12
            )
        ]),
    ]

def crawling(keyword, noOfTweet):
    tweets = tweepy.Cursor(api.search, q=keyword, lang='id', tweet_mode='extended').items(noOfTweet)
    tweet_list = []

    for tweet in tweets:
        if 'retweeted_status' in tweet._json:
            tweet_text = [tweet._json['retweeted_status']['created_at'],tweet._json['retweeted_status']['user']['screen_name'],tweet._json['retweeted_status']['full_text']]
        else:
            tweet_text = [tweet.created_at,tweet.user.screen_name,tweet.full_text]
        tweet_list.append(tweet_text)
    tweet_list = pd.DataFrame(tweet_list, columns=['created_at','username','full_text'])
    tweet_list['created_at'] = pd.to_datetime(tweet_list['created_at'])
    tweet_list.drop_duplicates(inplace = True)
    return tweet_list

def found_word(ind,words,word,sen,sencol,sentiment,add):
    if word in sencol:
        sen[sencol.index(word)] += 1
    else:
        sencol.append(word)
        sen.append(1)
        add += 1
    if (words[ind-1] in negasi):
        sentiment += -lexicon['weight'][lexicon_word.index(word)]
    else:
        sentiment += lexicon['weight'][lexicon_word.index(word)]
    
    return sen,sencol,sentiment,add

def go_get_it(ncrawl):
    keys = pd.read_sql_query("select * from keywords", con=engine).values.reshape(1,-1)[0].tolist()
    if update:
        df = pd.DataFrame()
        for key in keys:
            n_list = crawling(key,ncrawl)
            df = pd.concat([df,n_list])
        df.drop_duplicates(inplace = True)
        df['created_at'] = df['created_at'].astype(str).str.split('+',expand=True).loc[:,0]
        df.reset_index(drop=True,inplace=True)
        df.to_sql("crawling",con=engine,if_exists="replace",index=False)
    else:
        df = pd.read_sql_query("select * from crawling", con=engine)

    df_cleaned = df.iloc[:,:2].copy()
    cleaned = []
    for n in df['full_text'].values:
        n = p.clean(n)
        n = n.lower()
        n = re.sub(r':', '', n)
        n = re.sub(r'‚Ä¶', '', n)
        n = re.sub(r'[^\x00-\x7F]+',' ', n)
        n = emoji_pattern.sub(r'', n)
        n = re.sub('[^a-zA-Z]', ' ', n)
        n = re.sub("&lt;/?.*?&gt;","&lt;&gt;",n)
        n = re.sub("(\\d|\\W)+"," ",n)
        n = re.sub(r'â', '', n)
        n = re.sub(r'€', '', n)
        n = re.sub(r'¦', '', n)
        cleaned.append(n)
    df['cleaned'] = cleaned

    tokenized = []
    df_tokenized = df.iloc[:,:2].copy()
    for n in cleaned:
        n = word_tokenize(n)
        for w in n:
            if w in slang_words.keys():
                n[n.index(w)] = slang_words[w]
        tokenized.append(n)
    df['tokenized'] = [', '.join(n) for n in tokenized]

    removed = []
    for ts in tokenized:
        n = []
        for t in ts:
            if t not in keys+listtoberemoved and t not in listStopword and t not in emoticons and t not in string.punctuation:
                n.append(t)
        removed.append(n)
    df['removed'] = [', '.join(n) for n in removed]

    stemmed = []
    for n in removed:
        n = ' '.join(n)
        n = stemmer.stem(n)
        n = n.split(' ')
        stemmed.append(n)
    df['stemmed'] = [' '.join(n) for n in stemmed]

    df.to_csv(os.path.join('data','prep.csv'),index=False)
    df.to_excel(os.path.join('data','prep.xlsx'),index=False)

    sencol =[]
    senrow =np.array([])
    nsen = 0
    sentiment_list = []
                
    for i in range(len(stemmed)):
        nsen = senrow.shape[0]
        sentiment = 0 
        words = stemmed[i].copy()
        add = 0
        prev = [0 for ii in range(len(words))]
        n_words = len(words)
        if len(sencol)>0:
            sen =[0 for j in range(len(sencol))]
        else:
            sen =[]
        
        for word in words:
            ind = words.index(word)
            if word in lexicon_word :
                sen,sencol,sentiment,add= found_word(ind,words,word,sen,sencol,sentiment,add)
            else:
                kata_dasar = stemmer.stem(word)
                if kata_dasar in lexicon_word:
                    sen,sencol,sentiment,add= found_word(ind,words,kata_dasar,sen,sencol,sentiment,add)
                elif(n_words>1):
                    if ind-1>-1:
                        back_1    = words[ind-1]+' '+word
                        if (back_1 in lexicon_word):
                            sen,sencol,sentiment,add= found_word(ind,words,back_1,sen,sencol,sentiment,add)
                        elif(ind-2>-1):
                            back_2    = words[ind-2]+' '+back_1
                            if back_2 in lexicon_word:
                                sen,sencol,sentiment,add= found_word(ind,words,back_2,sen,sencol,sentiment,add)
        if add>0:  
            if i>0:
                if (nsen==0):
                    senrow = np.zeros([i,add],dtype=int)
                elif(i!=nsen):
                    padding_h = np.zeros([nsen,add],dtype=int)
                    senrow = np.hstack((senrow,padding_h))
                    padding_v = np.zeros([(i-nsen),senrow.shape[1]],dtype=int)
                    senrow = np.vstack((senrow,padding_v))
                else:
                    padding =np.zeros([nsen,add],dtype=int)
                    senrow = np.hstack((senrow,padding))
                senrow = np.vstack((senrow,sen))
            if i==0:
                senrow = np.array(sen).reshape(1,len(sen))
        elif(nsen>0):
            senrow = np.vstack((senrow,sen))
            
        sentiment_list.append(sentiment)

    sencol.append('sentiment')
    sentiment_array = np.array(sentiment_list).reshape(senrow.shape[0],1)
    sentiment_data = np.hstack((senrow,sentiment_array))
    df_sen = pd.DataFrame(sentiment_data,columns = sencol)
    df['sentiment'] = df_sen['sentiment']

    df.loc[df['sentiment'] == 0, 'label'] = 'neutral'
    df.loc[df['sentiment'] > 0, 'label'] = 'positive'
    df.loc[df['sentiment'] < 0, 'label'] = 'negative'

    vectorizer = CountVectorizer(max_features=1500, min_df=2, max_df=0.8)
    X = vectorizer.fit_transform([" ".join(x) for x in stemmed]).toarray()

    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()

    le = LabelEncoder()
    y = le.fit_transform(df['label'])

    sm = SMOTE()
    Xres,yres = sm.fit_resample(X,y)

    X_train, X_test, y_train, y_test = train_test_split(Xres, yres, test_size = 0.2, random_state=42, shuffle=True)

    nb_classifier = GaussianNB()

    params_NB = {'var_smoothing': np.logspace(0,-9, num=1000)}
    gs_NB = GridSearchCV(estimator=nb_classifier, 
                    param_grid=params_NB, 
                    cv=10,
                    verbose=1, 
                    scoring='accuracy') 
    gs_NB.fit(X_train, y_train)

    pickle.dump(vectorizer,open(os.path.join('model','vectorizer.sav'),'wb'))
    pickle.dump(tfidfconverter,open(os.path.join('model','tfidfconverter.sav'),'wb'))
    pickle.dump(le,open(os.path.join('model','le.sav'),'wb'))
    pickle.dump(sm,open(os.path.join('model','smote.sav'),'wb'))
    pickle.dump(gs_NB,open(os.path.join('model','clf.sav'),'wb'))

    df.to_excel(os.path.join("data","data_sentiment.xlsx"),index=False)
    df.to_sql("data",con=engine,if_exists="replace",index=False)

    df_sen.to_excel(os.path.join("data","scoring_sentiment.xlsx"),index=False)
    df_sen.to_sql("sentiment",con=engine,if_exists="replace",index=False)

    json.dump(gs_NB.best_params_,open(os.path.join('data','best_params.json'),'w'))