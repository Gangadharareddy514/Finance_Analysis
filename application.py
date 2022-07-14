#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask,render_template,request
from urllib.request import urlopen, Request
import pandas as pd
import matplotlib.pyplot as plt
import torch
import requests
from bs4 import BeautifulSoup
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yfinance as yf
from datetime import datetime,timedelta
import plotly
import plotly.express as px
import json
import plotly.graph_objects as go



application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/sentiment',methods=['POST'])
def sentiment():
    company = str(request.form.get('company'))
    finviz_url = 'https://finviz.com/quote.ashx?t='
    tickers = company.split(" ")
    
    news_tables = {}
    for ticker in tickers:
        url = finviz_url + ticker
        req = Request(url=url, headers={'user-agent': 'my-app'})
        response = urlopen(req)
    
        html = BeautifulSoup(response, features='html.parser')
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table
        
        parsed_data = []

    for ticker, news_table in news_tables.items():
    
        for row in news_table.findAll('tr'):
    
            title = row.a.text
            date_data = row.td.text.split(' ')
    
            if len(date_data) == 1:
                time = date_data[0]
            else:
                date = date_data[0]
                time = date_data[1]
    
            parsed_data.append([ticker, date, time, title])
    
    df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
    
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    
    def sentiment_score(txt):
        tokens = tokenizer.encode(txt, return_tensors='pt')
        result = model(tokens)
        return int(torch.argmax(result.logits))+1
    
    df['sentiment'] = df['title'].apply(lambda x: sentiment_score(x[:512]))
    
    d = datetime.today() - timedelta(days=14)
    
    tickers_list = tickers
    
    data_close = yf.download(tickers_list,d)['Adj Close']
    data_open = yf.download(tickers_list,d)['Open']
    
    
    if len(tickers_list)==1:
        data_close=data_close.reset_index()
        data_close["Company"]=tickers_list[0]
        
        data_open=data_open.reset_index()
        data_open["Company"]=tickers_list[0]
        
        data1=data_close
        data2=data_open
    
    else:
    
        data_close=data_close.reset_index()
        
        data_open=data_open.reset_index()
        
        data1 = pd.melt(data_open, id_vars='Date', value_vars=tickers_list)
        
        data1.rename(columns = {'variable': 'Company','value':'Adj Close'}, inplace = True)
        
        data2 = pd.melt(data_close, id_vars='Date', value_vars=tickers_list)
        
        data2.rename(columns = {'variable': 'Company','value':'Open'}, inplace = True)
        
    
    merged_inner = pd.merge(left=data2, right=data1, left_on=["Date","Company"], right_on=["Date","Company"])
        
    df1=df.sort_values('date',ascending=False).groupby(['ticker', 'date']).mean().reset_index()
    
    
    
    df1["date"]=df1["date"].apply(lambda x: datetime.strptime(x,'%b-%d-%y').strftime("%Y-%m-%d"))
    
    df1['date'] = pd.to_datetime(df1['date'])
    
    merged_inner_df = pd.merge(left=df1, right=merged_inner, left_on=["date","ticker"], right_on=["Date","Company"])
    
    merged_inner_df.drop(columns=["ticker","date"], inplace=True)
    
    
    fig_daily = px.bar(merged_inner_df, x="Date", y="sentiment", title = ticker + ' Mean Daily Sentiment Scores')
    fig_daily.update_yaxes(range=[0, 5])
    
    graphJSON_daily = json.dumps(fig_daily, cls=plotly.utils.PlotlyJSONEncoder)
    

    
    fig_stockprices = go.Figure()
    fig_stockprices.add_trace(go.Scatter(x=merged_inner_df['Date'], y=merged_inner_df['Open'],
                         mode='lines+markers',
                        name='Open'))
    
    fig_stockprices.add_trace(go.Scatter(x=merged_inner_df['Date'], y=merged_inner_df['Adj Close'],
                        mode='lines+markers',
                        name='Adj Close'))
    
    fig_stockprices.update_layout(title= ticker + ' Stock Prices',
                   xaxis_title='Date',
                   yaxis_title='Stock Price')
    

    
    graphJSON_stockprice = json.dumps(fig_stockprices, cls=plotly.utils.PlotlyJSONEncoder)
    
    header= "Daily Sentiment of {} Stock".format(ticker)
	
    description = """
	The above chart averages the sentiment scores of {} stock daily.
	The table below gives each of the most recent headlines of the stock and the sentiment score.
	The news headlines are obtained from the FinViz website.
	Sentiments are given by the BERT Transformer.
    """.format(ticker)
    


    merged_inner_df=merged_inner_df[["Company","Date","Open","Adj Close","sentiment"]]
    
    return render_template('sentiment.html',graphJSON_daily=graphJSON_daily, graphJSON_stockprice=graphJSON_stockprice, header=header,table=merged_inner_df.to_html(classes='data'),description=description)


if __name__ == '__main__':
    application.run(debug=True)