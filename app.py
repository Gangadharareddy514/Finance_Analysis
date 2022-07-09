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


#model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_placement():
    company = str(request.form.get('company'))
    finviz_url = 'https://finviz.com/quote.ashx?t='
    tickers = ["AMZN","GOOG"]
    
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
    
    tickers_list = ["GOOG","AMZN"]
    
    data_close = yf.download(tickers_list,d)['Adj Close']
    data_open = yf.download(tickers_list,d)['Open']
    
    data_close=data_close.reset_index()
    
    data_open=data_open.reset_index()
    
    data1 = pd.melt(data_open, id_vars='Date', value_vars=['AMZN','GOOG'])
    
    data1.rename(columns = {'variable': 'Company','value':'Adj Close'}, inplace = True)
    
    data2 = pd.melt(data_close, id_vars='Date', value_vars=['AMZN','GOOG'])
    
    data2.rename(columns = {'variable': 'Company','value':'Open'}, inplace = True)
    

    merged_inner = pd.merge(left=data2, right=data1, left_on=["Date","Company"], right_on=["Date","Company"])
    
    df1=df.sort_values('date',ascending=False).groupby(['ticker', 'date']).mean().reset_index()
    
    
    
    df1["date"]=df1["date"].apply(lambda x: datetime.strptime(x,'%b-%d-%y').strftime("%Y-%m-%d"))
    
    df1['date'] = pd.to_datetime(df1['date'])
    
    merged_inner_df = pd.merge(left=df1, right=merged_inner, left_on=["date","ticker"], right_on=["Date","Company"])
    
    merged_inner_df.drop(columns=["ticker","date"], inplace=True)
    
    
    
    result=''
    # prediction
    #result = model.predict(np.array([cgpa,iq,profile_score]).reshape(1,3))
    
   
    result= merged_inner_df.to_json(orient="records")
    #result= merged_inner_df["Open"][0]

    return render_template('index.html',result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)