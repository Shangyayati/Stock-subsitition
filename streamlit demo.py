import os
import csv
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from datetime import date
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import OrderedDict    
from itertools import tee
import datetime as dt
import time
import yfinance as yf
import urllib
from datetime import date, datetime

image = Image.open('yayati.png')
st.image(image)

#%%
factor_exposure=pd.read_csv('Axioma_data.csv',index_col=0)

#factor_exposure
#%%
from sklearn.cluster import KMeans
os.environ["OMP_NUM_THREADS"] = '1'

def get_company_full_name(ticker):
    #print(ticker)
    return yf.Ticker(ticker).info['longName']
    

def sub(t,factor_exposure,k):
    """
    K-means clustering method 
    """
    temp=factor_exposure.reset_index()
    if t not in list(temp['Ticker']):
        print("Wrong ticker")
        return
    industry=temp.loc[temp['Ticker']==t,'Industry'].values[0]
    MC_class=temp.loc[temp['Ticker']==t,'Market Cap class'].values[0]
    temp=temp[(temp['Industry']==industry) & (temp['Market Cap class']==MC_class)]
    temp['Company']=temp['Ticker'].apply(get_company_full_name)
    data=temp.iloc[:,:11]
    # K-means clustering
    kmeans = KMeans(n_clusters=k,tol=1e-6,random_state=0, n_init="auto")
    kmeans.fit(data.iloc[:,1:11])
    data['cluster']=kmeans.labels_
    cluster=data.loc[data['Ticker']==t,'cluster'].values[0]
    # output
    data=data[data['cluster']==cluster].sort_values('Market Sensitivity').reset_index(drop=True)
    data['Company']=data['Ticker'].apply(get_company_full_name)
    return data,temp[['Ticker','Company','Industry','Market Cap class']],\
            data[['Ticker','Company',"Earnings Yield", "Value", "Leverage", "Growth",
                    "Profitability", "Dividend Yield", "Size",'Liquidity',
                'Market Sensitivity','Medium-Term Momentum']]


#%%
stocks=['BA','CAT',
        'SRE','NEE',
        'JPM','V',
        'XOM','PSX',
        'TSLA','BKNG','SBUX',
        'AMD','AAPL',
        'JNJ','LLY','INCY','DVA',
        'PPG','FCG','EMN',
        'PLD','CCI','REG',
        'META','GOOG','ATVI','PARA','DIS'
        ]

def cal_corr(s,d):
    record=[]
    today=datetime.now()
    start = today - relativedelta(years=2)
    starting_date=str(start.date())
    l=list(data['Ticker'])
    for t in l:
        #print(t)
        temp=pd.DataFrame(yf.download(str(t).strip(), start=starting_date)['Close'])
        temp=temp.rename(columns = {'Close':t})
        record.append(temp)
    dataframe = pd.concat(record, axis=1)
    dataframe= dataframe.fillna(0).pct_change().fillna(0)
    #print(dataframe)
    #dataframe=dataframe.resample('m').sum()
    d['Correlation']=list(dataframe.corr()[s])
    output=d.sort_values('Correlation', ascending=False).reset_index(drop=True)
    return output


#%%
option = st.selectbox(
    'Select Ticker',
    ('','AAPL', 'BA', 'JPM','TSLA','META'))

    

if option!='':
    ####
    st.write("Company:")
    st.markdown('<center><b>'+get_company_full_name(option)+'</b></center>',\
                unsafe_allow_html=True)
    
    ####
    data,l1,l2=sub(option,factor_exposure,4)
    l1.reset_index(drop=True,inplace=True)
    
    data=cal_corr(option,data)
    
    st.write('Suggested Substitute Stock: ')
    st.markdown('<center><b>'+get_company_full_name(data['Ticker'][1])+'</b></center>', \
                unsafe_allow_html=True)
    ######
    
    
    #%%
    tab1, tab2, tab3 = st.tabs(["Same Class", "Yayati Research", "Correlation"])

    with tab1:
        st.subheader('Results Based on GICS and Market Cap:')
        l1
       
    
    with tab2:
        st.subheader('Results Based on Yayati Research:')
        l2
    
    with tab3:
        st.subheader('Correlation:')
        data[['Ticker','Company','Correlation']]
        
















