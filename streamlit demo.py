import os
import csv
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from datetime import date
from datetime import datetime
from dateutil.relativedelta import relativedelta
import axioma
from axioma import workspace
from axioma.account import Account
from axioma.rebalancing import Rebalancing
from axioma.analytics import Analytics
import axioma.workspace_io as handler
import axioma.dynamicactions as dynact
from axioma.group import DynamicGroup, Benchmark, Group
from axioma.metagroup import Metagroup
from axioma.workspace_element import Unit
from axioma.strategy import Strategy, Objective, Target, Scope
from collections import OrderedDict    
from axioma.assetset import AssetSet
from axioma.workspace import DatabaseProvider, Workspace, DerbyProvider
from axioma.workspace_element import ElementType
from itertools import tee
import datetime as dt
from datetime import datetime
from axioma.workspace import Workspace, FlatFileProvider
from axioma.workspace_element import ElementType
from axioma.batchjob import BatchJob, JobItem, BatchType
import time

import urllib
from datetime import date, datetime

image = Image.open('yayati.png')
st.image(image)
#axioma.ENDPOINT
# Code for changing the display format of dataframes 
#%%
def clean_ws():
    workspaceNames = workspace.get_available_workspace_names(include_temp_storage=True)
    for workspaceX in workspaceNames:
        ws_name = workspace.delete(workspaceX)      
        print('Workspace Destroyed: ' + workspaceX)
clean_ws()
# Create a workspace
current_date = datetime.strptime("2023-07-13", '%Y-%m-%d').date()
# next_period_date = datetime.strptime("2021-04-22", '%Y-%m-%d').date()
#axioma_data_dir = 'C:/Program Files/Axioma/AxiomaPortfolio/Data'+"/${yyyy}/${mm}"
axioma_data_dir="T:/axioma/data/${yyyy}/${mm}"
db_provider = DerbyProvider(axioma_data_dir,
                            risk_models="WW4AxiomaMH",
                            include_composites=True,
                            # optional only used to generate Period Returns
                            # next_period_date=next_period_date
                            )
                                
ws = Workspace("Risk Workspace", current_date, data_provider=db_provider) 
#ws.identity
# one token is used
err=handler.load_assets_from_data_provider(ws, asset_names=['37P4NKR33']) 
#%%
# build map
model=ws.get_risk_model('WW4AxiomaMH')
asset_map=ws.get_asset_map('All Map')
local_map={}
factor_value=model.get_factor_exposures('Earnings Yield')
for k,v in factor_value.items():
    key=asset_map.get_asset_mappings(k)
    for t in key:
        if t!=k:
            local_map[k]=t
factor_names = ["Earnings Yield", "Value", "Leverage", "Growth",
                    "Profitability", "Dividend Yield", "Size",'Liquidity',
                'Market Sensitivity','Medium-Term Momentum']
asset_map=ws.get_asset_map('All Map')
#factor_value=model.get_factor_exposures('Earnings Yield')
df=[]
for k,v in factor_value.items():
    temp=local_map[k]
    df.append([temp,v])
factor_exposure=pd.DataFrame(df,columns=['Ticker','Earnings Yield']).set_index('Ticker')
#factor_exposure
for factor in factor_names[1:]:
    factor_value=model.get_factor_exposures(factor)
    df=[]
    for k,v in factor_value.items():
        temp=local_map[k]
        df.append([temp,v])
    df=pd.DataFrame(df,columns=['Ticker',factor]).set_index('Ticker')
    factor_exposure[factor]=df
factor_exposure=factor_exposure.sort_index()

# Find industry
industries=ws.get_group('Classification-GICS').get_composition()
local_industries_map={}
for k,v in industries.items():
    local_industries_map[local_map[k]]=v
factor_exposure['Industry']=pd.Series(local_industries_map)

#%%
# Get market cap from Yahoo Finance
import yfinance as yf

market_cap={}
for i in factor_exposure.index:
    #print(i)
    try:
        ticker=yf.Ticker(i)
        market_cap[i]=ticker.info['marketCap']
    except:
        market_cap[i]=np.nan
factor_exposure['market_cap']=pd.Series(market_cap)
factor_exposure['Market Cap class']=factor_exposure.apply(lambda row: 'Large Cap' if row['market_cap']>1e10 
                            else ('Medium Cap' if row['market_cap']>2e9 else 
                        ('Micro Cap' if row['market_cap']<2.5e6 else 'Small Cap')), axis=1)

#factor_exposure
#%%
from sklearn.cluster import KMeans
os.environ["OMP_NUM_THREADS"] = '1'

def get_company_full_name(ticker):
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
    kmeans = KMeans(n_clusters=k,n_init=10)
    kmeans.fit(data.iloc[:,1:11])
    data['cluster']=kmeans.labels_
    cluster=data.loc[data['Ticker']==t,'cluster'].values[0]
    # output
    data=data[data['cluster']==cluster].sort_values('Market Sensitivity').reset_index(drop=True)
    data['Company']=data['Ticker'].apply(get_company_full_name)
    return data,temp[['Ticker','Company']],data[['Ticker','Company']]


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
        temp=pd.DataFrame(yf.download(t, start=starting_date)['Adj Close'])
        temp=temp.rename(columns = {'Adj Close':t})
        record.append(temp)
    dataframe = pd.concat(record, axis=1, join='inner')
    dataframe= dataframe.pct_change().dropna()
    #dataframe
    dataframe=dataframe.resample('m').sum()
    d['Correlation']=list(dataframe.corr()[s])
    output=d.sort_values('Correlation', ascending=False).reset_index(drop=True)
    return output


#%%
option = st.selectbox(
    'Select Ticker',
    ('','AAPL', 'BA', 'JPM','TSLA','META'))

    

if option!='':
    st.write("Stock to be replaced: "+get_company_full_name(option))
    
    data,l1,l2=sub(option,factor_exposure,4)
    l1.reset_index(drop=True,inplace=True)
    data=cal_corr(option,data)
    
    st.write('Suggested subsitite stock: '+get_company_full_name(data['Ticker'][1]))
    st.write('Results based on GICS and Market Cap:')
    l1
    st.write('Results Based on Yayati Search:')
    l2
    st.write('Risk factors and Corrrelation:')
    data





























