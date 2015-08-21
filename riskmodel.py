
equityPriceP = "processeddata/sp500_price_p.csv"
fundamentalDataP = "processeddata/sp500_fundamental_p.csv"
sectorDataP = "processeddata/sp500_sectors_p.csv"
ffFactorsP = "processeddata/fffactors_p.csv"
results = "results/results"


import pandas as pd

fields = ['Ticker', 'Date', 'AClose', 'AVol', 'DateTime', 'YM']
equityPrice = pd.read_csv(equityPriceP,  header=None, skiprows=1, names = fields)
equityPrice= equityPrice[equityPrice['YM'] > 199900]

factors = ['YM', 'Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
mreturnP = pd.read_csv(ffFactorsP, skiprows=1, header=None , names = factors)

sectors = pd.read_csv(sectorDataP)


equityADVP = "/home/assdef/Documents/D/FinanceData/sp500_adv_p.csv"
equityADV = pd.read_csv(equityADVP)
def normalize(dataFrame, column):
      from scipy.stats import norm
      dataFrame[column] = dataFrame[column].rank(ascending=True)
      dataFrame[column] -= 0.5
      dataFrame[column] /= len(dataFrame.index)
      dataFrame[column] = dataFrame[column].apply(lambda x: norm.ppf(x))
      return dataFrame

liquidity = normalize(equityADV, "ADV").loc[:, ["Ticker", "YM", "ADV"] ]
    

fundamentalDataP = "/home/assdef/Documents/D/FinanceData/sp500_fundamental_p.csv"
fundamentalDataP = pd.read_csv(fundamentalDataP)


common_tickers = pd.read_csv("/home/assdef/Documents/D/FinanceData/commonTicker.csv")
common_tickers = set(common_tickers.ix[:,1])


commonTime = set(equityPrice[equityPrice.Ticker == list(common_tickers)[0]].YM)
for tkr in common_tickers: 
   commonTime=commonTime | set(equityPrice['YM'].unique())


equityPriceC = equityPrice.loc[:, ['YM', 'Ticker', 'AClose']]
equityPriceC['AClose'] = equityPriceC['AClose'].astype(float)
equityPriceC = equityPriceC.loc[:,['Ticker','AClose','YM']]
common_tickers = set(equityPriceC.Ticker.unique())
t_quity = pd.DataFrame()
for ik,ticker_group in equityPriceC.groupby('Ticker'):
    t_quity = pd.concat([ticker_group.loc[:,['YM','AClose']].groupby('YM').mean(),t_quity],axis=1)
    
t_quity.columns = common_tickers
   


import math
currReturn = pd.DataFrame()  # initialize a return dataframe
import numpy as np
for tkr in list(common_tickers): 
    tmpdf = t_quity[tkr]
    tmprtndf = ((tmpdf-tmpdf.shift(1))/tmpdf).dropna()
    if (tmprtndf.isnull().values.sum() < 5):
        currReturn = pd.concat([currReturn, tmprtndf], axis=1)
currReturn['YM'] = list(commonTime)[1:]



futureReturn = pd.DataFrame()  # initialize a return dataframe
# Normalize all time series to have mean zero and variance one and compute their returns 
import numpy as np
for tkr in list(common_tickers): 
    tmpdf = t_quity[tkr]
    tmprtndf = ((tmpdf.shift(1)-tmpdf.shift(2))/tmpdf).dropna()
    if (tmprtndf.isnull().values.sum() < 5):
        futureReturn = pd.concat([futureReturn, tmprtndf], axis=1)
futureReturn['YM'] = list(commonTime)[2:]


import statsmodels.formula.api as sm
import statsmodels.api as sm
betas = pd.DataFrame()
for pos in range(0, (len(mreturnP)-60) ):
#for pos in range(0,1):
     ## MFactor Returns
    beta = pd.DataFrame()
    mRets  = mreturnP[pos:60]
    mRets.index = mreturnP[pos:60]['YM']
    currYM = list(mreturnP.YM)[pos + 60]
    nextYM = list(mreturnP.YM)[pos + 61]
    currRF = list(mreturnP.RF)[pos + 60]
    
    for tkt in common_tickers:
        stockReturn = currReturn[currReturn.YM.isin(mRets.YM)][tkt]
        #delete NAN data
        combined = pd.concat([stockReturn,mRets], axis =1)
        cleaned = combined.dropna(axis=0)
        betaReturn = cleaned[tkt]-cleaned['RF']
        # Create linear regression object
        if len(cleaned) is not 0:
            aweights = [math.pow(0.5, math.pow((1/23), x) ) for x in range(0, len(cleaned) )]
            params = pd.DataFrame ({
            'Ticker':[tkt],
            'YM': [currYM],
            'Mkt_RF': [sm.WLS(betaReturn.values,cleaned['Mkt_RF'].values, weights = aweights).fit().params[0]],
            'SMB': [sm.WLS(betaReturn.values,cleaned['SMB'].values, weights = aweights).fit().params[0]],
            'HML': [sm.WLS(betaReturn.values,cleaned['HML'].values, weights = aweights).fit().params[0]],
            'CMA': [sm.WLS(betaReturn.values,cleaned['RMW'].values, weights = aweights).fit().params[0]],
            'RF': [sm.WLS(betaReturn.values,cleaned['CMA'].values, weights = aweights).fit().params[0]] })
        beta = pd.concat([beta, params])
    
    betas = pd.concat([betas, beta])
        
    currLiquidity =  liquidity[liquidity.Ticker.isin(common_tickers)][liquidity.YM ==currYM].groupby('Ticker').mean()
    currLiquidity = currLiquidity.reset_index()
    currLiquidity.index = currLiquidity.Ticker
    currPrice   = equityPriceC[equityPriceC.Ticker.isin(common_tickers)][equityPrice.YM ==currYM].groupby('Ticker').mean()
    currPrice = currPrice.reset_index()
    currPrice.index = currPrice.Ticker
    existingTickers = currPrice.Ticker.unique()
    sectors.index = sectors.Ticker

    stockReturn = currReturn[currReturn.YM ==currYM]
    stockReturn = stockReturn.transpose()
    stockReturn = stockReturn.reset_index()
    stockReturn.columns =["Ticker", "Return"]
    stockReturn.index = stockReturn.Ticker
    beta.reset_index()
    beta.index = beta.Ticker
    del beta['Ticker']

    beta.drop_duplicates(inplace=True)
    univ = pd.concat([currPrices, currLiquidity, sectors, stockReturn, beta], axis =1, join='inner')
    univ = univ.dropna(axis=0)
    del univ['Ticker']
    del univ['YM']
    del univ['AClose']
    
    Y = univ["Return"] 
    del univ["Return"]
    X = univ
    
    bweights = [math.pow(0.5, math.pow((1/23), x) ) for x in range(0, len(Y) )]
    mod_wls = sm.WLS(Y, X, weights= bweights)
    res_wls = mod_wls.fit()
    
    print res_wls.summary()
    prediction = res_wls.predict(X)
    
    specialReturn = Y.sub(prediction)
    univ["Return"] = Y
    univ["Special Return"] = specialReturn
    fileName = ''.join([results, str(currYM) ,'.csv'])
    univ.to_csv(fileName)






