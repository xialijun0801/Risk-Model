# read data in from downloded csv file
equityData = "database/sp500_data.csv"
equityDataSmall = "database/sp500_data_small.csv"
equityADVP = "processeddata/sp500_adv_p.csv"
equityPriceP = "processeddata/sp500_price_p.csv"

fundamentalData = "database/sp500_fundamental.csv"
fundamentalDataSmall = "database/sp500_fundamental_small.csv"
fundamentalDataSmallS = "processeddata/sp500_fundamental_small_s.csv"
fundamentalDataP = "processeddata/sp500_fundamental_p.csv"

ffFactors = "database/F-F_Research_Data_5_Factors_2x3.txt"
ffFactorsP = "processeddata/fffactors_p.csv"
commonTicker = "processeddata/commonTicker.csv"

sectorData = "database/SP500_sectors.csv"
sectorDataP = "processeddata/sp500_sectors_p.csv"

results = "results/results"


if __name__ == '__main__':
   #add heads to csv file
   import pandas as pd
   import csv


   def normalize(dataFrame, column):
      from scipy.stats import norm
      dataFrame[column] = dataFrame[column].rank(ascending=True)
      dataFrame[column] -= 0.5
      dataFrame[column] /= len(dataFrame.index)
      dataFrame[column] = dataFrame[column].apply(lambda x: norm.ppf(x))
      return dataFrame

   # this part is to create new csv file with
   equityFundamental = pd.read_csv(fundamentalDataSmall)
   equityFundamental.columns = ['Agg', 'Date', 'Value']
   print(equityFundamental.ix[1:10,])
   agg = equityFundamental['Agg'].apply(lambda x: pd.Series(x.split('_')))
   agg.columns=['Ticker', 'Field', 'Frequency']
   del equityFundamental['Agg']
   frames = [agg, equityFundamental]
   combined = pd.concat(frames,axis=1)
   pd.unique(combined.Ticker)
   combined.to_csv(fundamentalDataSmallS)

   equityFundamentalS = pd.read_csv(fundamentalDataSmallS)

   # market return, smallcap-bigcap, highvalue-lowvalue, robust-weak, conservative-aggressive
   factors = ['YM', 'Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
   mreturns = pd.read_csv(ffFactors, skiprows=1, header=None , names = factors, delim_whitespace=True)
   mreturns.convert_objects(convert_numeric=True)
   returns = mreturns[mreturns.YM > 199900]
   mreturns['RF'] = mreturns['RF'] / 100
   mreturns.to_csv(ffFactorsP) # save macro factors

   fields = ['Ticker', 'Date', 'AClose', 'AVol']
   equityPrice = pd.read_csv(equityDataSmall, usecols= [0, 1, 12, 13], names = fields)
   print(equityPrice.ix[1:10, ])
   #get common tickers appear in both equityData and fundemental Data
   commonTickers=set(equityPrice['Ticker'].unique()).intersection(set(equityFundamentalS['Ticker'].unique()))
   pd.DataFrame(list(commonTickers)).to_csv(commonTicker, names = 'Ticker')
   equityPrice.dropna()
   equityPrice['DateTime'] = pd.to_datetime(equityPrice['Date'])
   equityPrice['YM'] = equityPrice['DateTime'].apply(lambda x: x.year*100 + x.month)
   equityPrice = equityPrice[equityPrice.Ticker.isin(commonTickers)]
   equityPrice.to_csv(equityPriceP, names = ['Ticker', 'Date', 'AClose', 'AVol','YM'], index = False)

   #read sector information
   sectorNames = ['Ticker', 'Code', 'Name', 'Sector']
   sectorInfo = pd.read_csv(sectorData, skiprows=1, header=None, names = sectorNames, sep = ',')
   sectorInfo.index = sectorInfo['Ticker']
   sectorsDummy = pd.crosstab(index=sectorInfo['Ticker'], columns=[sectorInfo['Sector']]) 
   sectorsDummy = sectorsDummy.reset_index()
   sectorsDummy = sectorsDummy[sectorsDummy.Ticker.isin(commonTickers)]
   sectorsDummy.to_csv(sectorDataP)

   #start to processData, only keep the common tickers
   securityFundamental =pd.read_csv(fundamentalDataSmallS)
   #securityFundamental.dropna()
   securityFundamental = securityFundamental[securityFundamental['Value']!=0]
   securityFundamental = securityFundamental[securityFundamental['Field']=="MARKETCAP"]
   securityFundamental['DateTime'] = pd.to_datetime(securityFundamental['Date'])
   securityFundamental['YM'] = securityFundamental['DateTime'].apply(lambda x: x.year * 100 + x.month)
   securityFundamental = securityFundamental[securityFundamental['YM'] > 199900]
   securityFundamental.index = securityFundamental.Ticker
   securityFundamental = securityFundamental[securityFundamental.Ticker.isin(commonTickers)]
   securityFundamental.to_csv(fundamentalDataP)

   fields = ['Ticker','Date', 'AClose', 'AVol', 'DateTime', 'YM']
   equityPrice = pd.read_csv(equityPriceP, skiprows = 1, names = fields)
   equityPrice = equityPrice[equityPrice['YM'] > 199900]
   equityPrice['ADV'] = equityPrice.loc[:, ('AVol')] * equityPrice.loc[:, ('AClose')]
   equityPrice.groupby(['Ticker', 'YM'])['ADV'].mean()
   equityPrice.to_csv(equityADVP, index =False)
