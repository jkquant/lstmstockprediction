import pandas as pd
import numpy as np
from features import *

# Load our CSV Data

data = pd.read_csv('pricedata/EURUSDhours.csv')

data.columns = [['Date', 'open','high', 'low', 'close', 'AskVol']]

data = data.set_index(pd.to_datetime(data.Date))

data = data[[ 'open','high', 'low', 'close', 'AskVol']]

prices = data.drop_duplicates(keep=False)

momentumKey = [3,4,5,8,9,10]
stochasticKey = [3,4,5,8,9,10]
williamsKey = [6,7,8,9,10]
procKey = [1,12,13,14,15]
wadlKey =[15]
macdKey = [15,30]
heikenashiKey = [15]
paverageKey = [2]
fourierKey = [10,20,30]
sineKey = [5,6]



Keylist = [momentumKey, stochasticKey, williamsKey, procKey, wadlKey, macdKey,
           heikenashiKey, paverageKey, fourierKey, sineKey ]

momentumDict = momentum(prices, momentumKey)
print('1')
stochasticDict = stochastic(prices, stochasticKey)
print('2')
williamsDict = williams(prices, williamsKey)
print('3')
procDict = proc(prices, procKey)
print('4')
wadlDict = wadl(prices, wadlKey)
print('5')
macdDict = macd(prices, macdKey)
print('6')
paverageDict = paverage(prices, paverageKey)
print('7')
fourierDict = fourier(prices, fourierKey)
print('8')
sineDict = sine(prices, sineKey)
print('9')

hkaprices = prices.copy()
hkaprices['Symbol'] = 'SYMB'

HKA = OHLCresample(hkaprices, '15H')
heikenDict = heikenashi(HKA, heikenashiKey)
print('10')

#Create list of Dictionaries

dictlist = [momentumDict.close, stochasticDict.close, williamsDict.close, procDict.proc, wadlDict.wadl,
            macdDict.line, heikenDict.candles, paverageDict.avs, fourierDict.coeffs, sineDict.coeffs]

# List of 'base column names:

colFeat = ['momentum', 'stoch', 'will', 'proc', 'wadl','macd', 'heiken', 'paverage', 'fourier', 'sine' ]


# Populate the Master Frame

masterFrame = pd.DataFrame(index=prices.index)

for i in range(0, len(dictlist)):

    print('Running.......: ', dictlist[i])
    if colFeat[i] == 'macd':
        colID = colFeat[i] + str(Keylist[5][0]) + str(Keylist[5][1])

        masterFrame[colID] = dictlist[i]
    else:
        for j in Keylist[i]:
            for k in list(dictlist[i][j]):

                colID = colFeat[i] + str(j) + k
                masterFrame[colID] = dictlist[i][j][k]


threshold = round(0.7*len(masterFrame))
masterFrame[['open','high','low','close']] = prices[['open', 'high','low', 'close']]

# Heiken Asji is resamples ==> empty data inbetween
# bafkfill all the Nan values in heiken
masterFrame.heiken15open = masterFrame.heiken15open.fillna(method='bfill')
masterFrame.heiken15high = masterFrame.heiken15high.fillna(method='bfill')
masterFrame.heiken15low = masterFrame.heiken15low.fillna(method='bfill')
masterFrame.heiken15close = masterFrame.heiken15close.fillna(method='bfill')


# Drop colunmns that have 30% or more NAN data

masterFrameCleaned = masterFrame.copy()

masterFrameCleaned = masterFrameCleaned.dropna(axis=1, thresh=threshold)
masterFrameCleaned = masterFrameCleaned.dropna(axis=0)

masterFrameCleaned.to_csv('pricedata/masterFrame.CSV')

print(masterFrameCleaned.head(10))
print(masterFrameCleaned.columns)
print('Completed Feature Calculations')