
# coding: utf-8

# # 特征提取函数库

# In[1]:


# !/usr/bin/python

#技术指标函数库，函数的输入均为DataFrame格式的data，以及时间跨度ndays
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import pandas_datareader.data as web


# ### Simple Moving Average
# 一个简单的移动平均线是通过计算特定数量的期间的平均价格来形成的。

# In[45]:


def SMA(data,ndays):
    data[str(ndays)+'_SMA'] = data['Close'].rolling(ndays).mean()
    return data


# ### Exponential Moving Average
# 指数均线（EMA）通过对最近的价格施加更多的权重来减少滞后。

# In[19]:


def EMA(data,ndays):
    #data['EMA'] = pd.ewma(data['Close'],span=ndays,min_periods=ndays-1)
    data[str(ndays)+'_EMA'] = data['Close'].ewm(ignore_na=False,span=ndays,min_periods=ndays,adjust=True).mean()
    return data
#Test Code
# ema_data = web.get_data_yahoo('AAPL',start='1/1/2010',end='1/1/2016')
# ema = EMA(ema_data,10)
# ema.head(20)


# ### Exponential Moving Average of Volume
# 指数均线（EMA）通过对最近的成交量施加更多的权重来减少滞后。

# In[61]:


def EMAV(data,ndays):
    data[str(ndays)+'_EMA'] = data['Volume'].ewm(ignore_na=False,span=ndays,min_periods=ndays,adjust=True).mean()
    return data


# ### Bollinger Bands
# 基于价格的标准偏差显示“正常”价格变动的上限和下限的图表叠加图。能够用于同样标准下不同证券的技术度量，还能够用于鉴别M型顶部以及W型底部或者鉴别趋势的强度。
# 计算公式：
#  Middle Band = 20-day simple moving average (SMA)
#  
#  Upper Band = 20-day SMA + (20-day standard deviation of price x 2) 
#  
#  Lower Band = 20-day SMA - (20-day standard deviation of price x 2)

# In[33]:


def BollingerBands(data,ndays=20):
    MA = Series(data['Close'].rolling(ndays).mean(),name=str(ndays)+' SMA')
    STD = Series(data['Close'].rolling(ndays).std())
    bd1 = MA + 2*STD
    BD1 = Series(bd1,name='Upper Bollinger Band')
    bd2 = MA - 2*STD
    BD2 = Series(bd2,name='Lower Bollinger Band')
    data = data.join(MA)
    data = data.join(BD1)
    data = data.join(BD2)
    return data
# Test Code
# Retrieve data from yahoo finance
# bb_data = web.get_data_yahoo('AAPL',start='1/1/2010',end='1/1/2016')
# BB = BollingerBands(bb_data)
# BB.tail()


# ### Average True Range
# 平均真实范围是用来衡量波动性的指标。首先看TR的概念。True Range取下面三个值中的最大值如下：
# Method 1: 当前时间段（通常为一个交易日）最高价减去最低价
# Method 2: 当前时间段（通常为一个交易日）最高价减去前一交易日收盘价的绝对值
# Method 3: 当前时间段（通常为一个交易日）最低价减去前一交易日收盘价的绝对值
# ATR通常取14个交易日为一个周期，计算这14天TR的平均值，第一个ATR为最前面的14个交易日TR的平均值，从第二个ATR开始，计算方法为(前一个ATR*13+当前TR)/14，为了简单起见用rolling后的mean值代替

# In[21]:


def ATR(data,ndays=14):
    data['HL'] = data['High']-data['Low']
    data['HC'] = abs(data['High']-data['Close'].shift())
    data['LC'] = abs(data['Low']-data['Close'].shift())
    data['TR'] = data[['HL','HC','LC']].max(axis=1)
    data['ATR'] = data['TR'].rolling(ndays).mean()
    data = data.drop(['HL','HC','LC','TR'],axis=1)
    return data
# Test Code
# Retrieve data from yahoo finance
# atr_data = web.get_data_yahoo('AAPL',start='1/1/2010',end='1/1/2016')
# ATR(atr_data).head(20)


# ### Chandelier Exit
# 可用于设置多头和空头的尾随止损的指标。吊灯出口在平均真实范围基础上设置尾随止损。该指标用于指导交易者跟踪趋势防止提前退出。通常该出口表现为上升行情在价格下方，下降行情在价格上方。
# 计算公式：
# #### Chandelier Exit (long) = 22-day High - ATR(22) x 3 
# #### Chandelier Exit (short) = 22-day Low + ATR(22) x 3

# In[13]:


def ChandelierExit(data,ndays=22):
    df = ATR(data,ndays)
    df['CEH'] = df['High'].rolling(ndays).max()
    df['CEL'] = df['Low'].rolling(ndays).min()
    df['CE Long'] = df['CEH']-df['ATR']*3
    df['CE Short'] = df['CEL']-df['ATR']*3
    df = df.drop(['TR','ATR','CEH','CEL'],axis=1)
    return df
# Test Code
# Retrieve data from yahoo finance
# ce_data = web.get_data_yahoo('AAPL',start='1/1/2010',end='1/1/2016')
# ChandelierExit(ce_data,22).head(25)


# ### Ichimoku Clouds
# 一个综合指标，定义支持和阻力，确定趋势方向，衡量动力和提供交易信号。
# 一目均衡图包含五条线，分别是：
# 转换线：(9-period high + 9-period low)/2
# 基本线：(26-period high + 26-period low)/2
# 先导跨度A：(转换线 + 基本线)/2
# 先导跨度B：(52-period high + 52-period low)/2
# 滞后跨度：Close plotted 26 days in the past

# In[2]:


def IchiCloud(data,d1=9,d2=26,d4=52,d5=26):
    data['Conv Line'] = (data['High'].rolling(d1).max()+data['Low'].rolling(d1).min())/2
    data['Base Line'] = (data['High'].rolling(d2).max()+data['Low'].rolling(d2).min())/2
    data['Leading SpanA'] = (data['Conv Line']+data['Base Line'])/2
    data['Leading SpanB'] = (data['High'].rolling(d4).max()+data['Low'].rolling(d4).min())/2
    data['Lagging Span'] = data['Close'].rolling(d5).mean()
    return data
# Test Code
# Retrieve data from yahoo finance
# ic_data = web.get_data_yahoo('AAPL',start='1/1/2010',end='1/1/2016')
# IchiCloud(ic_data).tail(20)


# ### Kaufman's Adaptive Moving Average (KAMA)
# 一个独特的移动平均线，可以解释波动，并自动适应价格行为。
# KAMA(10,2,30).
# 
# 10 is the number of periods for the Efficiency Ratio (ER).
# 2 is the number of periods for the fastest EMA constant.
# 30 is the number of periods for the slowest EMA constant.
# Before calculating KAMA, we need to calculate the Efficiency Ratio (ER) and the Smoothing Constant (SC). 
# #### ER = Change/Volatility
# Change = ABS(Close - Close (10 periods ago))
# Volatility = Sum10(ABS(Close - Prior Close))
# Volatility is the sum of the absolute value of the last ten price changes (Close - Prior Close).
# 
# #### SC = Smoothing Constant
# SC=[ER x (fastest SC - slowest SC) + slowest SC]的平方
# fastest SC是shorter EMA (2-periods)的常数2/(2+1)
# slowest SC是slowest EMA (30-periods)的常数2/(30+1)
# SC = [ER x (2/(2+1) - 2/(30+1)) + 2/(30+1)]的平方
# 
# ### Current KAMA = Prior KAMA + SC x (Price - Prior KAMA)
# the first KAMA is just a simple moving average.

# In[17]:


def KAMA(data,ndays=10):
    data['Change'] = abs(data['Close']-data['Close'].shift(10))
    data['Volatility'] = abs(data['Close']-data['Close'].shift()).rolling(ndays).sum()
    data['ER'] = data['Change']/data['Volatility']
    data['SC'] = np.square(data['ER']*(2.0/(2+1)-2.0/(30+1))+2.0/(30+1))
    data['KAMA'] = data['Close'].rolling(ndays).mean()
    data['KAMA'][:ndays]= np.nan
    i=1
    #这里用了一种最直观的方法，但是速度非常慢
    while i<len(data['KAMA'][ndays+1:]):
        s = data['KAMA']
        s.iloc[ndays+i] = data['KAMA'][ndays+i-1]+data['SC'][ndays+i]*(data['Close'][ndays+i]-data['KAMA'][ndays+i-1])
        data['KAMA'] = s
        i = i+1
    data = data.drop(['Change','Volatility','ER','SC'],axis=1)
    return data
# kama_data = web.get_data_yahoo('AAPL',start='1/1/2010',end='1/1/2016')
# df = KAMA(kama_data)
# df.head(30)


# ### Keltner Channels
# 基于平均真实价格范围的图表叠加显示价格变动的上限和下限。Keltner通道是基于波动率的信封，设置在指数移动平均线的上方和下方。跟Bollinger通道基于标准差的方式类似，只不过用的是平均真实范围（ATR）。
# 计算公式：
# Middle Line: 20-day exponential moving average 
# Upper Channel Line: 20-day EMA + (2 x ATR(10))
# Lower Channel Line: 20-day EMA - (2 x ATR(10))

# In[25]:


def KC(data,ndays=20):
    ema = EMA(data,ndays)
    df = ATR(data,10)
    df[str(ndays)+'_EMA'] = ema[str(ndays)+'_EMA']
    df['Upper Line'] = df[str(ndays)+'_EMA'] + 2*df['ATR']
    df['Lower Line'] = df[str(ndays)+'_EMA'] - 2*df['ATR']
    del df['ATR']
    return df
#Test Code
kc_data = web.get_data_yahoo('AAPL',start='1/1/2010',end='1/1/2016')
kc = KC(kc_data)
kc.head(30)


# ### Moving Average Envelopes
# 覆盖图由简单的移动平均线形成的通道组成。
# 计算公式：
# Upper Envelope: 20-day SMA + (20-day SMA x .025)
# Lower Envelope: 20-day SMA - (20-day SMA x .025)

# In[21]:


def MAE(data,ndays=20):
    data[str(ndays)+'_SMA'] = SMA(data,ndays)[str(ndays)+'_SMA']
    data['Upper Env'] = data[str(ndays)+'_SMA'] + data[str(ndays)+'_SMA']*0.025
    data['Lower Env'] = data[str(ndays)+'_SMA'] - data[str(ndays)+'_SMA']*0.025
    return data
# Test Code
# mae_data = web.get_data_yahoo('AAPL',start='1/1/2010',end='1/1/2016')
# mae = MAE(mae_data)
# mae.head(25)


# ### Pivot Points
# 图表覆盖图显示低于价格的逆转点，在下降趋势中高于价格。
# Stand Pivot Points计算公式,
# 注意如果想得到今天的Pivot Point图，High、Low、Close用昨天的数据，如果想得到这周的图，则用上周的数据，这个月的图则用上月数据：
# ---------------------------
# Pivot Point (P) = (High + Low + Close)/3
# 
# Support 1 (S1) = (P x 2) - High
# 
# Support 2 (S2) = P  -  (High  -  Low)
# 
# Resistance 1 (R1) = (P x 2) - Low
# 
# Resistance 2 (R2) = P + (High  -  Low)
# 
# Fibonacci Pivot Points计算公式：
# ------------------------------
# Pivot Point (P) = (High + Low + Close)/3
# 
# Support 1 (S1) = P - {.382 * (High  -  Low)}
# 
# Support 2 (S2) = P - {.618 * (High  -  Low)}
# 
# Support 3 (S3) = P - {1 * (High  -  Low)}
# 
# Resistance 1 (R1) = P + {.382 * (High  -  Low)}
# 
# Resistance 2 (R2) = P + {.618 * (High  -  Low)}
# 
# Resistance 3 (R3) = P + {1 * (High  -  Low)}
# 
# Demark Pivot Points计算公式：
# -----------------------------
# If Close < Open, then X = High + (2 x Low) + Close
# 
# If Close > Open, then X = (2 x High) + Low + Close
# 
# If Close = Open, then X = High + Low + (2 x Close)
# 
# Pivot Point (P) = X/4
# 
# Support 1 (S1) = X/2 - High
# 
# Resistance 1 (R1) = X/2 - Low

# In[22]:


def StandPP(data):
    data['PP'] = (data['High'].shift()+data['Low'].shift()+data['Close'].shift())/3
    data['S1'] = data['PP']*2 - data['High'].shift()
    data['S2'] = data['PP'] - (data['High'].shift()-data['Low'].shift())
    data['R1'] = data['PP']*2 - data['Low'].shift()
    data['R2'] = data['PP'] - (data['High'].shift()-data['Low'].shift())
    return data
# spp_data = web.get_data_yahoo('AAPL',start='1/1/2010',end='1/1/2016')
# spp = StandPP(spp_data)
# spp.head()


# In[24]:


def FiboPP(data):
    data['PP'] = (data['High'].shift()+data['Low'].shift()+data['Close'].shift())/3
    data['S1'] = data['PP'] - 0.382*(data['High'].shift() - data['Low'].shift())
    data['S2'] = data['PP'] - 0.618*(data['High'].shift() - data['Low'].shift())
    data['S3'] = data['PP'] - (data['High'].shift() - data['Low'].shift())
    data['R1'] = data['PP'] + 0.382*(data['High'].shift() - data['Low'].shift())
    data['R2'] = data['PP'] + 0.618*(data['High'].shift() - data['Low'].shift())
    data['R3'] = data['PP'] + (data['High'].shift() - data['Low'].shift())
    return data
# fpp_data = web.get_data_yahoo('AAPL',start='1/1/2010',end='1/1/2016')
# fpp = FiboPP(fpp_data)
# fpp.head()


# In[27]:


def DemarkPP(data):
    data['X'] = np.nan
    data['Prior_Close'] = data['Close'].shift()
    data['Prior_Open'] = data['Open'].shift()
    data['Prior_High'] = data['High'].shift()
    data['Prior_Low'] = data['Low'].shift()
    #布尔索引分三类
    df1 = data[data['Prior_Close']<data['Prior_Open']]
    df2 = data[data['Prior_Close']>data['Prior_Open']]
    df3 = data[data['Prior_Close']==data['Prior_Open']]
    #对三类分别进行处理
    df1.loc[:,'X'] = df1['Prior_High'] + 2*df1['Prior_Low'] + df1['Prior_Close']
    df2.loc[:,'X'] = 2*df2['Prior_High'] + df2['Prior_Low'] + df2['Prior_Close']
    df3.loc[:,'X'] = df3['Prior_High'] + df3['Prior_Low'] + 2*df3['Prior_Close']
    #将三类填充到原dataframe
    data[data['Prior_Close']<data['Prior_Open']] = df1
    data[data['Prior_Close']>data['Prior_Open']] = df2
    data[data['Prior_Close']==data['Prior_Open']] = df3
    data['PP'] = data['X']/4
    data['S1'] = data['X']/2 - data['Prior_High']
    data['R1'] = data['X']/2 - data['Prior_Low']
    data = data.drop(['X','Prior_Close','Prior_Open','Prior_High','Prior_Low'],axis=1)
    return data
# Test Code
# dpp_data = web.get_data_yahoo('AAPL',start='1/1/2010',end='1/1/2016')
# dpp = DemarkPP(dpp_data)
# dpp.head()


# ### Price Channels
# 一个覆盖图，显示了一段时间内从最高和最低的低点开始的渠道。
# 计算公式：
# Upper Channel Line: 20-day high
# Lower Channel Line: 20-day low
# Centerline: (20-day high + 20-day low)/2

# In[18]:


def PC(data,ndays=20):
    data[str(ndays)+'_High'] = data['High'].rolling(ndays).max()
    data[str(ndays)+'_Low'] = data['Low'].rolling(ndays).min()
    data['CenterLine'] = (data[str(ndays)+'_High'] + data[str(ndays)+'_Low'])/2
    return data
# Test Code
# pc_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# pc = PC(pc_data)
# pc.head(25)


# ### Accumulation Distribution Line
# 结合价格和交易量来显示资金如何流入或流出股票。
#  1. Money Flow Multiplier = [(Close  -  Low) - (High - Close)] /(High - Low) 
# 
#  2. Money Flow Volume = Money Flow Multiplier x Volume for the Period（介于-1和1之间）
# 
#  3. ADL = Previous ADL + Current Period's Money Flow Volume

# In[28]:


def ADL(data):
    data['MF Multiplier'] = (2*data['Close']-data['Low']-data['High'])/(data['High']-data['Low'])
    data['MF Volume'] = data['MF Multiplier']*data['Volume']
    data['Accu-Dist Line'] = data['MF Volume'].cumsum()
    data = data.drop(['MF Multiplier','MF Volume'],axis=1)
    return data
# Test Code
# adl_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# adl = ADL(adl_data)
# adl.head(10)


# ### Aroon
# 使用Aroon Up和Aroon Down来确定股票是否趋势.Aroon是梵文，是黎明伊始的光。本指标用来判断股票是否在趋势中以及衡量趋势的强度。

# In[29]:


def Aroon(data,ndays=25):
    rmlag_high = lambda xs: np.argmax(xs[::-1])
    rmlag_low = lambda xs: np.argmin(xs[::-1])
    data['Days since last High'] = data['High'].rolling(center=False,min_periods=0,window=ndays).apply(func=rmlag_high).astype(int)
    data['Days since last Low'] = data['Low'].rolling(center=False,min_periods=0,window=ndays).apply(func=rmlag_low).astype(int)
    data['Aroon-Up'] = ((25-data['Days since last High'])/25) * 100
    data['Aroon-Down'] = ((25-data['Days since last Low'])/25) * 100
    del data['Days since last High']
    del data['Days since last Low']
    return data
# Test Code
# aro_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# aro = Aroon(aro_data)
# aro.head(30)


# ### Aroon Oscillator
# 衡量Aroon Up和Aroon Down的区别。
# 该指标在-100和+100之间波动，零线为中线。 当振荡器为正时存在上升趋势偏差，而当振荡器为负时存在下降趋势偏差。
# Aroon Up = 100 x (25 - Days Since 25-day High)/25
# Aroon Down = 100 x (25 - Days Since 25-day Low)/25
# Aroon Oscillator = Aroon-Up  -  Aroon-Down

# In[30]:


def AroonOscillator(data,ndays=25):
    data = Aroon(data,ndays)
    data['AroonOscillator'] = data['Aroon-Up'] - data['Aroon-Down']
    del data['Aroon-Up']
    del data['Aroon-Down']
    return data
# Test Code
# aro_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# aro = AroonOscillator(aro_data)
# aro.head(30)


# ### Average Directional Index (ADX)
# 显示股票是趋势还是震荡。

# In[31]:


def ADX(data):
    data['Prior_High'] = data['High'].shift()
    data['Prior_Low'] = data['Low'].shift()
    data['Diff_High'] = data['High'] - data['Prior_High']
    data['Diff_Low'] = data['Low'] - data['Prior_Low']
    data['+DM'] = 0.00
    data['-DM'] = 0.00
    df1 = data[data['Diff_High']>data['Diff_Low']]
    df2 = data[data['Diff_High']<=data['Diff_Low']]
    func = lambda x: x if x>0 else 0.00
    df1.loc[:,'+DM'] = df1['Diff_High'].apply(func)
    df2.loc[:,'-DM'] = df2['Diff_Low'].apply(func)
    data[data['Diff_High']>data['Diff_Low']] = df1
    data[data['Diff_High']<=data['Diff_Low']] = df2
    data = data.drop(['Prior_High','Prior_Low','Diff_High','Diff_Low'],axis=1)
    return data
# adx_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# adx = ADX(adx_data)
# adx.head(10)


# ### Bollinger BandWidth
# 从Bollinger Bands中提取的指标。BandWidth测量高频带和低频带之间的百分比差异。
# 计算公式：
# ( (Upper Band - Lower Band) / Middle Band) * 100

# In[34]:


def BBWidth(data,ndays=20):
    data = BollingerBands(data,ndays)
    data['BBWidth'] = (data['Upper Bollinger Band'] - data['Lower Bollinger Band'])/data[str(ndays)+' SMA'] * 100
    data = data.drop([str(ndays)+' SMA','Upper Bollinger Band','Lower Bollinger Band'],axis=1)
    return data
# bbw_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# bbw = BBWidth(bbw_data)
# bbw.head(25)


# ### %B Indicator
# 显示价格和标准差布林频带之间的关系。
# 计算公式：
# %B = (Price - Lower Band)/(Upper Band - Lower Band)
# 含义：
# %B equals 1 when price is at the upper band
# 
# %B equals 0 when price is at the lower band
# 
# %B is above 1 when price is above the upper band
# 
# %B is below 0 when price is below the lower band
# 
# %B is above .50 when price is above the middle band (20-day SMA)
# 
# %B is below .50 when price is below the middle band (20-day SMA)

# In[35]:


def PercentB(data,ndays=20):
    data = BollingerBands(data,ndays)
    data['%B'] = (data['Close']-data['Lower Bollinger Band'])/(data['Upper Bollinger Band']-data['Lower Bollinger Band'])
    data = data.drop([str(ndays)+' SMA','Upper Bollinger Band','Lower Bollinger Band'],axis=1)
    return data
# pb_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# pb = PercentB(pb_data)
# pb.head(25)


# ### Chaikin Money Flow
# ADL之外另一种结合价格以及交易量衡量股票资金流入流出情况的指标。
# 计算公式：
# 1. Money Flow Multiplier = [(Close  -  Low) - (High - Close)] /(High - Low) 
# 
# 2. Money Flow Volume = Money Flow Multiplier x Volume for the Period
# 
# 3. 20-period CMF = 20-period Sum of Money Flow Volume / 20 period Sum of Volume 

# In[36]:


def CMF(data,ndays=20):
    data['MF Multiplier'] = (2*data['Close']-data['Low']-data['High'])/(data['High']-data['Low'])
    data['MF Volume'] = data['MF Multiplier']*data['Volume']
    data[str(ndays)+'_CMF'] = data['MF Volume'].rolling(ndays).sum()/data['Volume'].rolling(ndays).sum()
    data = data.drop(['MF Multiplier','MF Volume'],axis=1)
    return data
# cmf_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# cmf = CMF(cmf_data)
# cmf.head(25)


# ### Chaikin Oscillator
# 结合价格和交易量来显示资金如何流入或流出股票。基于累积/分配线。
# 1. Money Flow Multiplier = [(Close  -  Low) - (High - Close)] /(High - Low) 
# 
# 2. Money Flow Volume = Money Flow Multiplier x Volume for the Period
# 
# 3. ADL = Previous ADL + Current Period's Money Flow Volume
# 
# 4. Chaikin Oscillator = (3-day EMA of ADL)  -  (10-day EMA of ADL)	

# In[37]:


def ChaikinOscillator(data):
    data = ADL(data)
    data['ADL 3_EMA'] = data['Accu-Dist Line'].ewm(ignore_na=False,span=3,min_periods=2,adjust=True).mean()
    data['ADL 10_EMA'] = data['Accu-Dist Line'].ewm(ignore_na=False,span=10,min_periods=9,adjust=True).mean()
    data['ChaikinOscillator'] = data['ADL 3_EMA'] - data['ADL 10_EMA']
    data = data.drop(['Accu-Dist Line','ADL 3_EMA','ADL 10_EMA'],axis=1)
    return data
# co_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# co = ChaikinOscillator(co_data)
# co.head(20)


# ### Commodity Channel Index (CCI)
# 显示股票相对于其典型价格的变动。商品频道指数（CCI）是一个多功能的指标，可以用来确定一个新的趋势或警告极端的条件。
# 计算公式：
# CCI = (Typical Price  -  20-period SMA of TP) / (.015 x Mean Deviation)
# 
# Typical Price (TP) = (High + Low + Close)/3
# 
# Constant = .015
# 
# There are four steps to calculating the Mean Deviation: 
# First, subtract the most recent 20-period average of the typical price from each period's typical price. 
# Second, take the absolute values of these numbers. 
# Third, sum the absolute values. 
# Fourth, divide by the total number of periods (20). 

# In[39]:


def CCI(data,ndays=20):
    data['TP'] = (data['High']+data['Low']+data['Close'])/3
    data[str(ndays)+'_SMA of TP'] = data['TP'].rolling(ndays).mean()
    #暂时以std代替md，待有余力再写这个算法
    data[str(ndays)+'_MAD of TP'] = data['TP'].rolling(ndays).std()
    data['CCI'] = data[str(ndays)+'_SMA of TP']/(0.015*data[str(ndays)+'_MAD of TP'])
    data = data.drop(['TP',str(ndays)+'_SMA of TP',str(ndays)+'_MAD of TP'],axis=1)
    return data
# cci_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# cci = CCI(cci_data)
# cci.head(25)


# ### Rate of Change (ROC)
# 显示股票价格变化的速度。通常也叫作Momentum动量，是一个纯粹的动量振荡器，用来衡量一个时期的价格变化百分比。
# 计算公式：
# ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100

# In[41]:


def ROC(data,ndays):
    data[str(ndays)+'-day ROC'] = ((data['Close'] - data['Close'].shift(ndays))/data['Close'].shift(ndays)) * 100
    return data
# roc_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# roc = ROC(roc_data,10)
# roc.head(15)


# ### Coppock Curve
# 一个振荡器，使用变化率和加权移动平均来衡量动量。
# 计算公式：
# Coppock Curve = 10-period WMA of 14-period RoC + 11-perod RoC
# 
# WMA = Weighted moving average
# RoC = Rate-of-Change

# In[42]:


def CoppockCurve(data):
    ROC(data,11)
    ROC(data,14)
    data['Coppock Curve'] = (data['14-day ROC']+data['11-day ROC']).ewm(ignore_na=False,span=10,min_periods=10,adjust=True).mean()
    data = data.drop(['11-day ROC','14-day ROC'],axis=1)
    return data
# cc_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# cc = CoppockCurve(cc_data)
# cc.head(30)


# ### Correlation Coefficient
# 显示给定时间范围内两种证券之间的相关程度。系数为正表明证券移动往同一个方向移动，为负表明往相反方向移动。
# 相关系数的计算比较繁琐，跟两种证券的方差、协方差都有关系，这里直接用现成的公式计算。

# In[10]:


def CorrCoef(s1,s2):
    coef = s1.corr(s2)
    return coef
# wk_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# pa_data = web.get_data_yahoo('000001.sz',start='1/1/2010',end='1/1/2016')
# cc = CorrCoef(pa_data['Close'],wk_data['Close'])
# cc


# ### DecisionPoint Price Momentum Oscillator (PMO)
# 基于ROC的振荡器。
# 计算公式：
# 参考EMA的计算方式对ROC求两次平滑，不同的是计算平滑乘数直接用2/Time Period而不是2/(Time Period+1)，这里为了方便计算用EMA替代。
# 1、首先求出ROC
# 2、求ROC的35天平滑SMF(today)=（ROC-SMF(previous day)*(2/35)+SFM(previous)
# 3、对于第二步求出来的平滑序列继续求20天平滑
# 4、对于第三步求出来的平滑序列去10天EMA

# In[43]:


def PMO(data):
    data = ROC(data,1)
    data['35_EMA ROC'] = data['1-day ROC'].ewm(ignore_na=False,span=35,min_periods=0,adjust=True).mean()
    data['35_EMA ROC *10'] = data['35_EMA ROC'] * 10
    data['PMO Line'] = data['35_EMA ROC *10'].ewm(ignore_na=False,span=20,min_periods=0,adjust=True).mean()
    data['PMO Signal Line'] = data['PMO Line'].ewm(ignore_na=False,span=10,min_periods=0,adjust=True).mean()
    data = data.drop(['1-day ROC','35_EMA ROC','35_EMA ROC *10'],axis=1)
    return data
# wk_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# pmo = PMO(wk_data)
# pmo.head(30)


# ### Detrended Price Oscillator (DPO)
# 价格振荡器，使用错位的移动平均来确定周期。
# Price {X/2 + 1} periods ago less the X-period simple moving average. DPO(20)指用11天前的价格减去20天的移动平均。

# In[47]:


def DPO(data,ndays):
    data = SMA(data,ndays)
    shift = ndays/2+1
    data[str(shift)+'_shift'] = data['Close'].shift(shift)
    data['DPO'] = data[str(shift)+'_shift'] - data[str(ndays)+'_SMA']
    data = data.drop([str(ndays)+'_SMA',str(shift)+'_shift'],axis=1)
    return data
# wk_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# dpo = DPO(wk_data,20)
# dpo.head(30)


# ### Ease of Movement
# 一个指标，比较数量和价格，以确定重大举措。
# 计算公式：
# Distance Moved = ((H + L)/2 - (Prior H + Prior L)/2) 
# 
# Box Ratio = ((V/100,000,000)/(H - L))
# 
# 1-Period EMV = ((H + L)/2 - (Prior H + Prior L)/2) / ((V/100,000,000)/(H - L))
# 
# 14-Period Ease of Movement = 14-Period simple moving average of 1-period EMV

# In[48]:


def EMV(data,ndays=14):
    data['Distance Moved'] = (data['High']+data['Low'])/2 - (data['High'].shift()+data['Low'].shift())/2
    data['Box Ratio'] = (data['Volume']/100000000)/(data['High']-data['Low'])
    data['1-Period EMV'] = data['Distance Moved']/data['Box Ratio']
    data[str(ndays)+'-Period EMV'] = data['1-Period EMV'].rolling(ndays).mean()
    data = data.drop(['Distance Moved','Box Ratio','1-Period EMV'],axis=1)
    return data
# wk_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# emv = EMV(wk_data,20)
# emv.head(25)


# ### Force Index
# 一个简单的价-量振荡器。力量指数是一个使用价格和数量来评估移动背后的力量或识别可能的转折点的指标。
# 计算公式：
# Force Index(1) = {Close (current period)  -  Close (prior period)} x Volume
# 
# Force Index(13) = 13-period EMA of Force Index(1)

# In[49]:


def ForceIndex(data,ndays=13):
    data['1-Period FI'] = (data['Close'] - data['Close'].shift())*data['Volume']
    data[str(ndays)+'-Period FI'] = data['1-Period FI'].ewm(ignore_na=False,span=ndays,min_periods=ndays,adjust=True).mean()
    del data['1-Period FI']
    return data
# wk_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# fi = ForceIndex(wk_data)
# fi.head(20)


# ### Mass Index
# 当价格范围扩大时，指示逆转。
# 计算公式：
# Single EMA = 9-period exponential moving average (EMA) of the high-low differential  
# 
# Double EMA = 9-period EMA of the 9-period EMA of the high-low differential 
# 
# EMA Ratio = Single EMA divided by Double EMA 
# 
# Mass Index = 25-period sum of the EMA Ratio 

# In[52]:


def MassIndex(data):
    data['Single EMA'] = (data['High']-data['Low']).ewm(ignore_na=False,span=9,min_periods=9,adjust=True).mean()
    data['Double EMA'] = data['Single EMA'].ewm(ignore_na=False,span=9,min_periods=9,adjust=True).mean()
    data['EMA Ratio'] = data['Single EMA']/data['Double EMA']
    data['Mass Index'] = data['EMA Ratio'].ewm(ignore_na=False,span=25,min_periods=25,adjust=True).mean()
    data = data.drop(['Single EMA','Double EMA','EMA Ratio'],axis=1)
    return data
# wk_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# mi = MassIndex(wk_data)
# mi.head(50)


# ### MACD (Moving Average Convergence/Divergence Oscillator)
# 基于两个EMA之间的差异的动量振荡器。
# MACD Line: (12-day EMA - 26-day EMA)
# 
# Signal Line: 9-day EMA of MACD Line
# 
# MACD Histogram: MACD Line - Signal Line

# In[53]:


def MACD(data):
    data['MACD'] = EMA(data,12)['12_EMA']-EMA(data,26)['26_EMA']
    data['Signal Line'] = data['MACD'].ewm(ignore_na=False,span=9,min_periods=9,adjust=True).mean()
    data['MACD Histogram'] = data['MACD'] - data['Signal Line']
    data = data.drop(['12_EMA','26_EMA'],axis=1)
    return data
# wk_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# macd = MACD(wk_data)
# macd.head(40)


# ### Money Flow Index (MFI)
# 成交量加权版的RSI，显示买压和卖压。
# 计算公式：
# Typical Price = (High + Low + Close)/3
# 
# Raw Money Flow = Typical Price x Volume
# 
# Money Flow Ratio = (14-period Positive Money Flow)/(14-period Negative Money Flow)
# 
# Money Flow Index = 100 - 100/(1 + Money Flow Ratio)

# In[54]:


def MFI(data,ndays=14):
    data['Typical Price'] = (data['High']+data['Low']+data['Close'])/3
    def is_positive(x):
        if x>=0:
            return 1
        elif x<0:
            return -1
    data['Up or Down'] = (data['Typical Price'] - data['Typical Price'].shift()).apply(is_positive)
    data['Raw Money Flow'] = data['Typical Price']*data['Volume']*abs(data['Up or Down'])
    data['1-period Positive MF'] = 0.00
    data['1-period Negative MF'] = 0.00
    df1 = data[data['Up or Down']==1.0]
    df2 = data[data['Up or Down']==-1.0]
    df1.loc[:,'1-period Positive MF'] = df1['Raw Money Flow']
    df2.loc[:,'1-period Negative MF'] = df2['Raw Money Flow']
    data[data['Up or Down']==1.0] = df1
    data[data['Up or Down']==-1.0] = df2
    data[str(ndays)+'-period PMF'] = data['1-period Positive MF'].rolling(ndays).sum()
    data[str(ndays)+'-period NMF'] = data['1-period Negative MF'].rolling(ndays).sum()
    data[str(ndays)+'-period MF Ratio'] = data[str(ndays)+'-period PMF']/data[str(ndays)+'-period NMF']
    data[str(ndays)+'-period MF Index'] = 100 - 100/(1+data[str(ndays)+'-period MF Ratio'])
    data = data.drop(['Typical Price','Up or Down','Raw Money Flow','1-period Positive MF','1-period Negative MF',                      str(ndays)+'-period PMF',str(ndays)+'-period NMF',str(ndays)+'-period MF Ratio'],axis=1)
    return data
# wk_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# mfi = MFI(wk_data)
# mfi.head(20)


# ### Negative Volume Index (NVI)
# 用于识别趋势反转的基于累计量的指标。
# 计算公式：
# 1、计算1-period Close的ROC，Price Change Rate
# 2、计算1-period Volume的ROC,Volume Change Rate
# 3、如果Volume的ROC是负的，NVI取Price Change Rate，否则取0
# 4、初始的NVI为1000，计算累加NVI

# In[55]:


def NVI(data,ndays=1):
    data['ROC Price'] = ROC(data,ndays)[str(ndays)+'-day ROC']
    data['ROC Volume'] = ((data['Volume'] - data['Volume'].shift(ndays))/data['Volume'].shift(ndays)) * 100
    data['NVI Value'] = 0
    data['NVI Cumulative'] = 0
    df1 = data[data['ROC Volume']<0]
    df1['NVI Value'] = df1['ROC Price']
    data[data['ROC Volume']<0] = df1
    data['NVI Cumulative'] = 1000+data['NVI Value'].cumsum()
    data = data.drop(['1-day ROC','ROC Price','ROC Volume'],axis=1)
    return data
# wk_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# nvi = NVI(wk_data)
# nvi.head(20)


# ### On Balance Volume (OBV)
# 平衡量（OBV）衡量买入和卖出压力作为一个累积指标，在上涨的日子里增加交易量，在下跌的日子里减掉交易量。
# 计算公式：
# 1、计算当前日期相对于上一个日期上涨还是下跌，上涨记为1，下跌记为-1，相等记为0
# 2、第一步得到的值乘以Volume，得到一个带方向的Volume
# 3、计算第二步的累加值

# In[58]:


def OBV(data):
    def is_positive(x):
        if x>0:
            return 1
        elif x<0:
            return -1
        else:
            return 0
    data['Up or Down'] = (data['Close'] - data['Close'].shift()).apply(is_positive)
    data['Volume Directed'] = data['Up or Down'] * data['Volume']
    data['OBV'] = data['Volume Directed'].cumsum()
    data = data.drop(['Up or Down','Volume Directed'],axis=1)
    return data
# wk_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# obv = OBV(wk_data)
# obv.head(20)


# ### Percentage Price Oscillator (PPO)
# MACD指标的基于百分比的版本。
# 计算公式：
# Percentage Price Oscillator (PPO): {(12-day EMA - 26-day EMA)/26-day EMA} x 100
# 
# Signal Line: 9-day EMA of PPO
# 
# PPO Histogram: PPO - Signal Line

# In[59]:


def PPO(data):
    data['PPO'] = ((EMA(data,12)['12_EMA']-EMA(data,26)['26_EMA'])/EMA(data,26)['26_EMA']) * 100
    data['Signal Line'] = data['PPO'].ewm(ignore_na=False,span=9,min_periods=9,adjust=True).mean()
    data['PPO Histogram'] = data['PPO'] - data['Signal Line']
    data = data.drop(['12_EMA','26_EMA'],axis=1)
    return data
# wk_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# ppo = PPO(wk_data)
# ppo.head(40)


# ### Percentage Volume Oscillator (PVO)
# PPO指标适用于数量而不是价格。
# Percentage Volume Oscillator (PVO): 
# 
# ((12-day EMA of Volume - 26-day EMA of Volume)/26-day EMA of Volume) x 100
# 
# Signal Line: 9-day EMA of PVO
# 
# PVO Histogram: PVO - Signal Line

# In[62]:


def PVO(data):
    data['PVO'] = ((EMAV(data,12)['12_EMA']-EMAV(data,26)['26_EMA'])/EMAV(data,26)['26_EMA']) * 100
    data['Signal Line'] = data['PVO'].ewm(ignore_na=False,span=9,min_periods=9,adjust=True).mean()
    data['PVO Histogram'] = data['PVO'] - data['Signal Line']
    data = data.drop(['12_EMA','26_EMA'],axis=1)
    return data
# wk_data = web.get_data_yahoo('000002.sz',start='1/1/2010',end='1/1/2016')
# pvo = PVO(wk_data)
# pvo.head(40)


# ### Price Relative / Relative Strength
# 技术指标，通过除以价格数据来比较两只股票的表现。
# 计算公式：
# Price Relative = Base Security / Comparative Security
# 
# Ratio Symbol Close = Close of First Symbol / Close of Second Symbol
# 
# Ratio Symbol Open  = Open of First Symbol / Close of Second Symbol
# 
# Ratio Symbol High  = High of First Symbol / Close of Second Symbol
# 
# Ratio Symbol Low   = Low of First Symbol / Close of Second Symbol

# In[78]:


def PriceRelative(data1,stock_name1,data2,stock_name2):
    data = DataFrame([])
    data[stock_name1] = data1['Close']
    data[stock_name2] = data2['Close']
    data['Price Relative'] = data[stock_name1]/data[stock_name2]
    data['Percentage Change in Price Relative'] = ((data['Price Relative']-data['Price Relative'].shift())/data['Price Relative'].shift())*100
    return data
# sb_data = web.get_data_yahoo('SBUX',start='1/1/2010',end='1/1/2016')
# sp_data = web.get_data_yahoo('^GSPC',start='1/1/2010',end='1/1/2016')
# pr = PriceRelative(sb_data,'SBUX',sp_data,'S&P 500')
# pr.head(10)


# ### Know Sure Thing (KST)
# 马丁·普林基于四个不同时间框架的平滑变化率的动量振荡器。
# 计算公式：
# RCMA1 = 10-Period SMA of 10-Period Rate-of-Change 
# RCMA2 = 10-Period SMA of 15-Period Rate-of-Change 
# RCMA3 = 10-Period SMA of 20-Period Rate-of-Change 
# RCMA4 = 15-Period SMA of 30-Period Rate-of-Change 
# 
# KST = (RCMA1 x 1) + (RCMA2 x 2) + (RCMA3 x 3) + (RCMA4 x 4)  
# 
# Signal Line = 9-period SMA of KST

# In[66]:


def KST(data):
    data['10-ROC'] = ROC(data,ndays=10)['10-day ROC']
    data['15-ROC'] = ROC(data,ndays=15)['15-day ROC']
    data['20-ROC'] = ROC(data,ndays=20)['20-day ROC']
    data['30-ROC'] = ROC(data,ndays=30)['30-day ROC']
    data['RCMA1'] = data['10-ROC'].rolling(10).mean()
    data['RCMA2'] = data['15-ROC'].rolling(10).mean()
    data['RCMA3'] = data['20-ROC'].rolling(10).mean()
    data['RCMA4'] = data['30-ROC'].rolling(10).mean()
    data['KST'] = data['RCMA1']*1 + data['RCMA2']*2 + data['RCMA3']*3 + data['RCMA4']*4
    data['Signal Line'] = data['KST'].rolling(9).mean()
    data = data.drop(['10-ROC','15-ROC','20-ROC','RCMA1','RCMA2','RCMA3','RCMA4'],axis=1)
    return df
# sb_data = web.get_data_yahoo('SBUX',start='1/1/2010',end='1/1/2016')
# sb = KST(sb_data)
# sb.head(60)


# ### Pring's Special K
# 马丁·普林基于四个不同时间框架的平滑变化率的动量振荡器。
# Special K = 10 Period Simple Moving Average of ROC(10) * 1
#             + 10 Period Simple Moving Average of ROC(15) * 2
#             + 10 Period Simple Moving Average of ROC(20) * 3
#             + 15 Period Simple Moving Average of ROC(30) * 4
#             + 50 Period Simple Moving Average of ROC(40) * 1
#             + 65 Period Simple Moving Average of ROC(65) * 2
#             + 75 Period Simple Moving Average of ROC(75) * 3
#             +100 Period Simple Moving Average of ROC(100)* 4
#             +130 Period Simple Moving Average of ROC(195)* 1
#             +130 Period Simple Moving Average of ROC(265)* 2
#             +130 Period Simple Moving Average of ROC(390)* 3
#             +195 Period Simple Moving Average of ROC(530)* 4

# In[ ]:


def SpecialK(data):
    data['10-ROC'] = ROC(data,ndays=10)['10-day ROC']
    data['15-ROC'] = ROC(data,ndays=15)['15-day ROC']
    data['20-ROC'] = ROC(data,ndays=20)['20-day ROC']
    data['30-ROC'] = ROC(data,ndays=30)['30-day ROC']
    data['40-ROC'] = ROC(data,ndays=10)['40-day ROC']
    data['65-ROC'] = ROC(data,ndays=15)['65-day ROC']
    data['75-ROC'] = ROC(data,ndays=20)['75-day ROC']
    data['100-ROC'] = ROC(data,ndays=30)['100-day ROC']
    data['195-ROC'] = ROC(data,ndays=10)['195-day ROC']
    data['265-ROC'] = ROC(data,ndays=15)['265-day ROC']
    data['390-ROC'] = ROC(data,ndays=20)['390-day ROC']
    data['530-ROC'] = ROC(data,ndays=30)['530-day ROC']
    data['Spicial K'] = data['10-ROC'].rolling(10).mean() \
    + 2*data['15-ROC'].rolling(10).mean() \
    + 3*data['20-ROC'].rolling(10).mean() \
    + 4*data['30-ROC'].rolling(15).mean() \
    + 1*data['40-ROC'].rolling(50).mean() \
    + 2*data['60-ROC'].rolling(65).mean() \
    + 3*data['75-ROC'].rolling(75).mean() \
    + 4*data['100-ROC'].rolling(100).mean() \
    + 1*data['195-ROC'].rolling(130).mean() \
    + 2*data['265-ROC'].rolling(130).mean() \
    + 3*data['390-ROC'].rolling(130).mean() \
    + 4*data['530-ROC'].rolling(195).mean()
    return data


# ### Relative Strength Index (RSI)
# 显示股票目前的走势强弱。
# 计算公式：
#                   100
#         RSI = 100 - --------
#                  1 + RS
# 
#         RS = Average Gain / Average Loss
#  如果今日价格比昨天高，取差价放入对应的Gain列，否则把差价放入对应的Loss列。Gain列和Loss的其他列填充0，然后分别计算14-period的SMA。

# In[71]:


def RSI(data,ndays=14):
    data['Change'] = data['Close'] - data['Close'].shift()
    data['Gain'] = 0
    data['Loss'] = 0
    df1 = data[data['Change']>0]
    df2 = data[data['Change']<0]
    df1['Gain'] = df1['Change']
    df2['Loss'] = df2['Change']
    data[data['Change']>0] = df1
    data[data['Change']<0] = df2
    data['Avg Gain'] = data['Gain'].rolling(ndays).mean()
    data['Avg Loss'] = data['Loss'].rolling(ndays).mean()
    data['RS'] = data['Avg Gain']/data['Avg Loss']
    data[str(ndays)+'-day RSI'] = 100 - 100/(1+data['RS'])
    data = data.drop(['Change','Gain','Loss','Avg Gain','Avg Loss'],axis=1)
    return data
# rsi_data = web.get_data_yahoo('SBUX',start='1/1/2010',end='1/1/2016')
# rsi = RSI(rsi_data)
# rsi.head(20)


# ### Standard Deviation (Volatility)
# 股票波动的统计量度。
# StockCharts.com计算一个总体的标准偏差，它假定所涉及的周期代表整个数据集，而不是来自更大数据集的样本。
# 计算方式：
# 1、计算均值
# 2、计算每个值和均值的偏差
# 3、计算偏差的平方
# 4、偏差的平方相加后除以总数
# 5、求第4步的平方根
# 以上就是一个标注的求标准差的过程。

# In[3]:


def SD(data,ndays):
    data['SD'] = data['Close'].rolling(ndays).std()
    return data
# sd_data = web.get_data_yahoo('SBUX',start='1/1/2015',end='31/1/2015')
# sd = SD(sd_data,10)
# sd.head(20)


# ### Stochastic Oscillator
# 显示股票的价格相对于过去的走势。反映了在一定周期内相对于高低区的位置。
# 计算公式：
# %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
# %D = 3-day SMA of %K
# 
# Lowest Low = lowest low for the look-back period
# Highest High = highest high for the look-back period
# %K is multiplied by 100 to move the decimal point two places

# In[72]:


def StochasticOscillator(data,ndays=14):
    data['Hightest High('+str(ndays)+')'] = data['High'].rolling(ndays).max()
    data['Lowest Low('+str(ndays)+')'] = data['Low'].rolling(ndays).min()
    data['%K'] = 100*(data['Close']-data['Lowest Low('+str(ndays)+')'])/(data['Hightest High('+str(ndays)+')']-data['Lowest Low('+str(ndays)+')'])
    data['%D'] = data['%K'].rolling(3).mean()
    data = data.drop(['Hightest High('+str(ndays)+')','Lowest Low('+str(ndays)+')'],axis=1)
    return data
# sd_data = web.get_data_yahoo('SBUX',start='1/1/2015',end='31/1/2015')
# sd = StochasticOscillator(sd_data)
# sd.head(20)


# ### StochRSI
# 将随机指标与RSI指标相结合，可以帮助您更清楚地看到RSI的变化。
# 计算公式：
# StochRSI = (RSI - Lowest Low RSI) / (Highest High RSI - Lowest Low RSI)

# In[77]:


def StochRSI(data,ndays=14):
    data = RSI(data,ndays)
    data['Highest High('+str(ndays)+')'] = data[str(ndays)+'-day RSI'].rolling(ndays).max()
    data['Lowest Low('+str(ndays)+')'] = data[str(ndays)+'-day RSI'].rolling(ndays).min()
    data['StochRSI('+str(ndays)+')'] =         (data[str(ndays)+'-day RSI']-data['Lowest Low('+str(ndays)+')'])/(data['Highest High('+str(ndays)+')']-data['Lowest Low('+str(ndays)+')'])
    data = data.drop(['RS',str(ndays)+'-day RSI','Highest High('+str(ndays)+')','Lowest Low('+str(ndays)+')'],axis=1)
    return data
# srsi_data = web.get_data_yahoo('SBUX',start='1/1/2015',end='1/1/2016')
# srsi = StochRSI(srsi_data)
# srsi.head(40)


# ### TRIX
# 价格走势的三重平滑移动平均线。
# 计算公式：
# 1. Single-Smoothed EMA = 15-period EMA of the closing price
# 2. Double-Smoothed EMA = 15-period EMA of Single-Smoothed EMA
# 3. Triple-Smoothed EMA = 15-period EMA of Double-Smoothed EMA
# 4. TRIX = 1-period percent change in Triple-Smoothed EMA

# In[78]:


def TRIX(data,ndays=15):
    data['SS EMA'] = data['Close'].rolling(ndays).mean()
    data['DS EMA'] = data['SS EMA'].rolling(ndays).mean()
    data['TS EMA'] = data['DS EMA'].rolling(ndays).mean()
    data['TRIX(%)'] = 100*(data['TS EMA']-data['TS EMA'].shift())/data['TS EMA'].shift()
    data = data.drop(['SS EMA','DS EMA','TS EMA'],axis=1)
    return data
# trix_data = web.get_data_yahoo('SBUX',start='1/1/2015',end='1/1/2016')
# trix = TRIX(trix_data)
# trix.head(50)


# ### True Strength Index (TSI)
# 衡量趋势方向的指标，确定超买/超卖水平。
# 计算公式：
# Double Smoothed PC
# ------------------
# PC = Current Price minus Prior Price
# First Smoothing = 25-period EMA of PC
# Second Smoothing = 13-period EMA of 25-period EMA of PC
# 
# Double Smoothed Absolute PC
# ---------------------------
# Absolute Price Change |PC| = Absolute Value of Current Price minus Prior Price
# First Smoothing = 25-period EMA of |PC|
# Second Smoothing = 13-period EMA of 25-period EMA of |PC|
# 
# TSI = 100 x (Double Smoothed PC / Double Smoothed Absolute PC)

# In[79]:


def TSI(data):
    data['PC'] = data['Close'] - data['Close'].shift()
    data['PC-FS'] = data['PC'].ewm(ignore_na=False,span=25,min_periods=25,adjust=True).mean()
    data['PC-SS'] = data['PC-FS'].ewm(ignore_na=False,span=13,min_periods=13,adjust=True).mean()
    data['Absolute PC'] = abs(data['Close'] - data['Close'].shift())
    data['Absolute PC-FS'] = data['Absolute PC'].ewm(span=25,min_periods=25).mean()
    data['ABsolute PC-SS'] = data['Absolute PC-FS'].ewm(span=13,min_periods=13).mean()
    data['TSI'] = 100* data['PC-SS']/data['ABsolute PC-SS']
    data = data.drop(['PC','PC-FS','PC-SS','Absolute PC','Absolute PC-FS','ABsolute PC-SS'],axis=1)
    return data
# tsi_data = web.get_data_yahoo('SBUX',start='1/1/2015',end='1/1/2016')
# tsi = TSI(tsi_data)
# tsi.head(50)


# ### Ulcer Index
# 设计用于衡量市场风险或波动的指标。
# 计算公式：
# Percent-Drawdown = ((Close - 14-period Max Close)/14-period Max Close) x 100
# 
# Squared Average = (14-period Sum of Percent-Drawdown Squared)/14 
# 
# Ulcer Index = Square Root of Squared Average

# In[80]:


def UlcerIndex(data,ndays=14):
    data[str(ndays)+'-period Max Close'] = data['Close'].rolling(ndays).max()
    data['Percent-Drawdown'] = 100* (data['Close']-data[str(ndays)+'-period Max Close'])/data[str(ndays)+'-period Max Close']
    data['Percent-Drawdown Squared'] = data['Percent-Drawdown'] ** 2
    data['Squared Average'] = data['Percent-Drawdown Squared'].rolling(ndays).sum()/14
    data['Ulcer Index'] = data['Squared Average'].pow(0.5)
    data = data.drop([str(ndays)+'-period Max Close','Percent-Drawdown','Percent-Drawdown Squared','Squared Average'],axis=1)
    return data
# ui_data = web.get_data_yahoo('SBUX',start='1/1/2015',end='1/1/2016')
# ui = UlcerIndex(ui_data)
# ui.head(30)


# ### Ultimate Oscillator
# 将长期，中期和短期移动平均线组合成一个数字。
# 计算公式：
# BP = Close - Minimum(Low or Prior Close).
#  
# TR = Maximum(High or Prior Close)  -  Minimum(Low or Prior Close)
# 
# Average7 = (7-period BP Sum) / (7-period TR Sum)
# Average14 = (14-period BP Sum) / (14-period TR Sum)
# Average28 = (28-period BP Sum) / (28-period TR Sum)
# 
# UO = 100 x [(4 x Average7)+(2 x Average14)+Average28]/(4+2+1)

# In[81]:


def UltimateOscillator(data):
    data['Prior Close'] = data['Close'].shift()
    data['BP'] = data['Close'] - data[['Low','Prior Close']].min(axis=1)
    data['TR'] = data[['High','Prior Close']].max(axis=1) - data[['Low','Prior Close']].min(axis=1)
    data['Average7'] = data['BP'].rolling(7).sum()/data['TR'].rolling(7).sum()
    data['Average14'] = data['BP'].rolling(14).sum()/data['TR'].rolling(14).sum()
    data['Average28'] = data['BP'].rolling(28).sum()/data['TR'].rolling(28).sum()
    data['U0'] = 100 * (4*data['Average7']+2*data['Average14']+data['Average28'])/(4+2+1)
    data = data.drop(['Prior Close','BP','TR','Average7','Average14','Average28'],axis=1)
    return data
# uo_data = web.get_data_yahoo('SBUX',start='1/1/2015',end='1/1/2016')
# uo = UltimateOscillator(uo_data)
# uo.head(30)


# ### Vortex Indicator
# 一个指标，旨在确定新趋势的开始，并确定当前的趋势。
# 计算公式：
# Positive and negative trend movement:
# 
# +VM = Current High less Prior Low (absolute value)
# -VM = Current Low less Prior High (absolute value)
# 
# +VM14 = 14-period Sum of +VM
# -VM14 = 14-period Sum of -VM
# 
# 
# True Range (TR) is the greatest of:
# 
#   * Current High less current Low
#   * Current High less previous Close (absolute value)
#   * Current Low less previous Close (absolute value)
# 
# TR14 = 14-period Sum of TR
# 
# 
# Normalize the positive and negative trend movements:
# 
# +VI14 = +VM14/TR14
# -VI14 = -VM14/TR14

# In[83]:


def VortexIndicator(data,ndays=14):
    data['Prior Low'] = data['Low'].shift()
    data['Prior High'] = data['High'].shift()
    data['+VM'] = abs(data['High'] - data['Prior Low'])
    data['-VM'] = abs(data['Low'] - data['Prior High'])
    data['+VM'+str(ndays)] = data['+VM'].rolling(ndays).sum()
    data['-VM'+str(ndays)] = data['-VM'].rolling(ndays).sum()
    data['HL'] = data['High']-data['Low']
    data['HC'] = abs(data['High']-data['Close'].shift())
    data['LC'] = abs(data['Low']-data['Close'].shift())
    data['TR'] = data[['HL','HC','LC']].max(axis=1)
    del data['HL']
    del data['HC']
    del data['LC']
    data['TR'+str(ndays)] = data['TR'].rolling(ndays).sum()
    data['+VI'+str(ndays)] = data['+VM'+str(ndays)]/data['TR'+str(ndays)]
    data['-VI'+str(ndays)] = data['-VM'+str(ndays)]/data['TR'+str(ndays)]
    data = data.drop(['Prior Low','Prior High','+VM','-VM','+VM14','-VM14','TR','TR14'],axis=1)
    return data
# Vi_data = web.get_data_yahoo('SBUX',start='1/1/2015',end='1/1/2016')
# Vi = VortexIndicator(Vi_data)
# Vi.head(20)


# ### Williams %R
# 使用随机指标来确定超买和超卖水平。
# 计算公式：
# %R = (Highest High - Close)/(Highest High - Lowest Low) * -100
# 
# Lowest Low = lowest low for the look-back period
# Highest High = highest high for the look-back period
# %R is multiplied by -100 correct the inversion and move the decimal.

# In[84]:


def WilliamR(data,ndays=14):
    data['Lowest Low'] = data['Low'].rolling(ndays).min()
    data['Highest High'] = data['High'].rolling(ndays).max()
    data['%R'] = -100*(data['Highest High'] - data['Close'])/(data['Highest High'] - data['Lowest Low'])
    data = data.drop(['Lowest Low','Highest High'],axis=1)
    return data
# WR_data = web.get_data_yahoo('SBUX',start='1/1/2015',end='1/1/2016')
# WR = WilliamR(WR_data)
# WR.head(20)


# In[ ]:




