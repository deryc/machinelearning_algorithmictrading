import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import statistics as stats

#Fetch data for 4 years, for 7 major currency pairs
TRADING_INSTRUMENT = 'CADUSD=X'
SYMBOLS = ['AUDUSD=X',
           'GBPUSD=X',
           'CADUSD=X',
           'CHFUSD=X',
           'EURUSD=X',
           'JPYUSD=X',
           'NZDUSD=X']
START_DATE = '2014-01-01'
END_DATE = '2018-01-01'

#Data series for each currency
symbols_data = {}
for symbol in SYMBOLS:
    SRC_DATA_FILENAME = symbol + '_data.pkl'
    try:
        Data = pd.read_pickle(SRC_DATA_FILENAME)
    except FileNotFoundError:
        Data = data.DataReader(symbol, 'yahoo', START_DATE, END_DATE)
        Data.to_pickle(SRC_DATA_FILENAME)
    symbols_data[symbol] = Data


#Visualized prices for currency to inspect relationship between them
cycol = cycle('bgrcmky')

price_data = pd.DataFrame()
for symbol in SYMBOLS:
    multiplier = 1.0
    if symbol == 'JPYUSD=X':
        multiplier = 100

    label = symbol + ' ClosePrice'
    price_data = price_data.assign(label=pd.Series(symbols_data[symbol]['Close'] * multiplier, index=symbols_data[symbol].index))
    ax = price_data['label'].plot(color=next(cycol), lw=2, label=label)

plt.xlabel('Date', fontsize=18)
plt.ylabel('Scaled Price', fontsize=18)
plt.legend(prop={'size':6})
plt.show()


#Constants/variables that are used to compute simple moving average and price deviation from simple moving average
SMA_NUM_PERIODS = 20 # look back period
price_history = {} # history of prices

PRICE_DEV_NUM_PRICES = 200 # look back period of ClosePrice deviations from SMA
price_deviation_from_sma = {} # history of ClosePrice deviations from SMA

#Use this to iterate over all the days of data
num_days = len(symbols_data[TRADING_INSTRUMENT].index)
correlation_history = {} # history of correlations per currency pair
delta_projected_actual_history = {} # history of differences between Projected ClosePrice deviation and actual ClosePrice deviation per currency pair
final_delta_projected_history = [] # history of differences between final Projected ClosePrice deviation from TRADING_INSTRUMENT and actual ClosePrice deviation

#Variables for Trading Strategy trade, position & pnl management
orders = [] # container for tracking buy/sell order, +1 for buy, -1 for sell, 0 for no action
positions = [] # container for tracking positions, positive for long, negative for short, 0 for flat
pnls = [] # container for tracking total_pnls, this is the sum of closed_pnl, (i.e. pnls already locked in) and open_pnl, (i.e. pnls for open-position marked to market price

last_buy_price = 0 # Price at which last buy trade was made, used to prevent over-trading at/around the same price
last_sell_price = 0 #ibbid
position = 0 #Current position
buy_sum_price_qty = 0 #Summation of products of buy_trade_price and buy_trade_qty for every buy trade since being flat
buy_sum_qty = 0 #Summation of buy_trade_qty for every buy trade made since last time being flat
sell_sum_price_qty = 0
sell_sum_qty = 0
open_pnl = 0 #Open/Unrealized PnL marked to market
closed_pnl = 0 # Closed/Realized PnL so far

# Constants that define strategy behavior/thresholds
StatArb_VALUE_FOR_BUY_ENTRY = 0.01 #StatArb trading signal value below which to enter buy-orders/long position
StatArb_VALUE_FOR_SELL_ENTRY = -0.01
MIN_PRICE_MOVE_FROM_LAST_TRADE = 0.01 #Minimum price change since last trade before considering trading again, this is to preven over-trading at/around same price
MIN_PROFIT_TO_CLOSE = 10 #Minimum Open/Unrealized profit at which to close positions and lock profits
NUM_SHARES_PER_TRADE = 1000000 #Number of buy/sell on every trade

for i in range(0, num_days):
    close_prices = {}
    # Build ClosePrice series, compute SMA for each symbol and price-deviation from SMA for each symbol
    for symbol in SYMBOLS:
        close_prices[symbol] = symbols_data[symbol]['Close'].iloc[i]
        if not symbol in price_history.keys():
            price_history[symbol] = []
            price_deviation_from_sma[symbol] = []
        price_history[symbol].append(close_prices[symbol])
        if len(price_history[symbol]) > SMA_NUM_PERIODS: # track at mose SMA_NUM_PERIODS number of prices
            del (price_history[symbol][0])

        sma = stats.mean(price_history[symbol]) # Rolling SimpleMovingAverage
        price_deviation_from_sma[symbol].append(close_prices[symbol] - sma) # price deviation from mean
        if len(price_deviation_from_sma[symbol]) > PRICE_DEV_NUM_PRICES:
            del (price_deviation_from_sma[symbol][0])

    # Now compute covariance and correlation between TRADING_INSTRUMENT and every other lead symbol
    # also compute projected price deviation and find delta between projected and actual price deviations
    projected_dev_from_sma_using = {}
    for symbol in SYMBOLS:
        if symbol == TRADING_INSTRUMENT:
            continue
        correlation_label = TRADING_INSTRUMENT + '<-' + symbol
        if correlation_label not in correlation_history.keys(): #first entry in dictionary
            correlation_history[correlation_label] = []
            delta_projected_actual_history[correlation_label] = []
        if len(price_deviation_from_sma[symbol]) < 2: #need at least two measurements for covariance and correlation
            correlation_history[correlation_label].append(0)
            delta_projected_actual_history[correlation_label].append(0)
            continue

        corr = np.corrcoef(price_deviation_from_sma[TRADING_INSTRUMENT],
                           price_deviation_from_sma[symbol])
        cov = np.cov(price_deviation_from_sma[TRADING_INSTRUMENT],
                     price_deviation_from_sma[symbol])
        corr_trading_instrument_lead_instrument = corr[0,1] #get correlation between two series
        cov_trading_instrument_lead_instrument = cov[0,0] / cov[0,1] #ibbid.

        correlation_history[correlation_label].append(corr_trading_instrument_lead_instrument)

        #projected-price-deviation-in-TRADING_INSTRUMENT is covariance * price-deviation-in-lead-symbol
        projected_dev_from_sma_using[symbol] = price_deviation_from_sma[symbol][-1] * cov_trading_instrument_lead_instrument

        #delta positive => signal says TRADING_INSTRUMENT price should have moved up more than what it did
        #delta negative => signal says TRADING_INSTRUMENT price should have moved down more than what it did
        delta_projected_actual = (projected_dev_from_sma_using[symbol] - price_deviation_from_sma[TRADING_INSTRUMENT][-1])
        delta_projected_actual_history[correlation_label].append(delta_projected_actual)

    #weigh predictions from each pair, weight is correlation between those pairs
    sum_weights = 0 # sum of weights is sum of correlations for each symbol with TRADING_INSTRUMENT
    for symbol in SYMBOLS:
        if symbol == TRADING_INSTRUMENT:
            continue
        correlation_label = TRADING_INSTRUMENT + '<-' + symbol
        sum_weights += abs(correlation_history[correlation_label][-1])

    final_delta_projected = 0 #will hold final prediction of price deviation in TRADING_INSTRUMENT, weighing projections from all other symbols
    close_price = close_prices[TRADING_INSTRUMENT]
    for symbol in SYMBOLS:
        if symbol == TRADING_INSTRUMENT:
            continue
        correlation_label = TRADING_INSTRUMENT + '<-' + symbol
        #weight projection from a symbol by correlation
        final_delta_projected += (abs(correlation_history[correlation_label][-1]) * delta_projected_actual_history[correlation_label][-1])

    #normalize by dividing by sum of weights for all pairs
    if sum_weights != 0:
        final_delta_projected /= sum_weights
    else:
        final_delta_projected = 0
    final_delta_projected_history.append(final_delta_projected)


    #This section contains the execution logic
    if ((final_delta_projected < StatArb_VALUE_FOR_SELL_ENTRY and abs(close_price - last_sell_price) > MIN_PRICE_MOVE_FROM_LAST_TRADE) or (position > 0 and (open_pnl > MIN_PROFIT_TO_CLOSE))):
        orders.append(-1)
        last_sell_price = close_price
        position -= NUM_SHARES_PER_TRADE
        sell_sum_price_qty += (close_price * NUM_SHARES_PER_TRADE)
        sell_sum_qty += NUM_SHARES_PER_TRADE

    elif ((final_delta_projected > StatArb_VALUE_FOR_BUY_ENTRY and abs(close_price - last_sell_price) > MIN_PRICE_MOVE_FROM_LAST_TRADE) or (position < 0 and (open_pnl > MIN_PROFIT_TO_CLOSE))):
        orders.append(+1)
        last_buy_price = close_price
        position += NUM_SHARES_PER_TRADE
        buy_sum_price_qty += (close_price * NUM_SHARES_PER_TRADE)
        buy_sum_qty += NUM_SHARES_PER_TRADE
    else:
        orders.append(0)
    positions.append(position)

    #This section updates Open/Unrealized & Closed/Realized positions
    open_pnl = 0
    if position > 0:
        if sell_sum_qty > 0: # long position and some sell trades have been made against it, close that amount based on how much was sold against this long position
            open_pnl = abs(sell_sum_qty) * (sell_sum_price_qty / sell_sum_qty - buy_sum_price_qty / buy_sum_qty)
            # mark the remaining postion to market, i.e. pnl would be what it would be if we closed at current price
            open_pnl += abs(sell_sum_qty - position) * (close_price - buy_sum_price_qty / buy_sum_qty)
    elif position < 0:
        if buy_sum_qty > 0: # short position and some buy trades have been made against it, close that amount based on how much was bough against this short position
            open_pnl = abs(buy_sum_qty) * (sell_sum_price_qty / sell_sum_qty - buy_sum_price_qty / buy_sum_qty)
            # mark remaining position to market, i.e. pnl would be what it would be if we closed at current price
            open_pnl += abs(buy_sum_qty - position) * (sell_sum_price_qty / sell_sum_qty - close_price)
    else:
        # flat, so update closed_pnl and reset tracking variables for positions & pnl
        closed_pnl += (sell_sum_price_qty - buy_sum_price_qty)
        buy_sum_price_qty = 0
        buy_sum_qty = 0
        sell_sum_price_qty = 0
        sell_sum_qty = 0
        last_buy_price = 0
        last_sell_price = 0
    pnls.append(closed_pnl + open_pnl)  


#Plot correlations between TRADING_INSTRUMENT and other currency pairs
correlation_data = pd.DataFrame()
for symbol in SYMBOLS:
    if symbol == TRADING_INSTRUMENT:
        continue
    
    correlation_label = TRADING_INSTRUMENT + '<-' + symbol
    correlation_data = correlation_data.assign(label=pd.Series(correlation_history[correlation_label], index=symbols_data[symbol].index))
    ax = correlation_data['label'].plot(color=next(cycol), lw=2, label='Correlation ' + correlation_label)
for i in np.arange(-1,1,0.25):
    plt.axhline(y=i, lw=0.5, color='k')
plt.legend()
plt.show()

#Plot StatArb signal provided by each currency pair
delta_projected_actual_data = pd.DataFrame()
for symbol in SYMBOLS:
    if symbol == TRADING_INSTRUMENT:
        continue

    projection_label = TRADING_INSTRUMENT + '<-' + symbol
    delta_projected_actual_data = delta_projected_actual_data.assign(StatArbTradingSignal=pd.Series(delta_projected_actual_history[projection_label], index=symbols_data[TRADING_INSTRUMENT].index))
    ax = delta_projected_actual_data['StatArbTradingSignal'].plot(color=next(cycol), lw=1, label='StatArbTradingSignal ' + projection_label)
plt.legend()
plt.show()

#Create dataframe for visualization of close_price, trades, positions, and PnLs.
delta_projected_actual_data = delta_projected_actual_data.assign(ClosePrice=pd.Series(symbols_data[TRADING_INSTRUMENT]['Close'], index=symbols_data[TRADING_INSTRUMENT].index))
delta_projected_actual_data = delta_projected_actual_data.assign(FinalStatArbTradingSignal=pd.Series(final_delta_projected_history, index=symbols_data[TRADING_INSTRUMENT].index))
delta_projected_actual_data = delta_projected_actual_data.assign(Trades=pd.Series(orders, index=symbols_data[TRADING_INSTRUMENT].index))
delta_projected_actual_data = delta_projected_actual_data.assign(Position=pd.Series(positions, index=symbols_data[TRADING_INSTRUMENT].index))
delta_projected_actual_data = delta_projected_actual_data.assign(Pnl=pd.Series(pnls, index=symbols_data[TRADING_INSTRUMENT].index))

#Plot Market Price
plt.plot(delta_projected_actual_data.index, delta_projected_actual_data.ClosePrice, color='k', lw=1, label='ClosePrice')
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Trades == 1].index, delta_projected_actual_data.ClosePrice[delta_projected_actual_data.Trades == 1],
         color='r', lw=0, marker='^', markersize=7, label='buy')
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Trades == -1].index, delta_projected_actual_data.ClosePrice[delta_projected_actual_data.Trades == -1],
         color='g', lw=0, marker='v', markersize=7, label='sell')
plt.legend()
plt.show()


#Plot Trading Signal
plt.plot(delta_projected_actual_data.index, delta_projected_actual_data.FinalStatArbTradingSignal, color='k', lw=1, label='FinalStatArbTradingSignal')
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Trades == 1].index, delta_projected_actual_data.FinalStatArbTradingSignal[delta_projected_actual_data.Trades == 1],
         color='r', lw=0, marker='^', markersize=7, label='buy')
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Trades == -1].index, delta_projected_actual_data.FinalStatArbTradingSignal[delta_projected_actual_data.Trades == -1],
         color='g', lw=0, marker='v', markersize=7, label='sell')
plt.axhline(y=0, lw=0.5, color='k')
for i in np.arange(StatArb_VALUE_FOR_BUY_ENTRY, StatArb_VALUE_FOR_BUY_ENTRY * 10, StatArb_VALUE_FOR_BUY_ENTRY * 2):
    plt.axhline(y=i, lw=0.5, color='r')
for i in np.arange(StatArb_VALUE_FOR_SELL_ENTRY, StatArb_VALUE_FOR_SELL_ENTRY * 10, StatArb_VALUE_FOR_SELL_ENTRY * 2):
    plt.axhline(y=i, lw=0.5, color='g')
plt.legend()
plt.show()

#Plot Positions
plt.plot(delta_projected_actual_data.index, delta_projected_actual_data.Position, color='k', lw=1, label='Position')
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Position == 0].index,
         delta_projected_actual_data.Position[delta_projected_actual_data.Position == 0], color='k', lw=0, marker='.', label='flat')
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Position > 0].index,
         delta_projected_actual_data.Position[delta_projected_actual_data.Position > 0], color='r', lw=0, marker='+', label='long')
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Position < 0].index,
         delta_projected_actual_data.Position[delta_projected_actual_data.Position < 0], color='g', lw=0, marker='_', label='short')
plt.axhline(y=0, lw=0.5, color='k')
for i in range(NUM_SHARES_PER_TRADE, NUM_SHARES_PER_TRADE * 5, NUM_SHARES_PER_TRADE):
    plt.axhline(y=i, lw=0.5, color='r')
for i in range(-NUM_SHARES_PER_TRADE, -NUM_SHARES_PER_TRADE * 5, -NUM_SHARES_PER_TRADE):
    plt.axhline(y=i, lw=0.5, color='g')
plt.legend()
plt.show()

#Plot PnL

delta_projected_actual_data['Pnl'].plot(color='k', lw=1, legend=True)
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Pnl > 0].index,
         delta_projected_actual_data.Pnl[delta_projected_actual_data.Pnl > 0], color='g', lw=0, marker='.')
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Pnl < 0].index,
         delta_projected_actual_data.Pnl[delta_projected_actual_data.Pnl < 0], color='r', lw=0, marker='.')
plt.legend()
plt.show()


         


