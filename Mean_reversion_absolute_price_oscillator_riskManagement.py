import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt

import statistics as stats
from statistics import stdev
import math as math
import numpy as np

#Fetch daily data for 4 years
SYMBOL='GOOG'
start_date = '2014-01-01'
end_date = '2018-01-01'
SRC_DATA_FILENAME= SYMBOL + '_data.pkl'

try:
    data = pd.read_pickle(SRC_DATA_FILENAME)
except FileNotFoundError:
    data = data.DataReader(SYMBOL, 'yahoo', start_date, end_date)
    data.to_pickle(SRC_DATA_FILENAME)


#Variables for variable risk managment
MIN_NUM_SHARES_PER_TRADE = 1
MAX_NUM_SHARES_PER_TRADE = 50
INCREMENT_NUM_SHARES_PER_TRADE = 2
num_shares_per_trade = MIN_NUM_SHARES_PER_TRADE #beginning number of shares to buy or sell on every trade
num_shares_history = [] #history of num_shares
abs_position_history = [] #history of absolute-position

#Risk Limits
risk_limit_weekly_stop_loss = -20000
INCREMENT_RISK_LIMIT_WEEKLY_STOP_LOSS = -15000
risk_limit_monthly_stop_loss = -30000
INCREMENT_RISK_LIMIT_MONTHLY_STOP_LOSS = -30000
risk_limit_max_position = 5
INCREMENT_RISK_LIMIT_MAX_POSITION = 3
max_position_history = [] #history of max-position
RISK_LIMIT_MAX_POSITION_HOLDING_TIME_DAYS = 120 * 1.5
risk_limit_max_trade_size = 5
INCREMENT_RISK_LIMIT_MAX_TRADE_SIZE = 2
max_trade_size_history = [] #history of max-trade-size
RISK_LIMIT_MAX_TRADED_VOLUME = 5000 * 1.5

last_risk_change_index = 0

#Variables to track/check for risk violations
risk_violated = False

traded_volume = 0
current_pos = 0
current_pos_start = 0

#Variables/constants for EMA Calculation:

NUM_PERIODS_FAST = 10 #Static time period parameter for the fast EMA
K_FAST = 2 / (NUM_PERIODS_FAST + 1) #Static smoothing factor for fast EMA
ema_fast = 0
ema_fast_values = [] #Hold EMA values for visualization

NUM_PERIODS_SLOW = 40
K_SLOW = 2 / (NUM_PERIODS_SLOW + 1) #Static smoothing factor for slow EMA
ema_slow = 0
ema_slow_values = []

apo_values = [] # track computed absolute price oscillator value signals

# Variables for Trading Strategy trade, position, and pnl management
orders = [] #Container for tracking buy/sell order, +1 for buy, -1 for sell, 0 for no action
positions = [] #Container for tracking positions, positive for long, negative for short, 0 for flat
pnls = [] #Container for tracking total_pnls, sum of
#closed_pnl, i.e. pnls already locked in, and
#open_pnl, i.e. pnls for open-position marked to market price

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
APO_VALUE_FOR_BUY_ENTRY = -10 #APO trading signal value below which to enter buy-orders/long position
APO_VALUE_FOR_SELL_ENTRY = 10
MIN_PRICE_MOVE_FROM_LAST_TRADE = 10 #Minimum price change since last trade before considering trading again, this is to preven over-trading at/around same price
MIN_PROFIT_TO_CLOSE = num_shares_per_trade * 10 #Minimum Open/Unrealized profit at which to close positions and lock profits
NUM_SHARES_PER_TRADE = 1 #Number of buy/sell on every trade

# Constants/variables that are used to compute standard deviation as a volatility measure
SMA_NUM_PERIODS = 20 #look back period
price_history = [] #history of prices

close = data['Close']
for close_price in close:
    price_history.append(close_price)
    if len(price_history) > SMA_NUM_PERIODS: # we track at most 'time_period' number of prices
        del (price_history[0])
    sma = stats.mean(price_history)
    variance = 0 # variance is the square of the standard deviation
    for hist_price in price_history:
        variance = variance + ((hist_price - sma) ** 2)
    stdev = math.sqrt(variance / len(price_history))
    stdev_factor = stdev/15
    if stdev_factor == 0:
        stdev_factor = 1
    #This section updates fast and slow EMA and computes APO trading signal
    if (ema_fast == 0): #first observation
        ema_fast = close_price
        ema_slow = close_price
    else:
        ema_fast = (close_price - ema_fast) * K_FAST * stdev_factor + ema_fast
        ema_slow = (close_price - ema_slow) * K_SLOW * stdev_factor + ema_slow

    ema_fast_values.append(ema_fast)
    ema_slow_values.append(ema_slow)

    apo = ema_fast - ema_slow
    apo_values.append(apo)

    if NUM_SHARES_PER_TRADE > risk_limit_max_trade_size:
        print('RiskViolation NUM_SHARES_PER_TRADE', NUM_SHARES_PER_TRADE, ' > RISK_LIMIT_MAX_TRADE_SIZE', risk_limit_max_trade_size)
        risk_violated = True

    #APO above sell entry threshold, we should sell, or
    #long from negative APO and APO has gone positiive, or position is profitable, sell to close position
    if (not risk_violated and ((apo > APO_VALUE_FOR_SELL_ENTRY * stdev_factor and abs(close_price - last_sell_price) > MIN_PRICE_MOVE_FROM_LAST_TRADE * stdev_factor)\
                               or (position > 0 and (apo >= 0 or open_pnl > MIN_PROFIT_TO_CLOSE / stdev_factor)))):
        orders.append(-1) #mark sell trade
        last_sell_price = close_price
        #position -= NUM_SHARES_PER_TRADE #reduce postion by size of this trade
        if position == 0: #Opening a new position
            position -= num_shares_per_trade #reduce position by the size of the trade
            sell_sum_price_qty += (close_price * num_shares_per_trade)
            sell_sum_qty += num_shares_per_trade
            traded_volume += num_shares_per_trade
        else:
            sell_sum_price_qty += (close_price * abs(position))
            sell_sum_qty += abs(position)
            traded_volume += abs(position)
            position = 0
        #sell_sum_price_qty += (close_price * NUM_SHARES_PER_TRADE) #update vwap sell-price
        #sell_sum_qty += NUM_SHARES_PER_TRADE
        #traded_volume += NUM_SHARES_PER_TRADE
        #print('Sell ', NUM_SHARES_PER_TRADE, ' @ ', close_price, 'Position: ', position)

    #APO below buy entry threshold, we should buy, or
    #short from positive APO and APO has gone negative or position is profitable, buy to close position
    elif (not risk_violated and ((apo < APO_VALUE_FOR_BUY_ENTRY * stdev_factor and abs(close_price - last_buy_price) > MIN_PRICE_MOVE_FROM_LAST_TRADE * stdev_factor)\
                                 or  (position < 0 and (apo <= 0 or open_pnl > MIN_PROFIT_TO_CLOSE / stdev_factor)))):
        orders.append(+1) #mark the trade
        last_buy_price = close_price
        if position == 0: #Opening a new position
            position += num_shares_per_trade #reduce position by the size of the trade
            buy_sum_price_qty += (close_price * num_shares_per_trade)
            buy_sum_qty += num_shares_per_trade
            traded_volume += num_shares_per_trade
        else:
            buy_sum_price_qty += (close_price * abs(position))
            buy_sum_qty += abs(position)
            traded_volume += abs(position)
            position = 0
##        position += NUM_SHARES_PER_TRADE #increase position by size of trade
##        buy_sum_price_qty += (close_price * NUM_SHARES_PER_TRADE) #update vwap buy-price
##        buy_sum_qty += NUM_SHARES_PER_TRADE
##        traded_volume += NUM_SHARES_PER_TRADE
        #print('Buy ', NUM_SHARES_PER_TRADE, ' @ ', close_price, 'Position: ', position)
    else:
        #No trade since none of the conditions were met to buy or sell
        orders.append(0)
    positions.append(position)

    #flat starting a new position
    if current_pos == 0:
        if position != 0:
            current_pos = position
            current_pos_start = len(positions)
            #continue
    #going from long to flat or short, or
    #going from short to flat or long
    if current_pos * position <= 0:
        current_pos = position
        position_holding_time = len(positions) - current_pos_start
        current_pos_start = len(positions)

        if position_holding_time > RISK_LIMIT_MAX_POSITION_HOLDING_TIME_DAYS:
            print('RiskViolation position_holding_time', position_holding_time, ' > RISK_LIMIT_MAX_POSITION_HOLDING_TIME_DAYS', RISK_LIMIT_MAX_POSITION_HOLDING_TIME_DAYS)
            risk_violated = True

    if abs(position) > risk_limit_max_position:
        print('RiskViolation position', position, ' > RISK_LIMIT_MAX_POSITION', risk_limit_max_position)
        risk_violated = True

    if traded_volume > RISK_LIMIT_MAX_TRADED_VOLUME:
        print('RiskViolation traded_volume', traded_volume, ' > RISK_LIMIT_TRADED_VOLUME', RISK_LIMIT_MAX_TRADED_VOLUME)
        risk_violated = True            

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
        
    #print('Open PnL: ', open_pnl, ' Closed PnL: ', closed_pnl)
    pnls.append(closed_pnl + open_pnl)

    if len(pnls) > 5:
        weekly_loss = pnls[-1] - pnls[-6]

        if weekly_loss < risk_limit_weekly_stop_loss:
            print('RiskViolation weekly_loss', weekly_loss, ' > RISK_LIMIT_WEEKLY_STOP_LOSS', risk_limit_weekly_stop_loss)
            risk_violated = True             
    if len(pnls) > 20:
        monthly_loss = pnls[-1] - pnls[-21]

        if monthly_loss < risk_limit_monthly_stop_loss:
            print('RiskViolation monthly_loss', monthly_loss, ' > RISK_LIMIT_MONTHLY_STOP_LOSS', risk_limit_monthly_stop_loss)
            risk_violated = True

    #Adjust risk for good month
    if len(pnls) > 20:
        monthly_pnls = pnls[-1] - pnls[-20]

        if len(pnls) - last_risk_change_index > 20:
            if monthly_pnls > 0:
                num_shares_per_trade += INCREMENT_NUM_SHARES_PER_TRADE
                if num_shares_per_trade <= MAX_NUM_SHARES_PER_TRADE:
                    #print('Increasing trade-size and risk')
                    risk_limit_weekly_stop_loss += INCREMENT_RISK_LIMIT_WEEKLY_STOP_LOSS
                    risk_limit_monthly_stop_loss += INCREMENT_RISK_LIMIT_MONTHLY_STOP_LOSS
                    risk_limit_max_position += INCREMENT_RISK_LIMIT_MAX_POSITION
                    risk_limit_max_trade_size += INCREMENT_RISK_LIMIT_MAX_TRADE_SIZE
                else:
                    num_shares_per_trade = MAX_NUM_SHARES_PER_TRADE
            #Adjust risk for bad month
            elif monthly_pnls < 0:
                num_shares_per_trade -= INCREMENT_NUM_SHARES_PER_TRADE
                if num_shares_per_trade >= MIN_NUM_SHARES_PER_TRADE:
                    #print('Decreasing trade-size and risk')
                    risk_limit_weekly_stop_loss -= INCREMENT_RISK_LIMIT_WEEKLY_STOP_LOSS
                    risk_limit_monthly_stop_loss -= INCREMENT_RISK_LIMIT_MONTHLY_STOP_LOSS
                    risk_limit_max_position -= INCREMENT_RISK_LIMIT_MAX_POSITION
                    risk_limit_max_trade_size -= INCREMENT_RISK_LIMIT_MAX_TRADE_SIZE
                else:
                    num_shares_per_trade = MIN_NUM_SHARES_PER_TRADE
            last_risk_change_index = len(pnls)

    #Track trade-sizes/positions and risk limits as they evolve over time
    num_shares_history.append(num_shares_per_trade)
    abs_position_history.append(abs(position))
    max_trade_size_history.append(risk_limit_max_trade_size)
    max_position_history.append(risk_limit_max_position)
    

#This section prepares the dataframe from the trading strategy results and visualizes the results

data = data.assign(ClosePrice = pd.Series(close, index=data.index))
data = data.assign(Fast10DayEMA = pd.Series(ema_fast_values, index=data.index))
data = data.assign(Slow40DayEMA = pd.Series(ema_slow_values, index=data.index))
data = data.assign(APO = pd.Series(apo_values, index=data.index))
data = data.assign(Trades = pd.Series(orders, index=data.index))
data = data.assign(Position = pd.Series(positions, index=data.index))
data = data.assign(PnL = pd.Series(pnls, index=data.index))

data = data.assign(NumShares = pd.Series(num_shares_history, index=data.index))
data = data.assign(MaxTradeSize = pd.Series(max_trade_size_history, index=data.index))
data = data.assign(AbsPosition = pd.Series(abs_position_history, index=data.index))
data = data.assign(MaxPosition = pd.Series(max_position_history, index=data.index))

###This section begins Risk Management
##
###Stop-loss
##results = data
##
##num_days = len(results.index)
##
##pnl = results['PnL']
##
##weekly_losses = []
##monthly_losses = []
##
##for i in range(0,num_days):
##    if i >= 5 and pnl[i-5] > pnl[i]:
##        weekly_losses.append(pnl[i] - pnl[i-5])
##    if i >= 20 and pnl[i-20] > pnl[i]:
##        monthly_losses.append(pnl[i] - pnl[i-20])
##
##plt.hist(weekly_losses, 50)
##plt.gca().set(title='Weekly Loss Distribution', xlabel='$', ylabel='Frequency')
##plt.show()
##
##plt.hist(monthly_losses, 50)
##plt.gca().set(title='Monthly Loss Distribution', xlabel='$', ylabel='Frequency')
##plt.show()
##
###Max drawdown
##max_pnl = 0
##max_drawdown = 0
##drawdown_max_pnl = 0
##drawdown_min_pnl = 0
##
##for i in range(0, num_days):
##    max_pnl = max(max_pnl, pnl[i])
##    drawdown = max_pnl - pnl[i]
##
##    if drawdown > max_drawdown:
##        max_drawdown = drawdown
##        drawdown_max_pnl = max_pnl
##        drawdown_min_pnl = pnl[i]
##
##print('Max Drawdown:', max_drawdown)
##
##results['PnL'].plot(x='Date', legend=True)
##plt.axhline(y=drawdown_max_pnl, color='g')
##plt.axhline(y=drawdown_min_pnl, color='r')
##plt.show()
##
##
###Position Limits
##position = results['Position']
##plt.hist(position, 20)
##plt.gca().set(title='Position Distribution', xlabel='Shares', ylabel='Frequency')
##plt.show()
##
##
###Position Holding Time
##position_holding_time = []
##current_pos = 0
##current_pos_start = 0
##
##for i in range(0,num_days):
##    pos = results['Position'].iloc[i]
##
##    #flat and starting a new position
##    if current_pos == 0:
##        if pos != 0:
##            current_pos = pos
##            current_pos_start = i
##        continue
##    #going from long to flat or short, or
##    #going from short to flat or long
##    if current_pos * pos <= 0:
##        current_pos = pos
##        position_holding_time.append(i - current_pos_start)
##        current_pos_start = i
##
##print(position_holding_time)
##plt.hist(position_holding_time, 100)
##plt.gca().set(title='Position Holding Time Distribution', xlabel='Holding Time (Days)', ylabel='Frequency')
##plt.show()
##
##
###Variance of PnLs
##last_week = 0
##weekly_pnls = []
##for i in range(0, num_days):
##    if i - last_week >=5:
##        weekly_pnls.append(pnl[i] - pnl[last_week])
##        last_week = i
##
##print('Weekly PnL Standard Deviation:', np.std(weekly_pnls) )
##
##plt.hist(weekly_pnls, 50)
##plt.gca().set(title='Weekly PnL Distribution', xlabel='$', ylabel='Frequency')
##plt.show()
##
##
###Sharpe ratio
##last_week = 0
##weekly_pnls = []
##weekly_losses = []
##for i in range(0, num_days):
##    if i - last_week >= 5:
##        pnl_change = pnl[i] - pnl[last_week]
##        weekly_pnls.append(pnl_change)
##        if pnl_change < 0:
##            weekly_losses.append(pnl_change)
##        last_week = i
##sharpe_ratio = np.mean(weekly_pnls) / np.std(weekly_pnls)
##sortino_ratio = np.mean(weekly_pnls) / np.std(weekly_losses)
##
##print('Sharpe ratio:', sharpe_ratio)
##print('Sortino ratio:', sortino_ratio)
##
##
###Maximum executions per period
##executions_this_week = 0
##executions_per_week = []
##
##last_week = 0
##for i in range(0, num_days):
##    if results['Trades'].iloc[i] != 0:
##        executions_this_week += 1
##
##    if i - last_week >= 5:
##        executions_per_week.append(executions_this_week)
##        executions_this_week = 0
##        last_week = i
##
##plt.hist(executions_per_week, 10)
##plt.gca().set(title='Weekly number of executions Distribution', xlabel='Number of executions', ylabel='Frequency')
##plt.show()
##
##executions_this_month = 0
##executions_per_month = []
##
##last_week = 0
##for i in range(0, num_days):
##    if results['Trades'].iloc[i] != 0:
##        executions_this_month += 1
##
##    if i - last_week >= 20:
##        executions_per_month.append(executions_this_month)
##        executions_this_month = 0
##        last_week = i
##
##plt.hist(executions_per_month, 20)
##plt.gca().set(title='Monthly number of executions Distribution', xlabel='Number of executions', ylabel='Frequency')
##plt.show()
##
##
###Volume limits
##traded_volume = 0
##for i in range(0, num_days):
##    if results['Trades'].iloc[i] != 0:
##        traded_volume += abs(results['Position'].iloc[i] - results['Position'].iloc[i-1])
##print('Total traded volume:', traded_volume)

#Begin Strategy Plots
#Plot for Market Price with Fast and Slow EMA
data['ClosePrice'].plot(color='blue', lw=3, legend=True)
data['Fast10DayEMA'].plot(color='y', lw=1, legend=True)
data['Slow40DayEMA'].plot(color='m', lw=1, legend=True)
plt.plot(data.loc[data.Trades == 1].index, data.ClosePrice[data.Trades == 1], color='r', lw=0, marker='^', markersize=7, label='buy')
plt.plot(data.loc[data.Trades == -1].index, data.ClosePrice[data.Trades == -1], color='g', lw=0, marker='v', markersize=7, label='sell')
plt.legend()
plt.show()

#Plot for APO trading signal values
data['APO'].plot(color='k', lw=3, legend=True)
plt.plot(data.loc[data.Trades == 1].index, data.APO[data.Trades == 1], color='r', lw=0, marker='^', markersize=7, label='buy')
plt.plot(data.loc[data.Trades == -1].index, data.APO[data.Trades == -1], color='g', lw=0, marker='v', markersize=7, label='sell')
plt.axhline(y=0, lw=0.5, color='k')
for i in range(APO_VALUE_FOR_BUY_ENTRY, APO_VALUE_FOR_BUY_ENTRY * 5, APO_VALUE_FOR_BUY_ENTRY):
    plt.axhline(y=i, lw=0.5, color='r')
for i in range(APO_VALUE_FOR_SELL_ENTRY, APO_VALUE_FOR_SELL_ENTRY * 5, APO_VALUE_FOR_SELL_ENTRY):
    plt.axhline(y=i, lw=0.5, color='g')
plt.legend()
plt.show()

#Plot Position
data['Position'].plot(color='k', lw=1, legend=True)
plt.plot(data.loc[data.Position == 0].index, data.Position[data.Position == 0], color='k', lw=0, marker='.', label='flat')
plt.plot(data.loc[data.Position > 0].index, data.Position[data.Position > 0], color='r', lw=0, marker='+', label='long')
plt.plot(data.loc[data.Position < 0].index, data.Position[data.Position < 0], color='g', lw=0, marker='_', label='short')
plt.axhline(y=0, lw=0.5, color='k')
for i in range(NUM_SHARES_PER_TRADE, NUM_SHARES_PER_TRADE * 25, NUM_SHARES_PER_TRADE * 5):
    plt.axhline(y=i, lw=0.5, color='r')
for i in range(-NUM_SHARES_PER_TRADE, -NUM_SHARES_PER_TRADE * 25, -NUM_SHARES_PER_TRADE * 5):
    plt.axhline(y=i, lw=0.5, color='g')
plt.legend()
plt.show()

#Plot PnL
data['PnL'].plot(color='k', lw=1, legend=True)
plt.plot(data.loc[data.PnL > 0].index, data.PnL[data.PnL > 0], color='g', lw=0, marker='.')
plt.plot(data.loc[data.PnL < 0].index, data.PnL[data.PnL < 0], color='r', lw=0, marker='.')
plt.legend()
plt.show()

#Plot Number of Shares & Max Trade Size
data['NumShares'].plot(color='b', lw=3, legend=True)
data['MaxTradeSize'].plot(color='g', lw=1, legend=True)
plt.legend()
plt.show()

#Plot Absolute Position & Max Position
data['AbsPosition'].plot(color='b', lw=1, legend=True)
data['MaxPosition'].plot(color='g', lw=1, legend=True)
plt.legend()
plt.show()








        
