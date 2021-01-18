from ibapi.wrapper import *
from ibapi.client import *
from ibapi.contract import *
from ibapi.order import *
from threading import Thread
import queue
import datetime
import time
import math
from bs4 import BeautifulSoup
import requests
from random import randint
from ibapi.utils import iswrapper
import pandas as pd
import pickle
from pathlib import Path
import os
from tqdm import tqdm
import statistics as stats
from statistics import stdev
import math as math
import numpy as np
import multiprocessing as mp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

# Below are the global variables
availableFunds = 0 
buyingPower = 0
positionsDict = {}
stockPrice = 1000000
stockPriceBool = False

# Below are the custom classes and methods
def chunky(lst, n):
    chunk_list = []
    #Yield successive n-sized chunks from lst.
    for i in range(0, len(lst), n):
        chunk_list.append(lst[i:i + n])
    return chunk_list

def contractCreate(symbolEntered):
    # Fills out the contract object
    contract1 = Contract()  # Creates a contract object from the import
    contract1.symbol = symbolEntered   # Sets the ticker symbol 
    contract1.secType = "STK"   # Defines the security type as stock
    contract1.currency = "USD"  # Currency is US dollars 
    # In the API side, NASDAQ is always defined as ISLAND in the exchange field
    contract1.exchange = "SMART"
    contract1.PrimaryExch = "NASDAQ"

    #print('Contract created...')
    return contract1    # Returns the contract object

def orderCreate(quantityEntered=10):    # Defaults the quanity to 10 but is overriden later in orderExecution
    # Fills out the order object 
    order1 = Order()    # Creates an order object from the import

    # If the quantity is positive then we want to buy at that quantity, and sell if it is negative 
    if quantityEntered > 0: 
        order1.action = "BUY"   # Sets the order action to buy
        order1.totalQuantity = int(quantityEntered)   # Uses the quantity passed in orderExecution
    else: 
        order1.action = "SELL"
        order1.totalQuantity = abs(int(quantityEntered))   # Uses the quantity passed in orderExecution

    order1.orderType = "LMT"    # Sets order type to market buy
    order1.transmit = True

    print('Order object created...')
    return order1   # Returns the order object

######################################################################
######################################################################
############# Below is the TestWrapper/EWrapper class ################
######################################################################
######################################################################

'''
Here we will override the methods found inside api files. 
The wrapper is intended to receive responses from Interactive Brokers/TWS servers
'''

class TestWrapper(EWrapper):

    def __init__(self):
        self._my_contract_details = {}

    # error handling methods
    def init_error(self):
        error_queue = queue.Queue()
        self.my_errors_queue = error_queue

    def is_error(self):
        error_exist = not self.my_errors_queue.empty()
        return error_exist

    def get_error(self, timeout=6):
        if self.is_error():
            try:
                return self.my_errors_queue.get(timeout=timeout)
            except queue.Empty:
                return None
        return None

    def error(self, id, errorCode, errorString):
        ## Overrides the native method
        errormessage = "IB returns an error with %d errorcode %d that says %s" % (id, errorCode, errorString)
        self.my_errors_queue.put(errormessage)

    # time handling methods
    def init_time(self):
        time_queue = queue.Queue()
        self.my_time_queue = time_queue
        return time_queue

    def currentTime(self, server_time):
        ## Overriden method
        self.my_time_queue.put(server_time)

    def init_contractdetails(self, reqId):
        contract_details_queue = self._my_contract_details[reqId] = queue.Queue()

        return contract_details_queue

    def contractDetails(self, reqId, contractDetails):
        ## overridden method
        print("contractDetail: ", reqId, " ", contractDetails)

        if reqId not in self._my_contract_details.keys():
            self.init_contractdetails(reqId)

        self._my_contract_details[reqId].put(contractDetails)

    def contractDetailsEnd(self, reqId):
        ## overriden method
        if reqId not in self._my_contract_details.keys():
            self.init_contractdetails(reqId)

        self._my_contract_details[reqId].put(FINISHED)

    def init_nextvalidid(self):

        orderid_queue = self._my_orderid_data = queue.Queue()

        return orderid_queue

    def nextValidId(self, orderId):
        """
        Give the next valid order id
        Note this doesn't 'burn' the ID; if you call again without executing the next ID will be the same
        If you're executing through multiple clients you are probably better off having an explicit counter
        """
        if getattr(self, '_my_orderid_data', None) is None:
            ## getting an ID which we haven't asked for
            ## this happens, IB server just sends this along occassionally
            self.init_nextvalidid()

        self._my_orderid_data.put(orderId)

    # Account details handling methods
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        super().accountSummary(reqId, account, tag, value, currency)
        print("Acct Summary. ReqId:", reqId, "Acct:", account, "Tag: ",tag, "Value:", value, "Currency:", currency)
        self.my_acct_update_queue.put({'ReqId': reqId,
                                       'Account': account,
                                       'Tag': tag,
                                       'Value': value,
                                       'Currency': currency})

    def accountSummaryEnd(self, reqId: int):
        super().accountSummaryEnd(reqId)
        print("AccountSummaryEnd. Req Id: ", reqId)

    # Position handling methods
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        super().position(account, contract, position, avgCost)
        positionsDict[contract.symbol] = {'positions' : position, 'avgCost' : avgCost}
        print("Position.", account, "Symbol:", contract.symbol, "SecType:", contract.secType, "Currency:", contract.currency,"Position:", position, "Avg cost:", avgCost)
        self.my_positions_queue.put([contract.symbol, position, avgCost])

    def positionEnd(self):
        super().positionEnd()
        print("PositionEnd")

    # Market Price handling methods
    def tickPrice(self, reqId: TickerId, tickType: TickType, price: float, attrib: TickAttrib):
        super().tickPrice(reqId, tickType, price, attrib)
        print("Tick Price. Ticker Id:", reqId, "tickType:", 
            tickType, "Price:", price, "CanAutoExecute:", attrib.canAutoExecute, 
            "PastLimit:", attrib.pastLimit, end=' ')
        self.my_update_queue.put([reqId, tickType, price])
        #print('Added ticker to queue...')

        global stockPrice  # Declares that we want stockPrice to be treated as a global variable 
        global stockPriceBool    # A boolean flag that signals if the price has been updated

        # Use tickType 4 (Last Price) if you are running during the market day
        if int(tickType) == '4':
            print("\nParsed Tick Price: " + str(price))
            stockPrice = price
            stockPriceBool = True
            #self.my_update_queue.put(price)

        # Uses tickType 9 (Close Price) if after market hours 
        elif int(tickType) == '9': 
            print("\nParsed Tick Price: " + str(price))
            stockPrice = price
            stockPriceBool = True
            #self.my_update_queue.put(price)

    def tickSize(self, reqId: TickerId, tickType: TickType, size: int):
        super().tickSize(reqId, tickType, size)
        print("Tick Size. Ticker Id:", reqId, "tickType:", tickType, "Size:", size)
        self.my_update_queue.put([reqId, tickType, size])

    def tickString(self, reqId: TickerId, tickType: TickType, value: str):
        super().tickString(reqId, tickType, value)
        print("Tick string. Ticker Id:", reqId, "Type:", tickType, "Value:", value)
        self.my_update_queue.put([reqId, tickType, value])

    def tickGeneric(self, reqId: TickerId, tickType: TickType, value: float):
        super().tickGeneric(reqId, tickType, value)
        print("Tick Generic. Ticker Id:", reqId, "tickType:", tickType, "Value:", value)
        self.my_update_queue.put([reqId, tickType, value])

    def tickSnapshotEnd(self, reqId: int):
        super().tickSnapshotEnd(reqId)
        print("TickSnapshotEnd. TickerId:", reqId)

    def historicalData(self, reqId, bar):
        print("HistoricalData. ", reqId, " Date:", bar.date, "Open:", bar.open,
              "High:", bar.high, "Low:", bar.low, "Close:", bar.close, "Volume:", bar.volume,
              "Count:", bar.barCount, "WAP:", bar.average)
        self.my_hist_queue.put({'Request ID': reqId,
                                'Date': bar.date,
                                'Open': bar.open,
                                'High': bar.high,
                                'Low': bar.low,
                                'Close': bar.close,
                                'Volume': bar.volume,
                                'Count': bar.barCount,
                                'WAP': bar.average})       

    def pnlSingle(self, reqId: int, pos: int, dailyPnL: float,unrealizedPnL: float, realizedPnL: float, value: float):
        super().pnlSingle(reqId, pos, dailyPnL, unrealizedPnL, realizedPnL, value)
        print("Daily PnL Single. ReqId:", reqId, "Position:", pos,
                   "DailyPnL:", dailyPnL, "UnrealizedPnL:", unrealizedPnL,
                   "RealizedPnL:", realizedPnL, "Value:", value)
        self.my_pnl_queue.put({'Request ID': reqID,
                               'Position': pos,
                               'DailyPnL': dailyPnL,
                               'UnrealizedPnL': unrealizedPnL,
                               'RealizedPnL': realizedPnL,
                               'Value': value})

    def pnl(self, reqId: int, dailyPnL: float, unrealizedPnL: float, realizedPnL: float):
        super().pnl(reqId, dailyPnL, unrealizedPnL, realizedPnL)
        print("Daily PnL. ReqId:", reqId, "DailyPnL:", dailyPnL, "UnrealizedPnL:", unrealizedPnL, "RealizedPnL:", realizedPnL)
        self.my_account_pnl_queue.put({'Req ID': reqId,
                                       'Daily PnL': dailyPnL,
                                       'Unrealized PnL': unrealizedPnL,
                                       'Realized PnL': realizedPnL})

    def openOrder(self, orderId: OrderId, contract: Contract, order: Order, orderState: OrderState):
        super().openOrder(orderId, contract, order, orderState)
        print("OpenOrder. PermId: ", order.permId, "ClientId:", order.clientId, " OrderId:", orderId,
              "Account:", order.account, "Symbol:", contract.symbol, "SecType:", contract.secType,
              "Exchange:", contract.exchange, "Action:", order.action, "OrderType:", order.orderType,
              "TotalQty:", order.totalQuantity, "CashQty:", order.cashQty,
              "LmtPrice:", order.lmtPrice, "AuxPrice:", order.auxPrice, "Status:", orderState.status)

        order.contract = contract

        self.my_open_order_queue.put({'PermId': order.permId,
                                       "ClientId": order.clientId,
                                       "OrderId": orderId,
                                       "Account": order.account,
                                       "Symbol": contract.symbol,
                                       "SecType": contract.secType,
                                       "Exchange": contract.exchange,
                                       "Action": order.action,
                                       "OrderType": order.orderType,
                                       "TotalQty": order.totalQuantity,
                                       "CashQty": order.cashQty,
                                       "LmtPrice": order.lmtPrice,
                                       "AuxPrice": order.auxPrice,
                                       "Status": orderState.status})

    def orderStatus(self, orderId: OrderId, status: str, filled: float,
                    remaining: float, avgFillPrice: float, permId: int,
                    parentId: int, lastFillPrice: float, clientId: int,
                    whyHeld: str, mktCapPrice: float):
        super().orderStatus(orderId, status, filled, remaining,
                            avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
        print("OrderStatus. Id:", orderId, "Status:", status, "Filled:", filled,
              "Remaining:", remaining, "AvgFillPrice:", avgFillPrice,
              "PermId:", permId, "ParentId:", parentId, "LastFillPrice:",
              lastFillPrice, "ClientId:", clientId, "WhyHeld:",
              whyHeld, "MktCapPrice:", mktCapPrice)

    def openOrderEnd(self):
        super().openOrderEnd()
        print("OpenOrderEnd")


######################################################################
######################################################################
############# Below is the TestClient/EClient Class ##################
######################################################################
######################################################################
                                       
'''Here we will call our own methods, not overriding the api methods'''

class TestClient(EClient):

    def __init__(self, wrapper):
    ## Set up with a wrapper inside
        EClient.__init__(self, wrapper)

        #self.Ticker_ID = 1#randint(1,100000000000)
        #self.Ticker_ID = self.nextValidOrderId
        self.positionDictionary = {}

    def get_next_brokerorderid(self):
        """
        Get next broker order id
        :return: broker order id, int; or TIME_OUT if unavailable
        """

        ## Make a place to store the data we're going to return
        orderid_q = self.wrapper.init_nextvalidid()

        self.reqIds(-1) # -1 is irrelevant apparently (see IB API docs)

        #time.sleep(3)

##        while not orderid_q.empty():
##            brokerorderid = orderid_q.get()

        ## Run until we get a valid contract(s) or get bored waiting
        MAX_WAIT_SECONDS = 10
        try:
            brokerorderid = orderid_q.get(timeout=MAX_WAIT_SECONDS)
        except queue.Empty:
            print("Wrapper timeout waiting for broker orderid")
            #brokerorderid = self.run_algorithm() #TIME_OUT
            self.get_next_brokerorderid()

        while self.wrapper.is_error():
            print(self.get_error(timeout=MAX_WAIT_SECONDS))

        return brokerorderid
    
    
    # Account details handling methods
    def account_update(self):
        acct_update_queue = queue.Queue()
        self.my_acct_update_queue = acct_update_queue
        summaryTags = 'TotalCashValue, BuyingPower, AvailableFunds, NetLiquidation, Leverage'
                       
        self.reqAccountSummary(9001, "All", summaryTags)
        time.sleep(2)

        currentAccountSummary = {'Total Cash Value': 0,
                                 'Available Funds': 0,
                                 'Buying Power': 0,
                                 'Net Liquidation': 0,
                                 'Leverage': 0}

        for x in self.my_acct_update_queue:
            if x[2] == 'TotalCashValue':
                currentAccountSummary['Total Cash Value'] = x[3]
            if x[2] == 'BuyingPower':
                currentAccountSummary['Buying Power'] = x[3]
            if x[2] == 'AvailableFunds':
                currentAccountSummary['Available Funds'] = x[3]
            if x[2] == 'NetLiquidation':
                currentAccountSummary['Net Liquidation'] = x[3]
            if x[2] == 'Leverage':
                currentAccountSummary['Leverage'] = x[3]

        self.currentAccountSummary = currentAccountSummary

        #self.cancelAccountSummary(9001)

    def position_update(self):
        self.reqPositions()

    def price_update(self, Contract, tickerid):
        update = self.reqMktData(tickerid, Contract, "", False, False, [])
        return tickerid, update

    def init_hist(self):
        hist_queue = queue.Queue()
        self.my_hist_queue = hist_queue
        return hist_queue

    def request_historical_price(self, symbol):
        
        Contract = contractCreate(symbol)
        nextID = self.get_next_brokerorderid()
        ticker_list = []

        print('Asking server for historical data about {}...'.format(symbol))

        # Creates a queue to store the historical data
        hist_storage = self.wrapper.init_hist()

        max_wait_time = 10
        num_of_days = 120
        
        data = self.reqHistoricalData(nextID, Contract, '', '{} D'.format(num_of_days), '1 day', "TRADES", 1, 1, False, [])
        
        try:
            for i in range(num_of_days):
                ticker_list.append(hist_storage.get(timeout = max_wait_time))
        except queue.Empty:
            print("The queue was empty or max time reached")
            data = None

        df = pd.DataFrame(ticker_list)

        if df.empty:
            print('{} dataframe is empty...'.format(symbol))
            return data
        else:
            df.Date = df.Date.str[:4] + '-' + df.Date.str[4:6] + '-' + df.Date.str[6:8]
            df.Date = pd.to_datetime(df.Date)

            save_file = '{}/Dropbox (ASU)/Algorithmic_Trading/Historical_IB_Data'.format(str(Path.home()))

            df.to_pickle('{}/{}.pkl'.format(save_file,symbol))

            print(df)

            while self.wrapper.is_error():
              print("Error:")
              print(self.get_error(timeout=10))

            return data

    def request_short_historical_price(self, symbol):
    
        Contract = contractCreate(symbol)
        nextID = self.get_next_brokerorderid()
        ticker_list = []

        print('Asking server for historical data about {}...'.format(symbol))

        # Creates a queue to store the historical data
        hist_storage = self.wrapper.init_hist()

        max_wait_time = 10
        num_of_days = 5
        
        data = self.reqHistoricalData(nextID, Contract, '', '{} D'.format(num_of_days), '10 mins', "TRADES", 1, 1, False, [])
        
        try:
            for i in range(num_of_days):
                ticker_list.append(hist_storage.get(timeout = max_wait_time))
        except queue.Empty:
            print("The queue was empty or max time reached")
            data = None

        df = pd.DataFrame(ticker_list)

        if df.empty:
            print('{} dataframe is empty...'.format(symbol))
            return data
        else:
            df.Date = df.Date.str[:4] + '-' + df.Date.str[4:6] + '-' + df.Date.str[6:8]
            df.Date = pd.to_datetime(df.Date)

            save_file = '{}/Dropbox (ASU)/Algorithmic_Trading/Historical_IB_Data/Short_Histories'.format(str(Path.home()))

            df.to_pickle('{}/{}.pkl'.format(save_file,symbol))

            print(df)

            while self.wrapper.is_error():
              print("Error:")
              print(self.get_error(timeout=10))

            return data

    def init_update(self):
        #self.my_update_queue = []
        update_queue = queue.Queue()
        self.my_update_queue = update_queue
        return update_queue

    def get_ticker_update(self, symbol, ticker_ID):
        Contract = contractCreate(symbol)

        app.reqMarketDataType(1)
        app.reqMktData(ticker_ID, Contract, "", True, False, [])

        while self.wrapper.is_error():
          print("Error:")
          print(self.get_error(timeout=1))

        #return list(self.update_storage.queue)

    def iterate_update(self):

        ticker_id = self.myOrderID

        

        directory = '{}/Dropbox (ASU)/Algorithmic_Trading/Historical_IB_Data/'.format(str(Path.home()))

        data_dict = {}
        ticker_list = []
        
        reference_dict = {}
        self.reference_dict = reference_dict = {}

        reverse_ref_dict = {}
        self.reverse_ref_dict = reverse_ref_dict

        self.length = 0
        

        start_time = time.time()
        counter = 0
        
        for filename in os.listdir(directory):
            if filename.endswith(".pkl"):
                data = pickle.load(open(directory+filename, 'rb'))
                self.hist_file_name = directory+filename
                symbol = filename[:-4]
                print('Ticker update requested for {}.'.format(symbol))

                data_dict[ticker_id] = data
                self.reference_dict[self.myOrderID] = symbol
                self.reverse_ref_dict[symbol] = self.myOrderID
                
##                for ticker_data in self.get_ticker_update(symbol, ticker_id):
##                    ticker_list.append(ticker_data)

                self.get_ticker_update(symbol, self.myOrderID)
                self.myOrderID += 1
                    
                ticker_id += 1
                counter += 1

                if counter == 90:
                    counter = 0
                    elapsed_time = time.time() - start_time
                    if elapsed_time < 3:
                        print('Whoa there, guy! Can we slow it down just a bit? Patience is a virtue.')
                        time.sleep(3-elapsed_time)
                        start_time = time.time()

                self.length += 1

        print('Sleeping and waiting for the last of the requests to come in...')
        time.sleep(5)
        print('6 more seconds...')
        time.sleep(6)
                

        self.data_dict = data_dict
        self.ticker_list = ticker_list


    def create_updateRow(self):
        self.update_storage = self.wrapper.init_update()
        self.iterate_update()

        update_dict = {}

        print('Creating dictionary with empty values...')

        for ticker_id in list(self.data_dict.keys()):
            update_dict[ticker_id] = {'Request ID': 0,
                                    'Date': 0,
                                    'Open': 0,
                                    'High': 0,
                                    'Low': 0,
                                    'Close': 0,
                                    'Volume': 0,
                                    'Count': 0,
                                    'WAP': 0}
            

        print('Analyzing ticker_list to create dictionary of updates...')

        
        print('###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@')
        print('@@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###')
        print('###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@')
        print('@@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###')
        print('       The algorithm believes you have {} left in the account.'.format(self.cash))
        print('@@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###')
        print('###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@')
        print('@@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###')
        print('###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@   ###   @@@')

        if self.interationCounter == 0:
            for y in range(len(list(self.my_update_queue.queue))):
                x = self.my_update_queue.get()

                self.latest_information[self.reference_dict[x[0]]] = {'Open': 0,
                                                                      'High': 0,
                                                                      'Low': 0,
                                                                      'Close': 0,
                                                                      'Volume': 0,
                                                                      'Count': 0,
                                                                      'Bid': 0,
                                                                      'Ask': 0,
                                                                      'Position': 0,
                                                                      'Avg Price': 0}


        self.check_if_broken()
        
        #with tqdm(total = len(list(self.my_update_queue.queue))) as pbar:
            #for y in range(len(list(self.my_update_queue.queue))):
        while not self.my_update_queue.empty():
            x = self.my_update_queue.get(timeout=11)

##                self.latest_information[self.reference_dict[x[0]]] = {'Open': 0,
##                                                                      'High': 0,
##                                                                      'Low': 0,
##                                                                      'Close': 0,
##                                                                      'Volume': 0,
##                                                                      'Count': 0,
##                                                                      'Bid': 0,
##                                                                      'Ask': 0,
##                                                                      'Position': 0,
##                                                                      'Avg Price': 0}

            if self.reference_dict[x[0]] not in self.latest_information.keys():
                self.latest_information[self.reference_dict[x[0]]] = {}
            
            if int(update_dict[x[0]]['Request ID']) == 0:
                update_dict[x[0]]['Request ID'] = x[0]
            if str(update_dict[x[0]]['Date']) == '0':
                update_dict[x[0]]['Date'] = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
                
            if int(x[1]) == 14:
                update_dict[x[0]]['Open'] = x[2]
                self.latest_information[self.reference_dict[x[0]]]['Open'] = x[2]
            elif int(x[1]) == 6:
                update_dict[x[0]]['High'] = x[2]
                self.latest_information[self.reference_dict[x[0]]]['High'] = x[2]
            elif int(x[1]) == 7:
                update_dict[x[0]]['Low'] = x[2]
                self.latest_information[self.reference_dict[x[0]]]['Low'] = x[2]
            elif int(x[1]) == 4:
                update_dict[x[0]]['Close'] = x[2]
                self.latest_information[self.reference_dict[x[0]]]['Close'] = x[2]
            elif int(x[1]) == 8:
                update_dict[x[0]]['Volume'] = x[2]
                self.latest_information[self.reference_dict[x[0]]]['Volume'] = x[2]
            elif int(x[1]) == 54:
                update_dict[x[0]]['Count'] = x[2]
                self.latest_information[self.reference_dict[x[0]]]['Count'] = x[2]
            elif int(x[1]) == 9:
                now = datetime.datetime.now().time()
                market_open = now.replace(hour=6, minute=30, second=0, microsecond=0)
                market_close = now.replace(hour=13, minute=0, second=0, microsecond=0)
                if not market_open < now < market_close:
                    update_dict[x[0]]['Close'] = x[2]
                    self.latest_information[self.reference_dict[x[0]]]['Close'] = x[2]
                    

            elif int(x[1]) == 1:
                self.latest_information[self.reference_dict[x[0]]]['Bid'] = x[2]

            elif int(x[1]) == 2:
                self.latest_information[self.reference_dict[x[0]]]['Ask'] = x[2]

                #print(self.latest_information[self.reference_dict[x[0]]])

                #pbar.update()

        self.update_dict = update_dict

        self.iteration_update_dict = update_dict

    def init_positions(self):
        positions_queue = queue.Queue()
        self.my_positions_queue = positions_queue
        return positions_queue

    def open_day_check_positions(self):
        self.position_storage = self.wrapper.init_positions()
        self.position_update()
        time.sleep(3)
        holdings = list(self.position_storage.queue)

        self.request_account_pnl()
        self.cash = 5000 + self.account['PnL']['Realized PnL']

        for x in holdings:
            self.portfolio[x[0]] = {'position': x[1],
                                    'last price': x[2]}

            self.cash -= (abs(x[1]) * x[2])

            #Adjust for realized PnL here
            if x[0] not in self.latest_information.keys():
                    self.latest_information[x[0]] = {}

            self.latest_information[x[0]]['Position'] = x[1]
            self.latest_information[x[0]]['Avg Price'] = x[2]

        if self.cash < 0:
            self.cancel_outstanding_orders()
            time.sleep(2)
            print('Selling everything to raise capital...')
            self.close_day()
            time.sleep(10)
            self.open_day_check_positions()
                    

        while self.wrapper.is_error():
          print("Error:")
          print(self.get_error(timeout=1))

    def close_day(self):
        self.position_storage = self.wrapper.init_positions()
        self.position_update()
        print('Sleeping before closing the day...')
        time.sleep(1)
        stocks_to_normalize = list(self.position_storage.queue)

        for stock in stocks_to_normalize:
            if stock[1] <= -1:
                order1 = Order()
                order1.action="BUY"
                order1.orderType="LMT"
                order1.tif = 'DAY'
                order1.lmtPrice = self.latest_information[stock[0]]['Bid']
                order1.totalQuantity= abs(stock[1])
                order1.transmit = True
                contract1 = contractCreate(stock[0])

                self.placeOrder(self.myOrderID,contract1,order1)
                self.myOrderID += 1
                
                print('*** *** *** *** *** *** *** *** *** *** *** *** *** *** ***')
                print('*** *** *** CLOSE DAY: Bought {} for {} profit *** *** ***'.format(stock[0], stock[2] - self.portfolio[stock[0]]['last price']))
                print('*** *** *** *** *** *** *** *** *** *** *** *** *** *** ***')

            if stock[1] >= 1:
                order2 = Order()
                order2.action="SELL"
                order2.orderType="LMT"
                order2.tif = 'DAY'
                order2.lmtPrice = self.latest_information[stock[0]]['Ask']
                order2.totalQuantity= abs(stock[1])
                order2.transmit = True
                contract2 = contractCreate(stock[0])

                self.placeOrder(self.myOrderID,contract2,order2)
                self.myOrderID += 1

                print('*** *** *** *** *** *** *** *** *** *** *** *** *** *** ***')
                print('*** *** *** CLOSE DAY: Sold {} for {} profit *** *** ***'.format(stock[0], self.portfolio[stock[0]]['last price'] - stock[2]))
                print('*** *** *** *** *** *** *** *** *** *** *** *** *** *** ***')


    def fix_positions(self):
        self.position_storage = self.wrapper.init_positions()
        self.position_update()
        print('Sleeping before fixing positions...')
        time.sleep(1)
        stocks_to_adjust = list(self.position_storage.queue)
        
        for stock in stocks_to_adjust:
            if stock[1] < -self.max_stock_holdings:
                order1 = Order()
                order1.action="BUY"
                order1.orderType="LMT"
                order1.tif = 'IOC'
                order1.lmtPrice = self.latest_information[stock[0]]['Bid']
                order1.totalQuantity= abs(stock[1]) - self.max_stock_holdings
                order1.transmit = True
                contract1 = contractCreate(stock[0])

                self.placeOrder(self.myOrderID,contract1,order1)
                self.myOrderID += 1

                self.portfolio[symbol]['position'] = self.max_stock_holdings

                print('Order placed to buy {} to reduce position to -{}.'.format(stock[0],self.max_stock_holdings))

            if stock[1] > self.max_stock_holdings:
                order1 = Order()
                order1.action="SELL"
                order1.orderType="LMT"
                order1.tif = 'IOC'
                order1.lmtPrice = self.latest_information[stock[0]]['Ask']
                order1.totalQuantity= abs(stock[1]) - self.max_stock_holdings
                order1.transmit = True
                contract1 = contractCreate(stock[0])

                self.placeOrder(self.myOrderID,contract1,order1)
                self.myOrderID += 1

                print('Order placed to sell {} to reduce position to {}.'.format(stock[0],self.max_stock_holdings))

                self.portfolio[symbol]['position'] = self.max_stock_holdings


    def init_pnl(self):
        pnl_queue = queue.Queue()
        self.my_pnl_queue = pnl_queue
        return pnl_queue

    def init_account_pnl(self):
        account_pnl_queue = queue.Queue()
        self.my_account_pnl_queue = account_pnl_queue
        return account_pnl_queue

    def request_account_pnl(self):
        self.account_pnl_storage = self.wrapper.init_account_pnl()
        
        self.reqPnL(self.myOrderID, "DU2476263", "")
        time.sleep(2)
        self.cancelPnL(self.myOrderID)

        self.myOrderID += 1

        
        account_pnl_list = list(self.my_account_pnl_queue.queue)
        for x in account_pnl_list:
            self.account['PnL'] = {'Daily PnL': x['Daily PnL'],
                                  'Unrealized PnL': x['Unrealized PnL'],
                                  'Realized PnL': x['Realized PnL']}

    def realize_profit(self):
        self.request_account_pnl()
        
        if self.account['PnL']['Unrealized PnL'] > 10 or self.account['PnL']['Daily PnL'] > 10:
            self.close_day()


    def init_open_orders(self):
        open_orders_queue = queue.Queue()
        self.my_open_order_queue = open_orders_queue
        return open_orders_queue

    def cancel_outstanding_orders(self):
        self.open_order_storage = self.wrapper.init_open_orders()
        time.sleep(0.5)
        self.reqOpenOrders()
        time.sleep(0.5)

        open_order_list = list(self.open_order_storage.queue)

        for order in open_order_list:
            self.cancelOrder(order['OrderId'])

        self.open_order_list = []


    def store_performance(self):

        performance_list = []
        self.performance_list = performance_list

        timestamp_list = []
        self.timestamp_list = timestamp_list

    def map_performance(self):

        x = np.array(self.timestamp_list)
        start_time = datetime.datetime.today()
        start_time = start_time.replace(hour=6, minute=30, second=0, microsecond=0)
        np.insert(x, 0, start_time)
                  
        y = np.array(self.performance_list)
        np.insert(y, 0, 5000)

        d = {'time': x, 'US $': y}

        df = pd.DataFrame(d)

        sns.set(style='darkgrid')
        sns.lineplot(x='time', y='US $', data=df)

        plt.xticks(rotation=45)

        plt.show()


    def create_latest_dict(self):
        directory = '{}/Dropbox (ASU)/Algorithmic_Trading/Historical_IB_Data/'.format(str(Path.home()))
        for filename in os.listdir(directory):
            if filename.endswith(".pkl"):
                symbol = filename[:-4]
                self.latest_information[symbol] = {}

    def init_storage_stuff(self):
        self.pnl_storage = self.init_pnl()
        self.reference_dict = {}
        self.pnl_reference_dict = {}

        pnl_dict = {}
        self.pnl_dict = pnl_dict
        my_stocks = {}
        self.my_stocks = my_stocks
        final_data_dict = {}
        self.final_data_dict = final_data_dict
        orders_dict = {}
        self.orders_dict = orders_dict

        bid_ask_dict = {}
        self.bid_ask_dict = bid_ask_dict

        latest_information = {}
        self.latest_information = latest_information
        self.create_latest_dict()

        self.daily_dict = {}
        self.daily_setup()


    def check_if_broken(self):
        if self.cash < 0:
            self.cashlist.append(self.cash)
            self.brokenCounter +=1
            self.cash = 5000

            print('### ### ### ### ### ### ### ### ### ###')
            print('### ### ### ### ### ### ### ### ### ###')
            print('### ### ### ### ### ### ### ### ### ###')
            print('### ### ### ### ### ### ### ### ### ###')
            print('The algorithm went broke. Count = {}'.format(self.brokenCounter))
            print('### ### ### ### ### ### ### ### ### ###')
            print('### ### ### ### ### ### ### ### ### ###')
            print('### ### ### ### ### ### ### ### ### ###')
            print('### ### ### ### ### ### ### ### ### ###')

            time.sleep(1)

        return self.brokenCounter

    def set_cash_and_timestamp(self):
        try:
            self.cashlist = pickle.load(open('{}/Dropbox (ASU)/Algorithmic_Trading/{}_Cash_List.pkl'.format(str(Path.home()), datetime.datetime.today().strftime('%Y-%m-%d')), 'rb'))
            self.cash = self.cashlist[-1]
        except FileNotFoundError:
            self.cashlist = []
            self.cash = 5000

        try:
            self.timestamp_list = pickle.load(open('{}/Dropbox (ASU)/Algorithmic_Trading/{}_TimeStamps.pkl'.format(str(Path.home()), datetime.datetime.today().strftime('%Y-%m-%d')), 'rb'))
        except FileNotFoundError:
            self.timestamp_list = []


    def daily_setup(self):
        directory = '{}/Dropbox (ASU)/Algorithmic_Trading/Historical_IB_Data/'.format(str(Path.home()))
        for filename in os.listdir(directory):
            if filename.endswith(".pkl"):
                symbol = filename[:-4]
                self.daily_dict[symbol] = []

    def place_order(self,order_action,order_type,quantity,price,order_tif,symbol):
        order = Order()
        order.action=order_action #'SELL' #"BUY"
        order.orderType=order_type
        order.totalQuantity= quantity
        order.lmtPrice = price
        order.tif = order_tif
        order.transmit = True
        contract = contractCreate(symbol)
        self.placeOrder(self.myOrderID,contract,order)
        self.myOrderID += 1
        

    ''' **************************************************************
        **************************************************************
        **************************************************************
        **************************************************************
                            RUN ALGORITHM MODULE
        **************************************************************
        **************************************************************
        **************************************************************
        **************************************************************'''

    def run_algorithm(self):
        self.init_open_orders()

        iteration_update_dict = {}
        self.iteration_update_dict = iteration_update_dict
        
        if self.interationCounter == 4:
            self.cancel_outstanding_orders()
            self.interationCounter = 1
            print('Canceling outstanding orders...')

        print('Updating stock data with most current ticker...')
        self.create_updateRow()

        print('Updating positions...')
        if self.interationCounter == 0:
            self.account['PnL'] = {'Daily PnL': 0,
                                  'Unrealized PnL': 0,
                                  'Realized PnL': 0}

        print('Requesting account PnL...')
        self.request_account_pnl()

        
        time.sleep(2)
        self.open_day_check_positions()
        

        print('Running the algorithms...')
        self.trend_following = False
        self.mean_reversion = False
        with tqdm(total = len(self.data_dict)) as pbar:
            for reqID in self.reference_dict:
##                if self.interationCounter <= 1:
##                    self.mean_reversion_algo(self.data_dict[reqID], self.reference_dict[reqID], reqID)
##                elif self.interationCounter == 2:
##                    self.analyze_trend_algo(self.data_dict[reqID], self.reference_dict[reqID], reqID)
##                else:
##                    self.custom_daily_trading_algo(self.reference_dict[reqID])
                self.custom_daily_trading_algo(self.reference_dict[reqID])
                pbar.update()

        print('Calculating portfolio return...')
        with tqdm(total = len(self.data_dict)) as pbar:
            price_list = []
            for order in self.orders_dict:
                try:
                    price_list.append(self.orders_dict[order]['price'])
                    pbar.update()
                except KeyError:
                    pbar.update()
                    continue
                
        end_of_day = sum(price_list)
        self.end_of_day = end_of_day

        print('Fixing positions...hopefully....')
        self.fix_positions()

        print('Raw Dataframes: ', self.final_data_dict)
        print('Most Profitable Stocks: ', self.my_stocks)
        print('Average Profit per Day: ', self.pnl_dict)
        print('End of day profit: ', self.end_of_day)

        self.timestamp_list.append(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
        self.interationCounter += 1
        self.cashlist.append(self.cash)

        
        outfile_cash = open('{}/Dropbox (ASU)/Algorithmic_Trading/{}_Cash_List.pkl'.format(str(Path.home()), datetime.datetime.today().strftime('%Y-%m-%d')), 'wb')
        pickle.dump(self.cashlist, outfile_cash)
        outfile_cash.close()

        outfile_time = open('{}/Dropbox (ASU)/Algorithmic_Trading/{}_TimeStamps.pkl'.format(str(Path.home()), datetime.datetime.today().strftime('%Y-%m-%d')), 'wb')
        pickle.dump(self.timestamp_list, outfile_time)
        outfile_time.close()

        #self.init_open_orders()

        self.myOrderID += 1



    ''' **************************************************************
        **************************************************************
        **************************************************************
        **************************************************************
                            MEAN REVERSION MODULE
        **************************************************************
        **************************************************************
        **************************************************************
        **************************************************************'''

    def mean_reversion_algo(self, data, name, reqID):

        if 'Close' not in self.latest_information[name]:
            print('No update for {}'.format(name))
            return
        
        mean_reversion = True
        self.mean_reversion = mean_reversion

        reported_position = 0
        min_daily_profit = 5

        allowable_limit = self.cash * 0.05

        check_for_profit_sell = False
        check_for_profit_buy = False

        max_stock_holdings = 10
        self.max_stock_holdings = max_stock_holdings
        
        #Variables for variable risk managment
        MIN_NUM_SHARES_PER_TRADE = 1
        MAX_NUM_SHARES_PER_TRADE = 5
        INCREMENT_NUM_SHARES_PER_TRADE = 1
        num_shares_per_trade = MIN_NUM_SHARES_PER_TRADE #beginning number of shares to buy or sell on every trade
        num_shares_history = [] #history of num_shares
        abs_position_history = [] #history of absolute-position
        CAPITAL = self.cash

        #Risk Limits
        risk_limit_weekly_stop_loss = -100
        INCREMENT_RISK_LIMIT_WEEKLY_STOP_LOSS = -25
        risk_limit_monthly_stop_loss = -200
        INCREMENT_RISK_LIMIT_MONTHLY_STOP_LOSS = -50
        risk_limit_max_position = 5
        INCREMENT_RISK_LIMIT_MAX_POSITION = 1
        max_position_history = [] #history of max-position
        RISK_LIMIT_MAX_POSITION_HOLDING_TIME_DAYS = 5 #30
        risk_limit_max_trade_size = 5
        INCREMENT_RISK_LIMIT_MAX_TRADE_SIZE = 1
        max_trade_size_history = [] #history of max-trade-size
        RISK_LIMIT_MAX_TRADED_VOLUME = CAPITAL * 0.75 #1.5

        last_risk_change_index = 0

        #Variables to track/check for risk violations
        risk_violated = False

        traded_volume = 0
        current_pos = 0
        current_pos_start = 0

        #Variables/constants for EMA Calculation:

        NUM_PERIODS_FAST = 5 #10 #Static time period parameter for the fast EMA
        K_FAST = 2 / (NUM_PERIODS_FAST + 1) #Static smoothing factor for fast EMA
        ema_fast = 0
        ema_fast_values = [] #Hold EMA values for visualization

        NUM_PERIODS_SLOW = 20 #40
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
        position = reported_position #0 #Current position
        buy_sum_price_qty = 0 #Summation of products of buy_trade_price and buy_trade_qty for every buy trade since being flat
        buy_sum_qty = 0 #Summation of buy_trade_qty for every buy trade made since last time being flat
        sell_sum_price_qty = 0
        sell_sum_qty = 0
        open_pnl = 0 #Open/Unrealized PnL marked to market
        closed_pnl = 0 # Closed/Realized PnL so far

        # Constants that define strategy behavior/thresholds
        APO_VALUE_FOR_BUY_ENTRY = -2 #-10 #APO trading signal value below which to enter buy-orders/long position
        APO_VALUE_FOR_SELL_ENTRY = 2 #10
        MIN_PRICE_MOVE_FROM_LAST_TRADE = 2 #10 #Minimum price change since last trade before considering trading again, this is to preven over-trading at/around same price
        MIN_PROFIT_TO_CLOSE = num_shares_per_trade * 2 #10 #Minimum Open/Unrealized profit at which to close positions and lock profits
        NUM_SHARES_PER_TRADE = 1 #Number of buy/sell on every trade

        # Constants/variables that are used to compute standard deviation as a volatility measure
        SMA_NUM_PERIODS = 15 #20 #look back period
        price_history = [] #history of prices

        # Places to hold valuable stocks

        data = data.append(self.iteration_update_dict[reqID], ignore_index=True)
        
        log_prediction = self.logistic_regression(data,name)

        close = list(data.Close.values)

        early_stdev = np.std(close)
        early_stdev_factor = early_stdev/15 #15
        if early_stdev_factor == 0:
            early_stdev_factor = 1

        if name in self.portfolio.keys() and not self.trend_following:

            if  self.latest_information[name]['Close'] - self.latest_information[name]['Avg Price'] >= early_stdev_factor * MIN_PRICE_MOVE_FROM_LAST_TRADE and self.latest_information[name]['Position'] >= 1:
                check_for_profit_sell = True

            if self.latest_information[name]['Avg Price'] - self.latest_information[name]['Close'] >= early_stdev_factor * MIN_PRICE_MOVE_FROM_LAST_TRADE and self.latest_information[name]['Position'] <= -1:
                check_for_profit_buy = True

        if name not in self.portfolio.keys():
            self.portfolio[name] = {'position': 0,
                                    'last price': 0}
        
        for close_price in close:
            price_history.append(float(close_price))
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
                #print('RiskViolation NUM_SHARES_PER_TRADE', NUM_SHARES_PER_TRADE, ' > RISK_LIMIT_MAX_TRADE_SIZE', risk_limit_max_trade_size)
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
                    #print('RiskViolation position_holding_time', position_holding_time, ' > RISK_LIMIT_MAX_POSITION_HOLDING_TIME_DAYS', RISK_LIMIT_MAX_POSITION_HOLDING_TIME_DAYS)
                    risk_violated = True

            if abs(position) > risk_limit_max_position:
                #print('RiskViolation position', position, ' > RISK_LIMIT_MAX_POSITION', risk_limit_max_position)
                risk_violated = True

            if traded_volume > RISK_LIMIT_MAX_TRADED_VOLUME:
                #print('RiskViolation traded_volume', traded_volume, ' > RISK_LIMIT_TRADED_VOLUME', RISK_LIMIT_MAX_TRADED_VOLUME)
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

            #adjust captial funds
            CAPITAL += closed_pnl + open_pnl
            if CAPITAL < 0:
                risk_violated = True

            if len(pnls) > 5:
                weekly_loss = pnls[-1] - pnls[-6]

                if weekly_loss < risk_limit_weekly_stop_loss:
                    #print('RiskViolation weekly_loss', weekly_loss, ' > RISK_LIMIT_WEEKLY_STOP_LOSS', risk_limit_weekly_stop_loss)
                    risk_violated = True             
            if len(pnls) > 20:
                monthly_loss = pnls[-1] - pnls[-21]

                if monthly_loss < risk_limit_monthly_stop_loss:
                    #print('RiskViolation monthly_loss', monthly_loss, ' > RISK_LIMIT_MONTHLY_STOP_LOSS', risk_limit_monthly_stop_loss)
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

                    
        if not check_for_profit_sell and not check_for_profit_buy and self.cash > 0:
            if len(pnls) == len(data.index):
                normalized = False
                tooRich = False
                data = data.assign(PnL = pd.Series(pnls, index=data.index))
                data = data.assign(Position = pd.Series(positions, index=data.index))
                if float(list(data.PnL.values)[-1] / len(list(data.PnL.values))) > min_daily_profit:
                    self.pnl_dict[name] = round(float(list(data.PnL.values)[-1] / len(list(data.PnL.values))),2)
                    self.my_stocks[name] = round(data.PnL.values[-1], 2)
                    self.final_data_dict[name] = data


                    ######################
                    ###CREATE BUY ORDER###
                    ######################
                    
                    if positions[-1] >= -1 and int(data.Close.values[-1]) != int(0) and log_prediction == -1:
                        #If the stock is already in the portfolio
                        if name in list(self.portfolio.keys()): # and self.portfolio[name]['position'] < max_stock_holdings:

                            #To close a short position
                            if self.latest_information[name]['Position'] <= -1:
                                num_shares_per_trade = abs(self.latest_information[name]['Position']) #max_stock_holdings - self.portfolio[name]['position']
                                normalized = True

                            #Entering a long position
                            else:
                                if (num_shares_per_trade * self.latest_information[name]['Bid'] < allowable_limit) and (max_stock_holdings < num_shares_per_trade + abs(self.latest_information[name]['Position'])):
                                    num_shares_per_trade = allowable_limit // num_shares_per_trade
                                    if num_shares_per_trade < 1:
                                        tooRich = True
                                    if (abs(self.latest_information[name]['Position']) + num_shares_per_trade) > max_stock_holdings:
                                        if max_stock_holdings - self.latest_information[name]['Position'] > 0:
                                            num_shares_per_trade = max_stock_holdings - self.latest_information[name]['Position']
                                        else:
                                            tooRich = True
                                            
                            if num_shares_per_trade * self.latest_information[name]['Bid'] > self.cash:
                                tooRich = True

                            if not tooRich:

                                self.orders_dict[name] = {'buy' : num_shares_per_trade,
                                                          'price' : num_shares_per_trade * data.Close.values[-1]}
                            
                                order1 = Order()
                                order1.action='BUY' #"SELL"
                                order1.orderType="LMT"
                                order1.totalQuantity= num_shares_per_trade
                                order1.lmtPrice = self.latest_information[name]['Bid']
                                order1.tif = 'IOC'
                                order1.transmit = True
                                contract1 = contractCreate(name)

                                self.placeOrder(self.get_next_brokerorderid(),contract1,order1)
                                self.length += 1


                                
                                if normalized:
                                    self.portfolio[name]['position'] = 0
                                    self.cash += num_shares_per_trade * (self.latest_information[name]['Avg Price'] - self.latest_information[name]['Bid'])
                                    self.latest_information[name]['Last Order Price'] = 0
                                else:
                                    self.portfolio[name]['position'] += num_shares_per_trade
                                    self.cash -= self.orders_dict[name]['price']
                                    self.portfolio[name]['last price'] = data.Close.values[-1]
                                    self.latest_information[name]['Last Order Price'] = self.latest_information[name]['Bid']

                                print('MEAN REVERSION: Buy order created from algorithm signal for {} @ {}...'.format(name,num_shares_per_trade))
                                
     
                        #If this if the first time entering a position with this stock   
                        else:
                            if num_shares_per_trade * self.latest_information[name]['Bid'] < allowable_limit and max_stock_holdings < num_shares_per_trade + self.latest_information[name]['Position']:
                                num_shares_per_trade = allowable_limit // num_shares_per_trade
                                if num_shares_per_trade < 1 or max_stock_holdings - self.latest_information[name]['Position'] <= 0:
                                    tooRich = True

                            if num_shares_per_trade * self.latest_information[name]['Bid'] > self.cash:
                                tooRich = True

    
                            if not tooRich:

                                self.orders_dict[name] = {'buy' : num_shares_per_trade,
                                                          'price' : num_shares_per_trade * data.Close.values[-1]}
                                    
                                order1 = Order()
                                order1.action='BUY' #"SELL"
                                order1.orderType="LMT"
                                order1.totalQuantity= num_shares_per_trade
                                order1.lmtPrice = self.latest_information[name]['Bid']
                                order1.tif = 'IOC'
                                order1.transmit = True
                                contract1 = contractCreate(name)

                                self.placeOrder(self.get_next_brokerorderid(),contract1,order1)
                                self.length += 1

                                print('MEAN REVERSION: Buy order created for {} @ {} under allowable_limit...'.format(name,num_shares_per_trade))

                                self.cash -= self.orders_dict[name]['price']
                                self.portfolio[name] = {'position': num_shares_per_trade,
                                                        'last price': data.Close.values[-1]}

                                self.latest_information[name]['Last Order Price'] = self.latest_information[name]['Bid']

                        
                    #######################
                    ###CREATE SELL ORDER###
                    #######################
                                
                    if positions[-1] <= 1 and int(data.Close.values[-1]) != int(0) and log_prediction == -1:
                        #orderExecution(name)

                        #num_shares_per_trade = 0

                        #If the stock appears in the portfolio
                        if name in list(self.portfolio.keys()):
                        

                            #To close a long position
                            if self.latest_information[name]['Position'] >= 1:
                                num_shares_per_trade = self.latest_information[name]['Position'] #max_stock_holdings - self.portfolio[name]['position']
                                normalized = True

                            #Entering a short position
                            else:
                                if num_shares_per_trade * self.latest_information[name]['Ask'] < allowable_limit and max_stock_holdings < num_shares_per_trade + self.latest_information[name]['Position']:
                                    num_shares_per_trade = allowable_limit // num_shares_per_trade
                                    if num_shares_per_trade < 1:
                                        tooRich = True
                                    if self.latest_information[name]['Position'] + num_shares_per_trade > max_stock_holdings:
                                        if max_stock_holdings - self.latest_information[name]['Position'] > 0:
                                            num_shares_per_trade = max_stock_holdings - self.latest_information[name]['Position']
                                        else:
                                            tooRich = True
                            if num_shares_per_trade * self.latest_information[name]['Ask'] > self.cash:
                                tooRich = True


                            if not tooRich and num_shares_per_trade > 0:
                                self.orders_dict[name] = {'sell' : num_shares_per_trade,
                                                          'price' : -num_shares_per_trade * data.Close.values[-1]}

                                order1 = Order()
                                order1.action='SELL' #"BUY"
                                order1.orderType="LMT"
                                order1.totalQuantity= num_shares_per_trade
                                order1.lmtPrice = self.latest_information[name]['Ask']
                                order1.tif = 'IOC'
                                order1.transmit = True
                                contract1 = contractCreate(name)

                                self.placeOrder(self.get_next_brokerorderid(),contract1,order1)
                                self.length += 1

                                print('MEAN REVERSION: Sell order created for {} @ {} to reduce position to max_stock_holdings...'.format(name,num_shares_per_trade))

                                if normalized:
                                    self.portfolio[name] = {'position': 0,
                                                            'last price': data.Close.values[-1]}
                                    self.cash += num_shares_per_trade * (self.latest_information[name]['Ask'] - self.latest_information[name]['Last Order Price'])
                                else:
                                    self.portfolio[name]['position'] -= num_shares_per_trade
                                    self.portfolio[name]['last price'] = data.Close.values[-1]
                                    self.latest_information[name]['Last Order Price'] = self.latest_information[name]['Ask']

                        
                        #If this is the first time we entered a position with this stock
                        else:

                            if num_shares_per_trade * self.latest_information[name]['Ask'] < allowable_limit and max_stock_holdings < num_shares_per_trade + abs(self.latest_information[name]['Position']):
                                num_shares_per_trade = allowable_limit // num_shares_per_trade
                                if num_shares_per_trade < 1 or max_stock_holdings - abs(self.latest_information[name]['Position']) <= 0:
                                    tooRich = True

                            if num_shares_per_trade * self.latest_information[name]['Ask'] > self.cash:
                                tooRich = True

                            if not tooRich:
                                self.orders_dict[name] = {'sell' : num_shares_per_trade,
                                                          'price' : -num_shares_per_trade * data.Close.values[-1]}
                            
                                order2 = Order()
                                order2.action='SELL' #"BUY"
                                order2.orderType="LMT"
                                order2.totalQuantity= num_shares_per_trade
                                order2.lmtPrice = self.latest_information[name]['Ask']
                                order2.tif = 'IOC'
                                order2.transmit = True
                                contract2 = contractCreate(name)

                                self.placeOrder(self.get_next_brokerorderid(),contract2,order2)
                                self.length += 1

                                print('MEAN REVERSION: Buy order created for {} @ {} under allowable_limit...'.format(name,num_shares_per_trade))

                                self.cash -= self.orders_dict[name]['price']
                                self.portfolio[name] = {'position': -num_shares_per_trade,
                                                        'last price': self.orders_dict[name]['price']}

                                self.latest_information[name]['Last Order Price'] = self.latest_information[name]['Ask']


        elif check_for_profit_sell:
            order1 = Order()
            order1.action="SELL"
            order1.orderType="LMT"
            order1.totalQuantity= abs(self.latest_information[name]['Position'])
            order1.lmtPrice = self.latest_information[name]['Close']
            order1.tif = 'IOC'
            order1.transmit = True
            contract1 = contractCreate(name)

            self.cash += abs(self.latest_information[name]['Position']) * (self.latest_information[name]['Avg Price'] - self.latest_information[name]['Last Order Price'])

            print('MEAN REVERSION: Sell order created for {} from "check for profit sell..."'.format(name))

            self.portfolio[name] = {'position': 0,
                                    'last price': data.Close.values[-1]}

            self.placeOrder(self.get_next_brokerorderid(),contract1,order1)
            self.length += 1

            self.latest_information[name]['Last Order Price'] = 0

        elif check_for_profit_buy:
            order1 = Order()
            order1.action="BUY"
            order1.orderType="LMT"
            order1.totalQuantity= abs(self.latest_information[name]['Position'])
            order1.lmtPrice = data.Close.values[-1]
            order1.tif = 'IOC'
            order1.transmit = True
            contract1 = contractCreate(name)

            self.cash += self.portfolio[name]['position'] * (self.portfolio[name]['last price'] - self.latest_information[name]['Avg Price'])

            print('MEAN REVERSION: Buy order created for {} from "check for profit buy..."'.format(name))

            self.portfolio[name] = {'position': 0,
                                    'last price': data.Close.values[-1]}

            self.placeOrder(self.get_next_brokerorderid(),contract1,order1)
            self.length += 1

            self.latest_information[name]['Last Order Price'] = 0



    ''' **************************************************************
        **************************************************************
        **************************************************************
        **************************************************************
                            TREND FOLLOWING MODULE
        **************************************************************
        **************************************************************
        **************************************************************
        **************************************************************'''


    def analyze_trend_algo(self, data, name, reqID):

        if 'Close' not in self.latest_information[name]:
            print('No update for {}'.format(name))
            return

        trend_following = True
        self.trend_following = trend_following

        reported_position = 0
        min_daily_profit = 5

        allowable_limit = self.cash * 0.05

        check_for_profit_sell = False
        check_for_profit_buy = False

        max_stock_holdings = 10
        self.max_stock_holdings = max_stock_holdings
        
        #Variables for variable risk managment
        MIN_NUM_SHARES_PER_TRADE = 1
        MAX_NUM_SHARES_PER_TRADE = 5
        INCREMENT_NUM_SHARES_PER_TRADE = 1
        num_shares_per_trade = MIN_NUM_SHARES_PER_TRADE #beginning number of shares to buy or sell on every trade
        num_shares_history = [] #history of num_shares
        abs_position_history = [] #history of absolute-position
        CAPITAL = self.cash

        #Risk Limits
        risk_limit_weekly_stop_loss = -100
        INCREMENT_RISK_LIMIT_WEEKLY_STOP_LOSS = -25
        risk_limit_monthly_stop_loss = -200
        INCREMENT_RISK_LIMIT_MONTHLY_STOP_LOSS = -50
        risk_limit_max_position = 5
        INCREMENT_RISK_LIMIT_MAX_POSITION = 1
        max_position_history = [] #history of max-position
        RISK_LIMIT_MAX_POSITION_HOLDING_TIME_DAYS = 5 #30
        risk_limit_max_trade_size = 5
        INCREMENT_RISK_LIMIT_MAX_TRADE_SIZE = 1
        max_trade_size_history = [] #history of max-trade-size
        RISK_LIMIT_MAX_TRADED_VOLUME = CAPITAL * 0.75 #1.5

        last_risk_change_index = 0

        #Variables to track/check for risk violations
        risk_violated = False

        traded_volume = 0
        current_pos = 0
        current_pos_start = 0
        
        #Variables/constants for EMA Calculation:

        NUM_PERIODS_FAST = 5 #Static time period parameter for the fast EMA
        K_FAST = 2 / (NUM_PERIODS_FAST + 1) #Static smoothing factor for fast EMA
        ema_fast = 0
        ema_fast_values = [] #Hold EMA values for visualization

        NUM_PERIODS_SLOW = 20
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
        APO_VALUE_FOR_BUY_ENTRY = 2 #APO trading signal value below which to enter buy-orders/long position
        APO_VALUE_FOR_SELL_ENTRY = -2
        MIN_PRICE_MOVE_FROM_LAST_TRADE = 2 #Minimum price change since last trade before considering trading again, this is to preven over-trading at/around same price
        MIN_PROFIT_TO_CLOSE = 2 #Minimum Open/Unrealized profit at which to close positions and lock profits
        NUM_SHARES_PER_TRADE = 1 #Number of buy/sell on every trade

        # Constants/variables that are used to compute standard deviation as a volatility measure
        SMA_NUM_PERIODS = 20 #look back period
        price_history = [] #history of prices

        data = data.append(self.iteration_update_dict[reqID], ignore_index=True)
        
        log_prediction = self.logistic_regression(data,name)

        close = list(data.Close.values)

        early_stdev = np.std(close)
        early_stdev_factor = early_stdev/15
        if early_stdev_factor == 0:
            early_stdev_factor = 1

        if self.latest_information[name]['Close'] == 0:
            risk_violated = True

        if name in self.portfolio.keys() and not self.mean_reversion:
            if  self.latest_information[name]['Close'] - self.latest_information[name]['Avg Price'] >= early_stdev_factor * MIN_PRICE_MOVE_FROM_LAST_TRADE and self.latest_information[name]['Position'] >= 1:
                check_for_profit_sell = True

            if self.latest_information[name]['Avg Price'] - self.latest_information[name]['Close'] >= early_stdev_factor * MIN_PRICE_MOVE_FROM_LAST_TRADE and self.latest_information[name]['Position'] <= -1:
                check_for_profit_buy = True

        if name not in self.portfolio.keys():
            self.portfolio[name] = {'position': 0,
                                    'last price': 0}

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

            #APO above sell entry threshold, we should sell, or
            #long from negative APO and APO has gone positiive, or position is profitable, sell to close position
            if (not risk_violated and ((apo < APO_VALUE_FOR_SELL_ENTRY / stdev_factor and abs(close_price - last_sell_price) > MIN_PRICE_MOVE_FROM_LAST_TRADE * stdev_factor)\
                                       or  (position > 0 and (apo <= 0 or open_pnl > MIN_PROFIT_TO_CLOSE / stdev_factor)))):

                ###CODE FROM OTHER ALGORITHM STARTS HERE###
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

            #APO below buy entry threshold, we should buy, or
            #short from positive APO and APO has gone negative or position is profitable, buy to close position
            elif (not risk_violated and ((apo > APO_VALUE_FOR_BUY_ENTRY / stdev_factor and abs(close_price - last_buy_price) > MIN_PRICE_MOVE_FROM_LAST_TRADE * stdev_factor)\
                                         or  (position < 0 and (apo >= 0 or open_pnl > MIN_PROFIT_TO_CLOSE / stdev_factor)))):

                ###CODE FROM OTHER ALGORITHM STARTS HERE###
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
                
            else:
                #No trade since none of the conditions were met to buy or sell
                orders.append(0)
            positions.append(position)

            ###MORE NEW CODE HERE###
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
                    #print('RiskViolation position_holding_time', position_holding_time, ' > RISK_LIMIT_MAX_POSITION_HOLDING_TIME_DAYS', RISK_LIMIT_MAX_POSITION_HOLDING_TIME_DAYS)
                    risk_violated = True

            if abs(position) > risk_limit_max_position:
                #print('RiskViolation position', position, ' > RISK_LIMIT_MAX_POSITION', risk_limit_max_position)
                risk_violated = True

            if traded_volume > RISK_LIMIT_MAX_TRADED_VOLUME:
                #print('RiskViolation traded_volume', traded_volume, ' > RISK_LIMIT_TRADED_VOLUME', RISK_LIMIT_MAX_TRADED_VOLUME)
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

            ###AGAIN, MORE NEW CODE TO MODULATE RISK
            #adjust captial funds
            CAPITAL += closed_pnl + open_pnl
            if CAPITAL < 0:
                risk_violated = True

            if len(pnls) > 5:
                weekly_loss = pnls[-1] - pnls[-6]

                if weekly_loss < risk_limit_weekly_stop_loss:
                    #print('RiskViolation weekly_loss', weekly_loss, ' > RISK_LIMIT_WEEKLY_STOP_LOSS', risk_limit_weekly_stop_loss)
                    risk_violated = True             
            if len(pnls) > 20:
                monthly_loss = pnls[-1] - pnls[-21]

                if monthly_loss < risk_limit_monthly_stop_loss:
                    #print('RiskViolation monthly_loss', monthly_loss, ' > RISK_LIMIT_MONTHLY_STOP_LOSS', risk_limit_monthly_stop_loss)
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

        if not check_for_profit_sell and not check_for_profit_buy and self.cash > 0:
            if len(pnls) == len(data.index):
                normalized = False
                tooRich = False
                data = data.assign(PnL = pd.Series(pnls, index=data.index))
                data = data.assign(Position = pd.Series(positions, index=data.index))
                if float(list(data.PnL.values)[-1] / len(list(data.PnL.values))) > min_daily_profit:
                    self.pnl_dict[name] = round(float(list(data.PnL.values)[-1] / len(list(data.PnL.values))),2)
                    self.my_stocks[name] = round(data.PnL.values[-1], 2)
                    self.final_data_dict[name] = data

                    ######################
                    ###CREATE BUY ORDER###
                    ######################
                    
                    if positions[-1] >= 1 and int(data.Close.values[-1]) != int(0) and log_prediction == -1:
                        #If the stock is already in the portfolio
                        if name in list(self.portfolio.keys()): # and self.portfolio[name]['position'] < max_stock_holdings:
                            #To close a short position
                            if self.latest_information[name]['Position'] <= -1:
                                num_shares_per_trade = abs(self.latest_information[name]['Position']) #max_stock_holdings - self.portfolio[name]['position']
                                normalized = True

                            #Entering a long position
                            else:
                                if num_shares_per_trade * self.latest_information[name]['Bid'] < allowable_limit and max_stock_holdings < num_shares_per_trade + self.latest_information[name]['Position']:
                                    num_shares_per_trade = allowable_limit // num_shares_per_trade
                                    if num_shares_per_trade < 1:
                                        tooRich = True
                                    if self.latest_information[name]['Position'] + num_shares_per_trade > max_stock_holdings:
                                        if max_stock_holdings - self.latest_information[name]['Position'] > 0:
                                            num_shares_per_trade = max_stock_holdings - self.latest_information[name]['Position']
                                        else:
                                            tooRich = True
                                            
                            if num_shares_per_trade * self.latest_information[name]['Bid'] > self.cash:
                                tooRich = True
                                
                            if not tooRich:
                                self.orders_dict[name] = {'buy' : num_shares_per_trade,
                                                          'price' : num_shares_per_trade * data.Close.values[-1]}
                            
                                order1 = Order()
                                order1.action='BUY' #"SELL"
                                order1.orderType="LMT"
                                order1.totalQuantity= num_shares_per_trade
                                order1.lmtPrice = self.latest_information[name]['Bid']
                                order1.tif = 'IOC'
                                order1.transmit = True
                                contract1 = contractCreate(name)

                                self.placeOrder(self.get_next_brokerorderid(),contract1,order1)
                                self.length += 1

                                if normalized:
                                    self.portfolio[name]['position'] = 0
                                    self.cash += num_shares_per_trade * (self.latest_information[name]['Avg Price'] - self.latest_information[name]['Bid'])
                                    self.latest_information[name]['Last Price'] = 0
                                else:
                                    self.portfolio[name]['position'] += num_shares_per_trade
                                    self.cash -= self.orders_dict[name]['price']
                                    self.portfolio[name]['last price'] = data.Close.values[-1]

                                    self.latest_information[name]['Last Price'] = self.latest_information[name]['Bid']

                                print('TREND FOLLOWING: Buy order created from algorithm signal for {} @ {}...'.format(name,num_shares_per_trade))
                                

                        #If this if the first time entering a position with this stock   
                        else:
                            if num_shares_per_trade * self.latest_information[name]['Bid'] < allowable_limit and max_stock_holdings < num_shares_per_trade + self.latest_information[name]['Position']:
                                num_shares_per_trade = allowable_limit // num_shares_per_trade
                                if num_shares_per_trade < 1 or max_stock_holdings - self.latest_information[name]['Position'] <= 0:
                                    tooRich = True

                            if num_shares_per_trade * self.latest_information[name]['Bid'] > self.cash:
                                tooRich = True
                                
                            if not tooRich:
                                self.orders_dict[name] = {'buy' : num_shares_per_trade,
                                                          'price' : num_shares_per_trade * data.Close.values[-1]}
                                order1 = Order()
                                order1.action='BUY' #"SELL"
                                order1.orderType="LMT"
                                order1.totalQuantity= num_shares_per_trade
                                order1.lmtPrice = self.latest_information[name]['Bid']
                                order1.tif = 'IOC'
                                order1.transmit = True
                                contract1 = contractCreate(name)

                                self.placeOrder(self.get_next_brokerorderid(),contract1,order1)
                                self.length += 1

                                self.latest_information[name]['Last Price'] = self.latest_information[name]['Bid']

                                print('TREND FOLLOWING: Buy order created for {} @ {} under allowable_limit...'.format(name,num_shares_per_trade))

                                self.cash -= self.orders_dict[name]['price']
                                self.portfolio[name] = {'position': num_shares_per_trade,
                                                        'last price': data.Close.values[-1]}

                        
                    #######################
                    ###CREATE SELL ORDER###
                    #######################
                                
                    if positions[-1] <= -1 and int(data.Close.values[-1]) != int(0) and log_prediction == 1:
                        #orderExecution(name)

                        #num_shares_per_trade = 0

                        #If the stock appears in the portfolio
                        if name in list(self.portfolio.keys()):

                            #To close a long position
                            if self.latest_information[name]['Position'] >= 1:
                                num_shares_per_trade = self.latest_information[name]['Position'] #max_stock_holdings - self.portfolio[name]['position']
                                normalized = True

                            #Entering a short position
                            else:
                                if num_shares_per_trade * self.latest_information[name]['Ask'] < allowable_limit and max_stock_holdings < num_shares_per_trade + self.latest_information[name]['Position']:
                                    num_shares_per_trade = allowable_limit // num_shares_per_trade
                                    if num_shares_per_trade < 1:
                                        tooRich = True
                                    if self.latest_information[name]['Position'] + num_shares_per_trade > max_stock_holdings:
                                        if max_stock_holdings - self.latest_information[name]['Position'] > 0:
                                            num_shares_per_trade = max_stock_holdings - self.latest_information[name]['Position']
                                        else:
                                            tooRich = True

                            if num_shares_per_trade * self.latest_information[name]['Ask'] > self.cash:
                                tooRich = True

                            if not tooRich and num_shares_per_trade > 0:
                                self.orders_dict[name] = {'sell' : num_shares_per_trade,
                                                          'price' : -num_shares_per_trade * data.Close.values[-1]}

                                order1 = Order()
                                order1.action='SELL' #"BUY"
                                order1.orderType="LMT"
                                order1.totalQuantity= num_shares_per_trade
                                order1.lmtPrice = self.latest_information[name]['Ask']
                                order1.tif = 'IOC'
                                order1.transmit = True
                                contract1 = contractCreate(name)

                                self.placeOrder(self.get_next_brokerorderid(),contract1,order1)
                                self.length += 1

                                print('TREND FOLLOWING: Sell order created for {} @ {} to reduce position to max_stock_holdings...'.format(name,num_shares_per_trade))

                                if normalized:
                                    self.portfolio[name] = {'position': 0,
                                                            'last price': data.Close.values[-1]}
                                    self.cash += num_shares_per_trade * (self.latest_information[name]['Ask'] - self.latest_information[name]['Last Price'])

                                    self.latest_information[name]['Last Price'] = 0
                                else:
                                    self.portfolio[name]['position'] -= num_shares_per_trade
                                    self.portfolio[name]['last price'] = data.Close.values[-1]

                                    self.latest_information[name]['Last Price'] = self.latest_information[name]['Ask']

                        
                        #If this is the first time we entered a position with this stock
                        else:
                            if num_shares_per_trade * self.latest_information[name]['Ask'] < allowable_limit and max_stock_holdings < num_shares_per_trade + abs(self.latest_information[name]['Position']):
                                num_shares_per_trade = allowable_limit // num_shares_per_trade
                                if num_shares_per_trade < 1 or max_stock_holdings - abs(self.latest_information[name]['Position']) <= 0:
                                    tooRich = True

                            if num_shares_per_trade * self.latest_information[name]['Ask'] > self.cash:
                                tooRich = True

                            if not tooRich:

                                self.orders_dict[name] = {'sell' : num_shares_per_trade,
                                                          'price' : -num_shares_per_trade * data.Close.values[-1]}
                            
                                order2 = Order()
                                order2.action='SELL' #"BUY"
                                order2.orderType="LMT"
                                order2.totalQuantity= num_shares_per_trade
                                order2.lmtPrice = self.latest_information[name]['Ask']
                                order2.tif = 'IOC'
                                order2.transmit = True
                                contract2 = contractCreate(name)

                                self.placeOrder(self.get_next_brokerorderid(),contract2,order2)
                                self.length += 1

                                self.latest_information[name]['Last Price'] = self.latest_information[name]['Ask']

                                print('TREND FOLLOWING: Buy order created for {} @ {} under allowable_limit...'.format(name,num_shares_per_trade))

                                self.cash -= self.orders_dict[name]['price']
                                self.portfolio[name] = {'position': -num_shares_per_trade,
                                                        'last price': self.orders_dict[name]['price']}


        elif check_for_profit_sell:
            order1 = Order()
            order1.action="SELL"
            order1.orderType="LMT"
            order1.totalQuantity= self.latest_information[name]['Position']
            order1.lmtPrice = self.latest_information[name]['Ask']
            order1.tif = 'IOC'
            order1.transmit = True
            contract1 = contractCreate(name)

            self.cash += self.latest_information[name]['Position'] * (self.latest_information[name]['Ask'] - self.latest_information[name]['Avg Price'])

            print('TREND FOLLOWING: Sell order created for {} from "check for profit sell..."'.format(name))

            self.portfolio[name] = {'position': 0,
                                    'last price': data.Close.values[-1]}

            self.placeOrder(self.get_next_brokerorderid(),contract1,order1)
            self.length += 1

            self.latest_information[name]['Last Price'] = 0

        elif check_for_profit_buy:
            order1 = Order()
            order1.action="BUY"
            order1.orderType="LMT"
            order1.totalQuantity= abs(self.latest_information[name]['Position'])
            order1.lmtPrice = self.latest_information[name]['Bid']
            order1.tif = 'IOC'
            order1.transmit = True
            contract1 = contractCreate(name)

            self.cash += self.latest_information[name]['Position'] * (self.latest_information[name]['Avg Price'] - self.latest_information[name]['Bid'])

            print('TREND FOLLOWING: Buy order created for {} from "check for profit buy..."'.format(name))

            self.portfolio[name] = {'position': 0,
                                    'last price': data.Close.values[-1]}

            self.placeOrder(self.get_next_brokerorderid(),contract1,order1)
            self.length += 1

            self.latest_information[name]['Last Price'] = 0



    ''' **************************************************************
        **************************************************************
        **************************************************************
        **************************************************************
                         LOGISTIC REGRESSION MODULE
        **************************************************************
        **************************************************************
        **************************************************************
        **************************************************************'''

    def logistic_regression(self,df,symbol):
        
        def create_classification_trading_condition(df,symbol):
            try:
                df['Open-Close'] = df['Open'] - df['Close']
                df['High-Low'] = df['High'] - df['Low']

                df.dropna(inplace=True)
                X = df[['Volume','Open-Close','High-Low']]
                #print('Getting Y and Z values')
                try:
                    Y = np.ravel(np.where((df['Close'].shift(-1) + (df['Close'].shift(-1) * 0.05)) >= df['Close'], 1, (np.where(df['Close'].shift(-1) == df['Close'], -1, 0))))

                    Z = np.ravel(np.where(df['Close'] > df['Close'].shift(1), 1, -1))

                except ValueError:
                    Y = 0
                    Z = 0
                    pass
                return (df,X, Y, Z)

            except ValueError:
                pass
                return df,0,0,0

        def create_train_split_group(X, Y, split_ratio=0.8):
            return train_test_split(X, Y, shuffle=False, train_size=split_ratio)

        
        df,X,Y,Z = create_classification_trading_condition(df,symbol)
            
        try:
            X_train, X_test, Y_train, Y_test = create_train_split_group(X,Y,split_ratio=0.8)
            Xz_train, Xz_test, Z_train, Z_test = create_train_split_group(X,Z,split_ratio=0.8)
            log_r = linear_model.LogisticRegression(penalty='l2', tol=0.01, solver='liblinear') #LogisticRegression(solver='lbfgs')
            log_r.fit(X_train,Y_train)

            accuracy_train = accuracy_score(Y_train, log_r.predict(X_train))
            accuracy_test = accuracy_score(Y_test, log_r.predict(X_test))

            #print('{} for {} training accuracy score: \n'.format('Logistic Regression',symbol),accuracy_train)
            #print('{} for {} test accuracy score: \n'.format('Logistic Regression',symbol),accuracy_test)

            df['LogR_Predicted_Signal'] = log_r.predict(X)
            df['Returns']= np.log(df['Close'] / df['Close'].shift(1))

            log_r.fit(Xz_train,Z_train)
            df['LogR_Predicted_Future_Signal'] = log_r.predict(X)

        except ValueError or KeyError or TypeError:
            return 0

        return df['LogR_Predicted_Future_Signal'].iloc[-1]


    ''' **************************************************************
        **************************************************************
        **************************************************************
        **************************************************************
                         CUSTOM DAY TRADE MODULE
        **************************************************************
        **************************************************************
        **************************************************************
        **************************************************************'''

##    self.latest_information[self.reference_dict[x[0]]] = {'Open': 0,
##                                                                      'High': 0,
##                                                                      'Low': 0,
##                                                                      'Close': 0,
##                                                                      'Volume': 0,
##                                                                      'Count': 0,
##                                                                      'Bid': 0,
##                                                                      'Ask': 0,
##                                                                      'Position': 0,
##                                                                      'Avg Price': 0}
##    place_order(self,order_action,order_type,quantity,order_tif,symbol)

    def custom_daily_trading_algo(self,symbol):

        
        
        max_stock_holdings = self.max_stock_holdings
        
        num_shares_per_trade = 10

        if 'Close' not in self.latest_information[symbol] or self.latest_information[symbol]['Close'] == 0:
            print('No update for {}'.format(symbol))
            return

        if 'Position' in self.latest_information[symbol]:
            # Check previous short positions
            if self.latest_information[symbol]['Position'] <= -1:
                if self.latest_information[symbol]['Ask'] - self.latest_information[symbol]['Avg Price'] <= -0.50:
                    
                    self.place_order(order_action='BUY',
                                     order_type='LMT',
                                     quantity=abs(self.latest_information[symbol]['Position']),
                                     price=self.latest_information[symbol]['Ask'],
                                     order_tif='IOC',
                                     symbol=symbol)
                    print('CUSTOM DAILY: Placed Buy Order to Normalize Position for {}...'.format(symbol))
                    return

            # Check previous long positions
            if self.latest_information[symbol]['Position'] >= 1:
                if self.latest_information[symbol]['Bid'] - self.latest_information[symbol]['Avg Price'] >= 0.50:
                    self.place_order(order_action='SELL',
                                     order_type='LMT',
                                     quantity=abs(self.latest_information[symbol]['Position']),
                                     price=self.latest_information[symbol]['Bid'],
                                     order_tif='IOC',
                                     symbol=symbol)
                    print('CUSTOM DAILY: Placed Sell Order to Normalize Position for {}...'.format(symbol))
                    return

        
        
        if self.latest_information[symbol]['Close'] == 0:
            print('##########################################')
            print('Did not get new information for {}...'.format(symbol))
            print('##########################################')
            return
        else:
            self.daily_dict[symbol].append(self.latest_information[symbol])
        #print(self.daily_dict[symbol])

        if len(self.daily_dict[symbol]) < 3:
            return

        else: # 3 <= len(self.daily_dict[symbol]) <= 40:
            df = pd.DataFrame(self.daily_dict[symbol])
            print(df)

##        else:
##            df = pd.DataFrame(self.daily_dict[symbol][1:])

            
        # New Columns
        df['High-Low'] = df['High'] - df['Low']
        df['Bid-Ask'] = df['Bid'] - df['Ask']

        # Short and Long Moving Averages
        SMA_high = df['High'].iloc[:5].mean()
        LMA_high = df['High'].mean()
        SMA_low = df['Low'].iloc[:5].mean()
        LMA_low = df['Low'].mean()
        SMA_close = df['Close'].iloc[:5].mean()
        LMA_close = df['Close'].mean()
        SMA_highlow = df['High-Low'].iloc[:5].mean()
        LMA_highlow = df['High-Low'].mean()
        SMA_bidask = df['Bid-Ask'].iloc[:5].mean()
        LMA_bidask = df['Bid-Ask'].mean()

        # Standard Deviations
        std_high = df['High'].std(ddof=(df.shape[1] - 1))
        std_low = df['Low'].std(ddof=(df.shape[1] - 1))
        std_close = df['Close'].std(ddof=(df.shape[1] - 1))
        std_highlow = df['High-Low'].std(ddof=(df.shape[1] - 1))
        std_bidask = df['Bid-Ask'].std(ddof=(df.shape[1] - 1))
                
        # Enter Long Position
        if SMA_close + (std_close / 25) > LMA_close:
            try:
                num_shares_per_trade = (self.cash * 0.1) // self.latest_information[symbol]['Ask']
            except ZeroDivisionError:
                num_shares_per_trade = 1
            if (df.Close.iloc[-1] - df.Close.iloc[-2]) > self.latest_information[symbol]['Close']*(std_close / 25) and num_shares_per_trade > 0:
                if num_shares_per_trade + abs(self.latest_information[symbol]['Position']) <= max_stock_holdings:
                    if self.latest_information[symbol]['Ask'] * num_shares_per_trade <= self.cash:
                        self.place_order(order_action='BUY',
                                         order_type='LMT',
                                         quantity=num_shares_per_trade,
                                         price=self.latest_information[symbol]['Ask'],
                                         order_tif='IOC',
                                         symbol=symbol)
                        print('CUSTOM DAILY: Placed Buy Order to Enter Position for {}...'.format(symbol))

        # Enter Short Position
        if SMA_close - (std_close / 25) < LMA_close:
            try:
                num_shares_per_trade = (self.cash * 0.1) // self.latest_information[symbol]['Bid']
            except ZeroDivisionError:
                num_shares_per_trade = 1
            if (df.Close.iloc[-1] - df.Close.iloc[-2]) < self.latest_information[symbol]['Close']*(-std_close / 25) and num_shares_per_trade > 0:
                if num_shares_per_trade + abs(self.latest_information[symbol]['Position']) <= max_stock_holdings:
                    if self.latest_information[symbol]['Bid'] * num_shares_per_trade <= self.cash:
                        self.place_order(order_action='SELL',
                                         order_type='LMT',
                                         quantity=num_shares_per_trade,
                                         price=self.latest_information[symbol]['Bid'],
                                         order_tif='IOC',
                                         symbol=symbol)
                        print('CUSTOM DAILY: Placed Sell Order to Enter Position for {}...'.format(symbol))

        

        return

    
        
        


    
    
######################################################################
######################################################################
###################### Below is TestApp Class ########################
######################################################################
######################################################################


class TestApp(TestWrapper, TestClient):
    #Intializes our main classes 
    def __init__(self, ipaddress, portid, clientid):
        TestWrapper.__init__(self)
        TestClient.__init__(self, wrapper=self)

        self.init_error()

        

        #Connects to the server with the ipaddress, portid, and clientId specified in the program execution area
        self.connect(ipaddress, portid, clientid)

        #Initializes the threading
        thread = Thread(target = self.run)
        thread.start()
        setattr(self, "_thread", thread)

        

        #self.init_error()

        self.init_open_orders()

        self.set_cash_and_timestamp()
        self.init_storage_stuff()
        
        self.portfolio = {}
        self.account = {}
        self.brokenCounter = 0
        self.open_order_list = []

        self.interationCounter = 0

        self.cash = 5000
        self.max_stock_holdings = 50

        self.myOrderID = self.get_next_brokerorderid()

        

        #Starts listening for errors 
        #self.init_error()
    

'''##################################################################'''
########################################################################
'''##################################################################'''
########################################################################
'''##################################################################'''
########################################################################
'''*********        Below is the program execution        ***********'''
'''##################################################################'''
########################################################################
'''##################################################################'''
########################################################################
'''##################################################################'''
########################################################################

if __name__ == '__main__':

    print("before start")

    # Specifies that we are on local host with port 7497 (paper trading port number)
    app = TestApp("127.0.0.1", 7497, 0)
    #app.init_open_orders()

    # A printout to show the program began
    print("The program has begun")

###THIS BLOCK RUNS THE CODE 
    current_time = datetime.datetime.now().time()
    open_time = current_time.replace(hour=6, minute=0, second=0, microsecond=0) 
    close_time = current_time.replace(hour=13, minute=00, second=0, microsecond=0)    

    print('Starting loop...')
    while open_time <= current_time <= close_time:
        start_time = time.time()
        app.run_algorithm()
        end_time = time.time()
        current_time = datetime.datetime.now().time()
        print('Realizing profit and checking if broken...')
        app.realize_profit()
        app.check_if_broken()

    print('Today, the algorithm broke {} times....'.format(app.check_if_broken()))
    #app.map_performance()
###THIS BLOCK RUNS THE CODE


###THIS BLOCK GETS HISTORICAL PRICE DATA

    # long histories
    for subdir, dirs, files in os.walk('{}/Dropbox (ASU)/Algorithmic_Trading/Historical_IB_Data'.format(str(Path.home()))):
        for file in files:
            if file.endswith('.pkl'):
                s = file[:-4]
                starttime = time.time()
                app.request_historical_price(s)
                elapsed_time = time.time() - starttime
                
                if elapsed_time < 0.75:
                    time.sleep(0.75 - elapsed_time)
                    
    # short histories
    for subdir, dirs, files in os.walk('{}/Dropbox (ASU)/Algorithmic_Trading/Historical_IB_Data'.format(str(Path.home()))):
        for file in files:
            if file.endswith('.pkl'):
                s = file[:-4]
                starttime = time.time()
                app.request_short_historical_price(s)
                elapsed_time = time.time() - starttime
                
                if elapsed_time < 0.75:
                    time.sleep(0.75 - elapsed_time)

###THIS BLOCK GETS HISTORICAL PRICE DATA

    # Optional disconnect. If keeping an open connection to the input don't disconnet
    app.disconnect()


