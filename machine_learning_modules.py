import pandas as pd
from pandas_datareader import data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def load_financial_data(start_date, end_date, output_file):
    try:
        df = pd.read_pickle(output_file)
        print('File data found...reading GOOG data')
    except FileNotFoundError:
        print('File not found...downloading the GOOG data')
        df = data.DataReader('GOOG', 'yahoo', start_date, end_date)
        df.to_pickle(output_file)

    return df

def create_classification_trading_condition(df):
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
    df.dropna(inplace=True)
    X = df[['Open-Close','High-Low']]
    Y = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
    ###This assumes the close price tomorrow is not the same as today.
    ###To account for this, simply add a 3rd categorical var '0'
    return (df,X, Y)

def create_regression_trading_condition(df):
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
    df.dropna(inplace=True)
    X = df[['Open-Close','High-Low']]
    
    #create regression variable
    Y = df['Close'].shift(-1) - df['Close']
    
    #create target variable
    df['Target'] = df['Close'].shift(1) - df['Close']

    return (df,X,Y)

def create_train_split_group(X, Y, split_ratio=0.8):
    return train_test_split(X, Y, shuffle=False,
                            train_size=split_ratio)

def calculate_return(df, split_value, symbol):
    cum_goog_return = df[split_value:]['{}_Returns'.format(symbol)].cumsum() * 100
    df['Strategy_Returns'] = df['{}_Returns'.format(symbol)] * df['Predicted_Signal'].shift(1)

    return df, cum_goog_return

def calculate_strategy_return(df, split_value, symbol):
    cum_strategy_return = df[split_value:]['Strategy_Returns'].cumsum() * 100

    return cum_strategy_return

def plot_chart(cum_symbol_return, cum_strategy_return, symbol):
    plt.figure(figsize=(10,5))
    plt.plot(cum_symbol_return, label='{} Returns'.format(symbol))
    plt.plot(cum_strategy_return, label='Strategy Returns')
    plt.legend()
    plt.grid()
    plt.show()

def sharpe_ratio(symbol_returns, strategy_returns):
    strategy_std = strategy_returns.std()
    sharpe = (strategy_returns - symbol_returns) / strategy_std

    return sharpe.mean()

##############################
###LINEAR REGRESSION MODELS###
##############################
goog_data = load_financial_data(start_date='2001-01-01',
                                end_date='2018-01-01',
                                output_file='goog_data_large.pkl')

goog_data, X, Y = create_regression_trading_condition(goog_data)

###PLOT TWO FEATURE MATRIX VS. TARGET (FUTURE PRICE)
##pd.plotting.scatter_matrix(goog_data[['Open-Close', 'High-Low', 'Target']],
##                           grid=True, diagonal='kde')
##plt.suptitle('Two Feature vs. Future Price')
##plt.show()

X_train, X_test, Y_train, Y_test = create_train_split_group(X,Y,split_ratio=0.8)

X_test.drop(X_test.tail(1).index,inplace=True)
Y_test.drop(Y_test.tail(1).index,inplace=True)

ols = linear_model.LinearRegression()
ols.fit(X_train, Y_train)

print('Coefficients: \n', ols.coef_)

print('Training Data')
print('Mean Squared Error: %.2f' % mean_squared_error(Y_train, ols.predict(X_train)))
#Explained Variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_train, ols.predict(X_train)))

print('Test Data')
print('Mean Squared Error: %.2f' % mean_squared_error(Y_test, ols.predict(X_test)))
#Explained Variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, ols.predict(X_test)))


###PREDICT PRICES AND CALCULATE STRATEGY RETURNS###

goog_data['Predicted_Signal'] = ols.predict(X)
goog_data['GOOG_Returns'] = np.log(goog_data['Close'] / goog_data['Close'].shift(1))

goog_data, cum_goog_return = calculate_return(goog_data, split_value=len(X_train),symbol='GOOG')
cum_strategy_return = calculate_strategy_return(goog_data, split_value=len(X_train),symbol='GOOG')

#plot_chart(cum_goog_return,cum_strategy_return,symbol='GOOG')

print('Sharpe Ratio: \n', sharpe_ratio(cum_strategy_return, cum_goog_return))

###Lasso Model (L1)
lasso = linear_model.Lasso(alpha=0.1) #if regularization parameter is increased to 0.6, first coeffecient shrinks to zero
lasso.fit(X_train,Y_train)
print('Lasso Coefficients: \n', lasso.coef_)

###Ridge Model (L2)
ridge = linear_model.Ridge(alpha=10000)
ridge.fit(X_train, Y_train)
print('Ridge Coefficients: \n', ridge.coef_)

print('End of linear regression models')

###########################
###CLASSIFICATION MODELS###
###########################

###K-Nearest Neighbors Model
goog_data,X,Y = create_classification_trading_condition(goog_data)
X_train, X_test, Y_train, Y_test = create_train_split_group(X,Y,split_ratio=0.8)

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,Y_train)


accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
accuracy_test = accuracy_score(Y_test, knn.predict(X_test))

print('{} training accuracy score: \n'.format('K-Nearest Neighbors'),accuracy_train)
print('{} test accuracy score: \n'.format('K-Nearest Neighbors'),accuracy_test)

goog_data['Predicted_Signal'] = knn.predict(X)

cum_goog_return = calculate_return(goog_data, split_value=len(X_train),symbol='GOOG')
cum_strategy_return = calculate_strategy_return(goog_data, split_value=len(X_train),symbol='GOOG')

#plot_chart(cum_goog_return[1], cum_strategy_return, symbol='GOOG')

###Support Vector Machine Model

svc = SVC()
svc.fit(X_train,Y_train)

accuracy_train = accuracy_score(Y_train, svc.predict(X_train))
accuracy_test = accuracy_score(Y_test, svc.predict(X_test))

print('{} training accuracy score: \n'.format('Support Vector Classifier'),accuracy_train)
print('{} test accuracy score: \n'.format('Support Vector Classifier'),accuracy_test)

goog_data['Predicted_Signal'] = svc.predict(X)

cum_goog_return = calculate_return(goog_data, split_value=len(X_train),symbol='GOOG')
cum_strategy_return = calculate_strategy_return(goog_data, split_value=len(X_train),symbol='GOOG')

#plot_chart(cum_goog_return[1], cum_strategy_return, symbol='GOOG')

###

log_r = linear_model.LogisticRegression()
log_r.fit(X_train,Y_train)

accuracy_train = accuracy_score(Y_train, log_r.predict(X_train))
accuracy_test = accuracy_score(Y_test, log_r.predict(X_test))

print('{} training accuracy score: \n'.format('Logistic Regression'),accuracy_train)
print('{} test accuracy score: \n'.format('Logistic Regression'),accuracy_test)

goog_data['Predicted_Signal'] = log_r.predict(X)

cum_goog_return = calculate_return(goog_data, split_value=len(X_train),symbol='GOOG')
cum_strategy_return = calculate_strategy_return(goog_data, split_value=len(X_train),symbol='GOOG')

#plot_chart(cum_goog_return[1], cum_strategy_return, symbol='GOOG')




















