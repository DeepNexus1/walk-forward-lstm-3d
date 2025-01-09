import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, Embedding, Reshape, BatchNormalization
from sklearn.metrics import accuracy_score

# split the dataset into train and test samples with the walk-forward approach
def walk_forward_split(df, train_bars, test_bars): 
    total_bars = len(df)
    number_of_sets = np.int((total_bars-train_bars)/test_bars)
    train_sets = []
    test_sets = []
    train_dates = []
    test_dates = []
    for i in range(number_of_sets):
        test_start_bar = total_bars - (i+1)*test_bars
        test_end_bar = total_bars - i*test_bars
        train_start_bar = test_start_bar - train_bars
        test_sets.append(df[test_start_bar:test_end_bar])
        train_sets.append(df[train_start_bar:test_start_bar])
        train_dates.append(dates[train_start_bar:test_start_bar])
        test_dates.append(dates[test_start_bar:test_end_bar])
    return train_sets, test_sets, train_dates, test_dates

# generate sets of input and output sequences
def to_sequences(dataset, dataset_dates, tickers_columns, n_pre, n_post):
    n_features = dataset.shape[2]
    X = dataset[:, :, :n_features - 1]
    Y = pd.DataFrame(dataset[:, :, n_features-1], index=dataset_dates, columns=tickers_columns)
    X_list, Y_list = [], []
    
    for i in range(len(X) - n_pre - n_post + 1):
        df_x = X[i:i + n_pre].transpose(1, 0, 2)
        df_y = Y[i + n_pre:i + n_pre + n_post]
        X_list.append(df_x)
        Y_list.append(df_y)
    # convert list to 3d array    
    dataX_3d = np.concatenate(X_list, axis = 0) 
    dataY = (pd.concat(Y_list).unstack().sort_index(level=['date', 'ticker'], ascending=[True, True]))
    tickers = pd.factorize(dataY.index.get_level_values('ticker'))[0]
    months = dataY.index.get_level_values('date').month
    months = pd.get_dummies(months, columns=['month'], prefix='month')

    dataX = [
        dataX_3d,
        tickers,
        months
    ]     
    return dataX, dataY

# train the model
def build_model(dataX, dataY, model_filepath):
    n_pre = dataX[0].shape[1]
    n_features = dataX[0].shape[2]
    n_tickers = len(np.unique(dataX[1]))
    n_months = len(np.unique(dataY.index.get_level_values('date').month))
    returns = Input(shape=(n_pre, n_features), name='Returns')
    tickers = Input(shape=(1,), name='Tickers')    
    months = Input(shape=(n_months,), name='Months')    
    lstm1_units = 25
    lstm2_units = 10    
    lstm1 = LSTM(units=lstm1_units,
                 input_shape=(n_pre,
                              n_features),
                 name='LSTM1',
                 dropout=.2,
                 return_sequences=True)(returns)    
    lstm_model = LSTM(units=lstm2_units,
                 dropout=.2,
                 name='LSTM2')(lstm1)       
    ticker_embedding = Embedding(input_dim=n_tickers,
                                 output_dim=5,
                                 input_length=1)(tickers)
    ticker_embedding = Reshape(target_shape=(5,))(ticker_embedding)    
    merged = concatenate([lstm_model,
                          ticker_embedding,
                          months], name='Merged')
    
    bn = BatchNormalization()(merged)
    hidden_dense = Dense(10, name='FC1')(bn)    
    output = Dense(4, name='Output', activation='softmax')(hidden_dense)    
    rnn = Model(inputs=[returns, tickers, months], outputs=output)        
    optimizer = tf.keras.optimizers.RMSprop(lr=0.001,
                                            rho=0.9,
                                            epsilon=1e-08,
                                            decay=0.0)   
    rnn.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=optimizer,
                metrics=['accuracy'])    
    checkpointer = ModelCheckpoint(filepath=model_filepath,
                                   verbose=1,
                                   monitor='val_accuracy',
                                   mode='max',
                                   save_best_only=True)
   
    early_stopping = EarlyStopping(monitor='val_accuracy',
                                  patience=5,
                                  restore_best_weights=True,
                                  mode='max')    
    rnn.fit(dataX,
            dataY,
            epochs=4,
            batch_size=32,
            validation_split=.10,
            callbacks=[early_stopping, checkpointer],
            verbose=1)

# make a forecast
def forecast(test_set, test_dates, model_filepath):  
    dataX, y_target = to_sequences(test_set, test_dates, tickers_columns, n_pre, n_post)   
    model = tf.keras.models.load_model(model_filepath)
    y_predict = model.predict(dataX, batch_size=256, verbose=1)
    y_predict = y_predict.argmax(axis=1)
    return y_predict, y_target

# calculate the quality metric for the model 
def evaluate_forecast(y_predict, y_target):    
    score = accuracy_score(y_target, y_predict)
    return score

# the final function that joins previous stages
def evaluate_model(train_sets, train_dates, test_sets, test_dates, n_pre, n_post):
    score_all = list()    
    start = 1
    # loop to process every train/test pair
    for train_set, train_date, test_set, test_date in zip(train_sets, train_dates, test_sets, test_dates):
        dataX, dataY = to_sequences(train_set, train_date, tickers_columns, n_pre, n_post)
        model_filepath = 'model{:03d}.hdf5'.format(start)
        build_model(dataX, dataY, model_filepath)
        y_predict, y_target = forecast(test_set, test_date, model_filepath)
        np.savetxt('prediction{:03d}.csv'.format(start), y_predict, delimiter=",")
        np.savetxt('target{:03d}.csv'.format(start), y_target, delimiter=",") 
        score = evaluate_forecast(y_predict, y_target)
        score_all.append(score)
        # extension in the end for the next test set in the loop
        start += 1
    return score_all


prices = (pd.read_hdf('______.h5', '_______')  # load data in .h5 format with appropriate key
          .reset_index(level='ticker')  # 'ticker' level given as example index level; depends on user's data structure
          .loc['______':]  # use to select a portion of the data, not necessarily the entire file
          )

# select first 100 tickers to limit the dataset
tickers = pd.factorize(prices.ticker)[1][:100].tolist()

prices = (prices.query('ticker == @tickers')
          .set_index('ticker', append=True)
          .sort_index(ascending=True)
          )

# transform the dataframe to list of dataframes where every dataframe is separate feature with timestamp and tickers
prices_list = list()
for i in range(len(prices.columns)):
    name = prices.columns[i]
    df = prices[name].unstack().resample('W').last()
    prices_list.append(df)

# EXAMPLE FEATURE
# add log_returns as feature
adj_close_list_id = prices.columns.get_loc('adj_close') - 1
returns = prices_list[adj_close_list_id].pct_change() 
log_returns = np.log(returns + 1)
prices_list.append(log_returns)  
for price in prices_list:
    price.drop(price.head(1).index, inplace=True)

# EXAMPLE FEATURE
# add rolling_sum_log_returns as feature
log_returns_list_id = len(prices_list) - 1
rolling_step = 5
rolling_sum_log_returns = prices_list[log_returns_list_id].rolling(rolling_step).sum()
prices_list.append(rolling_sum_log_returns)
for price in prices_list:
    price.drop(price.head(rolling_step - 1).index, inplace=True)

# EXAMPLE TARGET
# add target
forward_return = 4   
target = prices_list[log_returns_list_id].shift(1).rolling(forward_return).sum()
for i in range(forward_return, len(target)):
    ranked = pd.qcut(target.iloc[i], 4, labels=False)
    target.iloc[i] = ranked
prices_list.append(target)
for price in prices_list:
    price.drop(price.head(forward_return).index, inplace=True)
 
# select only needed features and target
features = list(range(log_returns_list_id, len(prices_list)))
dates = prices_list[0].index
tickers_columns = prices_list[0].columns
df = np.stack([prices_list[i] for i in features]).transpose(1, 2, 0)

# set the parameters for train/test and input/output sequences
train_bars = 60
test_bars = 60
n_pre = 10
n_post = 1


train_sets, test_sets, train_dates, test_dates = walk_forward_split(df, train_bars, test_bars)
score_all = evaluate_model(train_sets, train_dates, test_sets, test_dates, n_pre, n_post)
score_avg = np.mean(score_all)
print(score_avg)
