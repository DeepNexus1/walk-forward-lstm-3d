import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import tensorflow.keras.backend as K
import joblib


# split the dataset into train and test samples with the walk-forward approach
def walk_forward_split(df, train_bars, test_bars): 
    total_bars = len(df)
    number_of_sets = np.int((total_bars-train_bars)/test_bars)
    train_sets = []
    test_sets = []
    for i in range(number_of_sets):
        test_start_bar = total_bars - (i+1)*test_bars
        test_end_bar = total_bars - i*test_bars
        train_start_bar = test_start_bar - train_bars
        test_sets.append(df.iloc[test_start_bar:test_end_bar, :])
        train_sets.append(df.iloc[train_start_bar:test_start_bar, :])
    return train_sets, test_sets


# generate extra cumulative sum feature and sets of input and output sequences
def to_sequences(dataset, n_pre, n_post):
    # select all fields with 'input' in the title
    X = dataset.filter(like='input').to_numpy()
    Y = dataset.filter(['target']).to_numpy()   
    X_list, Y_list = [], []

    # the number of column in array the cumulative sum feature will be created from
    to_cumsum = 2
    for i in range(len(X) - n_pre - n_post + 1):
        X_list.append(X[i:i + n_pre])
        # add cumulative sum feature to every array
        X_list[i] = np.c_[X_list[i], np.cumsum(X_list[i][:, to_cumsum])]
        Y_list.append(Y[i + n_pre:i + n_pre + n_post])
    # convert list to 3d array    
    dataX_3d = np.array(X_list)     
    dataY_3d = np.array(Y_list)      
    return dataX_3d, dataY_3d


# scale input sequences
def train_preprocessing(train_set, n_pre, n_post):
    dataX_3d, dataY_3d = to_sequences(train_set, n_pre, n_post)
    n_features = dataX_3d.shape[2]
    dataX_2d = dataX_3d.reshape(-1, n_features)    
    scaler = preprocessing.RobustScaler().fit(dataX_2d)
    dataX_2d_scaled = scaler.transform(dataX_2d)    
    dataX_3d_scaled = dataX_2d_scaled.reshape(np.array(dataX_3d.shape))
    return dataX_3d_scaled, dataY_3d, scaler, n_features


# train the model
def build_model(dataX_3d_scaled, dataY_3d, n_features, model_filepath):
    optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # LSTM Sample Neural Network
    input = tf.keras.Input(shape=(n_pre, n_features))
    output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, input_shape=(n_pre, n_features),
                                                                return_sequences=False, return_state=False))(input)
    output = tf.keras.layers.Activation('relu')(output)
    output = tf.keras.layers.RepeatVector(n_post)(output)
    output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, return_state=False))(output)
    output = tf.keras.layers.Activation('relu')(output)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(output)
    output = tf.keras.layers.Activation('sigmoid')(output)
    model = tf.keras.Model(inputs=input, outputs=output) 
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=model_filepath, monitor='val_loss',
                                                      save_weights_only=False, verbose=1, save_best_only=True)
    epochs = 2
    batch_size = 256
    model.fit(dataX_3d_scaled, dataY_3d, validation_split=.10,
              batch_size=batch_size, epochs=epochs,
              callbacks=[checkpointer],
              shuffle=False)


# make a forecast
def forecast(scaler, test_set, train_set, n_features, model_filepath, test_set_end_extension, extension_start_bar):  
    # extension for the test set from the last rows of the train set
    test_set_start_extension = train_set.iloc[extension_start_bar:,:]
    # merge test set with two extensions on the start and the end
    test_set_extended = pd.concat([test_set_start_extension, test_set, test_set_end_extension], axis=0)
    dataX_3d, dataY_3d = to_sequences(test_set_extended, n_pre, n_post)   
    dataX_2d = dataX_3d.reshape(-1, n_features)    
    dataX_2d_scaled = scaler.transform(dataX_2d) 
    dataX_3d_scaled = dataX_2d_scaled.reshape(np.array(dataX_3d.shape)) 
    model = tf.keras.models.load_model(model_filepath)
    y_predict = model.predict(dataX_3d_scaled, batch_size=256, verbose=1)
    y_target = np.reshape(dataY_3d, (-1, n_post, 1))    
    # obtain predictions and transform to 2-dimensions
    y_predict = y_predict.transpose(0, 1, 2).reshape(-1, n_post)
    # obtain original prediction targets and transform to 2-dimensions
    y_target = y_target.transpose(0, 1, 2).reshape(-1, n_post)    
    return y_predict, y_target


# calculate the quality metric for the model 
def evaluate_forecast(y_predict, y_target):    
    scores_per_n_post = list()

    # calculate an ROC AUC score for each n_post
    for i in range(n_post):
        score = roc_auc_score(y_target[:, i], y_predict[:, i])
        scores_per_n_post.append(score)

    # transform to 1-dimensions    
    targets_1d = y_target.reshape(-1)
    predictions_1d = y_predict.reshape(-1)
    score_total = roc_auc_score(targets_1d, predictions_1d)
    return scores_per_n_post, score_total


# the final function that joins previous stages: preprocessing, building model, and evaluating the forecast
def evaluate_model(train_sets, test_sets, n_pre, n_post):
    scores_per_n_post_all, score_total_all, predictions_by_bar_and_n_post_all = list(), list(), list()
    # an empty dataframe to extend the first test set with. All the next sets will have n additional rows in the end
    test_set_end_extension = pd.DataFrame()
    # the number of the start bar for the test set extension
    extension_start_bar = len(train_sets[0]) - n_pre
    # counter to title saved files
    start = 1
    # loop to process every train/test pair
    for train_set, test_set in zip(train_sets, test_sets):
        dataX_3d_scaled, dataY_3d, scaler, n_features = train_preprocessing(train_set, n_pre, n_post)
        scaler_file = 'scaler{:03d}.save'.format(start)
        joblib.dump(scaler, scaler_file)
        model_filepath = 'model{:03d}.hdf5'.format(start)
        build_model(dataX_3d_scaled, dataY_3d, n_features, model_filepath)
        y_predict, y_target = forecast(scaler, test_set, train_set, n_features, model_filepath, test_set_end_extension,
                                       extension_start_bar)
        np.savetxt('prediction{:03d}.csv'.format(start), y_predict, delimiter=",")
        np.savetxt('target{:03d}.csv'.format(start), y_target, delimiter=",") 
        predictions_by_bar_and_n_post_all.append(y_predict)
        scores_per_n_post, score_total = evaluate_forecast(y_predict, y_target)
        model_performance_per_n = pd.DataFrame({'n_post': range(1, n_post+1), 
                                                'score': scores_per_n_post})
        model_performance_per_n.to_csv('model_performance_per_n{:03d}.csv'.format(start), index=False)
        scores_per_n_post_all.append(scores_per_n_post)
        score_total_all.append(score_total)
        # extension in the end for the next test set in the loop
        test_set_end_extension = test_set.iloc[:n_post - 1,:]
        start += 1
    return scores_per_n_post_all, score_total_all, predictions_by_bar_and_n_post_all


np.random.seed(1335)  # for reproducibility
print('Reading csv...')
# USER ADDS DATA FILE HERE
df = pd.read_csv("________.csv", index_col='Date', parse_dates=True).loc['________':'_______']  # date range
# rename columns if required
df = df.reset_index()
df = df.rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close"})
# sort rows by date just in case
df = df.sort_values('Date')

# save index as a column to use for checking results of walk_forward_split function later
df['index_number'] = df.index
# calculate service fields using for creating features
df['stage0'] = np.log(df.Close/df.Open).fillna(0.0)
# create features
df['input1'] = np.where(df.stage0 > 0, 1, 0)
df['input2'] = np.where(df.input1 > 0, 1, -1)
rolling_step = 5
df['input3'] = df.input2.rolling(rolling_step).sum()
#  create target variable
df['target'] = np.where(df.Close > df.Close.shift(1), 1, 0)
#  cut off initial nans
df = df.drop(list(range(rolling_step)))

# set the parameters for train/test and input/output sequences
train_bars = 1000
test_bars = 100
n_pre = 64
n_post = 10
train_sets, test_sets = walk_forward_split(df, train_bars, test_bars)


# the main function, generate evaluations and predictions
scores_per_n_post, score_total, predictions_by_bar_and_n_post_all = evaluate_model(train_sets, test_sets, n_pre, n_post)

# calculate average per n_post score (AUC ROC)
scores_per_n_post_avg = [np.mean(x) for x in scores_per_n_post]

# calculate general score (AUC ROC)
score_total_avg = np.mean(score_total)

# calculate average prediction by bar and n_post
predictions_by_bar_and_n_post_aggr = [np.mean(x, axis=1) for x in predictions_by_bar_and_n_post_all]

# merge predictions by bar and n_post and its aggregations
predictions_by_bar_and_n_post_all_aggr = [np.hstack([i, j.reshape(-1, 1)]) for i, j in
                                          zip(predictions_by_bar_and_n_post_all, predictions_by_bar_and_n_post_aggr)]

# name columns for predictions_by_bar_and_n_post
column_names = ['n_post_' + str(x) for x in range(1, n_post + 1)]

# name the last column with average values
column_names.append('n_post_avg')

# convert list of arrays to list of dataframes
predictions_by_bar_and_n_post_list_of_dfs = [pd.DataFrame(x, columns=column_names)
                                             for x in predictions_by_bar_and_n_post_all_aggr]

# merge test sets and predictions
test_sets_with_prediction_list = [pd.concat([i.reset_index(), j.reset_index(drop=True)], axis=1).set_index('index')
                                  for i, j in zip(test_sets, predictions_by_bar_and_n_post_list_of_dfs)]

# make dataframe from list of dataframes, sort by dates
test_sets_with_prediction = pd.concat(test_sets_with_prediction_list).sort_values('Date')
test_sets_with_prediction.to_csv('___________.csv', index=False)  # name your csv as you wish
print(scores_per_n_post_avg)
print(score_total_avg)
