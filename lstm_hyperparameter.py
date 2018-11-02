from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dropout, Flatten

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


def create_model(learn_rate=0.01, momentum=0, optimizer='adam',activation='relu'):
    # default values
    activation='relu' # or linear
    dropout_rate=0.0 # or 0.2
    init_mode='uniform'
    weight_constraint=0 # or  4
    optimizer='adam' # or SGD

    # design network
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(None, 60), activation=activation,))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(256))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mae', optimizer=optimizer)

    return model

if __name__ == '__main__':
    # load dataset
    dataset = read_csv('pricedata/masterFrame.csv', index_col=0)#, header=0, index_col=0)
    dataset['target'] = dataset.proc1close
    del dataset['proc1close']
    data_cols = dataset.columns
    print(dataset.head())

    n_features = len(dataset.columns)
    n_train = round(0.7*len(dataset))


    print('Size of dataset: ', len(dataset), 'No. of features: ', n_features)


    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # specify the number of lag hours
    n_hours = 1

    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_hours, 1)
    print(reframed.head())
    print(reframed.shape)

    # split into train and test sets
    values = reframed.values
    n_train_hours = n_train
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    # split into input and outputs
    n_obs = n_hours * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    print(train_X.shape, len(train_X), train_y.shape)

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


    # create model
    model = KerasRegressor(build_fn=create_model, batch_size=1000, epochs=10)


    # Use scikit-learn to grid search
    activation = ['relu', 'tanh', 'sigmoid']  # softmax, softplus, softsign
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    learn_rate = [0.001, 0.01]
    dropout_rate = [0.0, 0.1]
    weight_constraint = [1, 2, 3, 4, 5]
    neurons = [100, 200, 256]
    init = ['uniform', 'lecun_uniform', 'normal']
    optimizers = ['SDG', 'Adam']

    #For Full Hyper paraneter Optimization un comment this section
    #activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']  # softmax, softplus, softsign
    #momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    #learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    #dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #weight_constraint = [1, 2, 3, 4, 5]
    #neurons = [100, 200, 256]
    #init = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    #optimizers = ['SDG', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

    # grid search epochs, batch size
    epochs = [10, 50, 100]  # add 50, 100, 150 etc
    batch_size = [100, 500]  # add 5, 10, 20, 40, 60, 80, 100 etc
    param_grid = dict(epochs=epochs, batch_size=batch_size, learn_rate=learn_rate, optimizer=optimizers, activation = activation) #optimizer=optimizers, batch_size=batch_size,  learn_rate=learn_rate,
                      #momentum=momentum)
    #param_grid = dict(optimizer=optimizers)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(train_X, train_y)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

