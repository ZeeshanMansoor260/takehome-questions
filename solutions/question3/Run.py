import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras.models import model_from_json
import h5py
import argparse
import sys

#************
#If want to train the model. Change train to 1
train_mode = 0
if len(sys.argv)==1:
	train_mode=0
else:
	if sys.argv[1] != 1:
		print("Usage:\nTraining Mode: python Run.py 1 \nTesting Mode: python Run.py")
		exit()
	else:
		train_mode=1
#************

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def train_lstm(dropout):
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back), recurrent_dropout=dropout))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1)
    return model

#predict single measument for different models
def experiment(model):
    forcast_x = numpy.reshape(testX[-2],(1,1,1))
    forcast = model.predict(forcast_x)
    fx = scaler.inverse_transform(testX[-2])
    fr = scaler.inverse_transform(forcast)
    #print("fx: ", fx, " fr: ", fr)
    return fx,fr



# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('data/candy_production.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))




n_dropout = [0.0, 0.2, 0.4, 0.6]
predictions = []
#run/train the model using different dropouts
for dropout in n_dropout:
    if train_mode == 1: #if want to train the model
        model = train_lstm(dropout)
        model_json = model.to_json()
        with open("models/lstm_dropout"+str(dropout)+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("models/lstm_dropout"+str(dropout)+".h5")
   
    else:#will run in a testing mode
        json_file = open("models/lstm_dropout"+str(dropout)+".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("models/lstm_dropout"+str(dropout)+".h5")
        
    fx,fr = experiment(model)
    predictions.append(fr)
mean = numpy.array(predictions).mean()
std = numpy.array(predictions).std()

print("mean: ", mean, " uncertainty: ", std)
