
The model will try to predict the uncertainty of a single prediction using an LSTM model.

## System Specification
- Python 3.6
- keras 2.2.4
- h5py 2.9.0
- numpy 1.16.1
- sklearn 0.19.1

## Instructions for Running the Code
If you want to run in a test mode. It will use the already trained models from the model's directory
```
python Run.py
```

If you want to run in training mode
```
python Run.py 0
```
## Uncertainty of a Single Prediction
```
Predicting for: 2017-08-01  true value: 114.06
mean:  111.12692  uncertainty:  1.9513785
```

## Assuming you are writing a LSTM model, how do you predict the next value in the above time-series on a given confidence interval? (i.e. 90%)
To predict the next value in a time series data within the confidence interval we can use a model similar to Bayesian Neural Networks. To implement this concept in LSTM, we can use dropouts to train the model. Dropout will randomly drop some units
along with their connections. This can be considered as roughly equivalent to performing Bernoulli distribution and sampling from the Network. After repeating this process for several iterations during testing, we will get different prediction values and compute mean and variance to get confidence interval for each step.

## Overall Model Accuracy
Used 70% for training without dropout
```
Test Score: 7.21 RMSE
```
Visualization of model's performance
![Alt text](figures/test.jpg?raw=true "Title")
