The model will try to predict uncertainty of a single prediction using a LSTM model.


System Specification:
Python 3.6
keras 2.2.4
h5py 2.9.0
numpy 1.16.1
sklearn 0.19.1

Instructions for running the code:
if want to run in a test mode:
python Run.py

if want to run in training mode
python Run.py 0

Output
mean:  111.12692  confidence:  1.9513785

Question:
Assuming you are writing a LSTM model, how do you predict the next value in the above time-series on a
given confidence interval? (i.e. 90%)
Answer:
To predict the next value in a time series data within the confidence interval we can use a model similar to Bayesian Neural Networks.
To implement this concept in LSTM, we can use dropouts to train the model. Dropout will randomly drop some units
along with their connections. This can be considered as roughly equivalent to performing Bernoulli distribution and sampling from the Network.
After repeating this process for several iterations during testing, we will get different prediction values and compute mean and variance to get confidence interval for each step.
