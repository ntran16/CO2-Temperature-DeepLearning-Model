# CO2-Temperature-DeepLearning-Model

First, run CheckLag.m to estimate the lag of CO2 vs. Temperature. This function check for variance between different lags and pick one with lowest variance. 

Then, run Narnet.m to fit a RNN model with feedback delay (input the range of feedback delay appropriately.)

Then, run TimeDelayNet.m to fit a RNN model with input delay (CO2 input delay). Specify the range and the initial delay. Model and lag with the best RMSE (minimum RMSE) is the best model and the best time lag.

Finally, run the NARX model using parameters found in Narnet.m and TimeDelayNet.m to build the CO2 model. The fit probably is not perfect.

Note: close feedback loop perform much worst than open loop.
