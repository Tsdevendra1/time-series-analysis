# Sin Wave time series analysis

This repository contains a tensorflow model to perform time series analysis.

The data used in this model is a self generated sin wave, with the option of using different frequencies for the generation of te data.

The main component of this model is the use of a LSTM cell.

Images of how different learning rates effected the performance of the model can be seen below.
 
The image below shows the model performing with a **0.0001** learning rate.
![alt text](https://github.com/Tsdevendra1/time-series-analysis/blob/master/graphs/0.0001.png)

The image below shows the model performing with a **0.001** learning rate.
![alt text](https://github.com/Tsdevendra1/time-series-analysis/blob/master/graphs/0.001.png)

Examples of other learning rates can be seen in:
https://github.com/Tsdevendra1/time-series-analysis/blob/master/graphs

From viewing the different performances, the conclusion can be drawn that the 0.001 learning rate performed best for this particular data. 
