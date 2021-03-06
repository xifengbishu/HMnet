# HMnet
A Hybrid Multivariate Deep Learning Networks for Multistep ahead Sea Level Anomaly Forecasting

The accumulated remote sensing data of altimeters and scatterometers have provided new opportunities to ocean states forecasting and improve our knowledge of ocean-atmosphere exchanges. Studies on multivariate, multi-step, spatiotemporal sequence forecast of sea level anomaly (SLA) for different modalities, however, remain problematic. In this paper, we present a novel Hybrid and Multivariate deep neural network, named HMnet3, which can be used for SLA forecasting in the South China Sea (SCS). First, a spatiotemporal sequence forecasting network is trained by an improved Convolutional Long Short-Term Memory (ConvLSTM) network using the channel-wise attention mechanism and multivariate data from 1993 to 2015. Then, a time series forecasting network is trained by an improved LSTM network, which is realized by the ensemble empirical mode decomposition. Finally, the two networks are combined by a successive correction method to produces SLA forecasts for lead times of up to 15 days, with a special focus on the open sea and coastal regions of SCS. During the testing period of 2016-2018, the performance of HMnet3, which inputs sea surface temperature anomaly (SSTA), wind speed anomaly (SPDA) and SLA data, is much better than those of state-of-the-art dynamic and statistical (ConvLSTM, persistence and climatology) forecast models. The stricter testbed of trial simulation experiments with realtime datasets are investigated, where the eddy classification metrics of HMnet3 compares favourably for all properties, especially for those of small scale eddies.


## CRAM-ConvLSTM
An improved channel residual attention-based ConvLSTM network involving DMS (CRAM-ConvLSTM), which is trained by multivariate daily remote sensing observations from T-14 to T, is used for the SLA spatiotemporal sequence forecasting in the SCS

## EEMD-LSTM
A novel EEMD-LSTM approach, a hybrid of EEMD and LSTM, is proposed to the SLA time series forecasting. 

Step 1: Determines the relative maxima for the RMSE array produced by the CRAM-ConvLSTM, which returns 20 sites of relative maxima values found in five areas (see Table 1) to construct the SLA time series forecasting networks with the EEMD-LSTM method. 

Step 2: The original time series of input   from each site, which contains a set of SLA, SSTA, and SPDA time series,  are decomposed into five intrinsic mode functions (IMFs) to obtain more realistic and physically meaningful signals (Huang et al. 2019; Liu et al. 2019), yielding relatively stationary IMF that can be readily modeled by LSTM. 

Step 3: We respectively, use an improved multivariate time series forecasting with LSTMs network to fit each IMF from T-14 to T. Then, each IMF is forecasted for SLA time series of  using the corresponding LSTM. 

Step 4: The SLA time series forecasting result are calculated by the sum of the forecasting values of every IMF. 

## Successive Correction
successive correction method, which integrates the multivariate time series forecasting with EEMD-LSTM and the multivariate spatiotemporal sequence forecasting with CRAM-ConvLSTM, is proposed to improve the quality of SLA forecasting. The successive correction method, which uses relative scales of horizontal distance and SLA difference between CRAM-ConvLSTM-forecasting value and EEMD-LSTM-forecasting value



## Test

ncl successive_correction.ncl
