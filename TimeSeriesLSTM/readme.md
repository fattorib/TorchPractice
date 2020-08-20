# LSTM Time Series Prediction

Practing PyTorch by working through different RNN archcitectures for Seq2Seq prediction. Originally, I started with AAPL. However, this might not be the best stock to work with. The last few years of data (what our model is being tested on) are quite unlike the first 4000 or so training entries. Because of this, normalizing the data can be problematic. 

Might be better to either: 
1. Choose a penny stock to avoid normalization issues
2. Change train/test split to 90/10 instead of 80/20. This could improve the normalization quality
