import pandas as pd
import numpy as np


stock_prices = pd.read_csv('NOK.csv')


close_prices = stock_prices['Close']


train_data = np.array(close_prices[0:4000])

import matplotlib.pyplot as plt


frame1 = plt.gca()
plt.style.use('ggplot')
plt.title('NOK Price (USD)')
plt.ylabel('Price')
plt.plot(np.linspace(0,1000,1000),train_data[3000:])
frame1.axes.get_xaxis().set_ticks([])