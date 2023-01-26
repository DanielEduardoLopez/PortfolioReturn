<p align="center">
	<img src="Images/Header.png?raw=true" width=80% height=80%>
</p>


# Prediction of a Stocks Portfolio Return using Recurrent Neural Networks
#### By Daniel Eduardo López

**[LinkedIn](https://www.linkedin.com/in/daniel-eduardo-lopez)**

**[Github](https://github.com/DanielEduardoLopez)**


____
### **1. Introduction**
A **portfolio** is a set of financial instruments that belongs to a investor, and where each instrument has a given weight within it (i.e. the percentage in the portfolio value). Modern Portfolio Theory (MPT) allows to establish a relationship framework between risk and return for an adequate decision-making and, in particular, to maximize the return while minimizing the risk by means of an efficient diversification accomplished by the selection of different instruments with distinct returns, risks and values (Baca Urbina & Marcelino Aranda, 2016).

Even though the Time Series Analysis has been traditionally used to forecast portfolio returns, the role of neural networks is increasing as several deep learning models have been successfully built to optimize investment portfolios and its return and risk (Freitas, De Souza & de Almeida, 2009).

**Recurrent Neural Networks (RNN)** are a "a class of neural networks that are naturally suited to processing time-series data and other sequential data" (DiPietro & Hager, 2020). In this sense, this kind of Artificial Neural Networks (ANN) allows to process variable-length sequences, in which the recurrency is expressed by using the output of a given neuron(s) as feedback to the neuron(s) of the former layer (DiPietro & Hager, 2020; Gupta & Raza, 2019). This lets the RNN to "learn" patterns and trends from the data.

For simplicity, in the present project, only 10 stocks selected from the biggest companies in the world by market capitalization (Johnston, 2022) were used to build the portfolio, with the goal of using RNN to predict its returns over time and assess the accuracy of the predictions.

____
### **2. General Objective**
To predict the return of a stocks portfolio return using Recurrent Neural Networks and assess its accuracy.
____
### **3. Research Question**
What is the return of a stocks portfolio according to a model based on Recurrent Neural Networks and its accuracy?
____
### **4. Hypothesis**
The prediction of a stocks portfolio return using Recurrent Neural Networks will yield a R<sup>2</sup> larger than 70%.
____
### **5. Abridged Methodology**
The methodology of the present study is based on Rollin’s Foundational Methodology for Data Science (Rollins, 2015):

1. **Analytical approach**: Building and evaluation of regression model.
2. **Data requirements**: Stock daily values from 2020/01/01 to 2023/01/09.
3. **Data collection**: Data was retrieved from Yahoo Finance by using the Python library [yfinance](https://pypi.org/project/yfinance/).
4. **Data exploration**: Data was explored with Python 3 and its libraries Numpy, Pandas, Matplotlib and Seaborn.
5. **Data preparation**: The calculation of the daily returns and the optimization of the portfolio using the Markowitz's methodology (Starke, Edwards & Wiecki, 2016) was carried out with Python 3 and its libraries Numpy, Pandas, Matplotlib and Cvxopt.
5. **Data modeling**: A RNN was created and trained in Python 3 and its libraries Numpy, Pandas, Sklearn, Keras and Tensorflow were utilized.
6. **Evaluation**: The model predictions were primarily evaluated through the R<sup>2</sup>, RMSE and MAE.

___
### **6. Main Results**

#### **6.1 Data Collection**
For simplicity, in the present project, only 10 stocks selected from the biggest companies in the world by market capitalization <a href="https://www.investopedia.com/biggest-companies-in-the-world-by-market-cap-5212784">(Johnston, 2022)</a> were used to build the portfolio:
1. Apple Inc. (**AAPL**)
2. Saudi Aramco (**2222.SR**) 
3. Microsoft Corp. (**MSFT**)
4. Alphabet Inc. (**GOOGL**)
5. Amazon.com Inc. (**AMZN**)
6. Tesla Inc. (**TSLA**)
7. Berkshire Hathaway Inc. (**BRK-B**)
8. NVIDIA Corp. (**NVDA**)
9. Taiwan Semiconductor Manufacturing Co. Ltd. (**TSM**)
10. Meta Platforms Inc. (**META**)

On the other hand, stock value data was retrieved from the previous three years, i.e., **from 2020/01/01 to 2023/01/09** by using the Python library yfinance.

#### **6.2 Data Exploration**
Stock value data retrieved from Yahoo Finance was explored through some simple visualizations.

<p align="center">
	<img src="Images/Fig1_HistStockCloseValues.png?raw=true" width=80% height=80%>
</p>

The EDA suggests that **the market has been bearish since mids 2021**. At the end of 2022 and begining of 2023, all stocks but BRK-B have decreased their value to the 2020 levels.

Notwithstanding the above, the data collected from *Yahoo Finance* through **yfinance** seems to be complete and consistent.

#### **6.3 Data Preparation**
Stock data value was cleaned and processed to calculate the **daily close value returns** which serves as a basis for the portfolio optimization. The **descriptive statistics of the daily returns** for each asset are as follows:

AAPL |	AMZN |	BRK-B |	GOOGL |	META |	MSFT |	NVDA |	TSLA |	TSM |	2222.SR
:---: |	:---: |	:---: |	:---: |	:---: |	:---: |	:---: |	:---: |	:---: |	:---:
count |	727.000000 |	727.000000 |	727.000000 |	727.000000 |	727.000000 |	727.000000 |	727.000000 |	727.000000 |	727.000000 |	727.000000
mean |	0.001057 |	0.000202 |	0.000607 |	0.000586 |	-0.000143 |	0.000728 |	0.001907 |	0.002920 |	0.000731 |	0.000194
std |	0.024311 |	0.025279 |	0.016443 |	0.022033 |	0.031346 |	0.022239 |	0.035705 |	0.046334 |	0.025512 |	0.012754
min |	-0.128647 |	-0.140494 |	-0.095921 |	-0.116341 |	-0.263901 |	-0.147390 |	-0.184521 |	-0.210628 |	-0.140341 |	-0.090909
25% |	-0.011410 |	-0.013169 |	-0.007101 |	-0.009822 |	-0.013412 |	-0.009711 |	-0.018861 |	-0.022199 |	-0.014852 |	-0.004307
50% |	0.000246 |	0.000529 |	0.000621 |	0.001054 |	0.000648 |	0.000681 |	0.002897 |	0.001768 |	-0.000232 |	0.000000
75% |	0.014410 |	0.012433 |	0.008086 |	0.012451 |	0.015185 |	0.012255 |	0.022738 |	0.024981 |	0.014375 |	0.004283
max |	0.198469 |	0.135359 |	0.116099 |	0.092412 |	0.175936 |	0.142169 |	0.171564 |	0.198949 |	0.126522 |	0.098765

**TSLA** had the **highest average return** (0.292%); whereas **META** has **the lowest one** (-0.014%) in the analyzed time period.

On the other hand, **TSLA** also exhibited the **largest volatility** (4.633%); whereas **2222.SR** exhibited **the lowest one** (1.275%).

Thus, even though the **TSLA** has yielded the highest average return in the last three years, it comes with a high risk. In this sense, the Beta of this company is 2.03 according to <a href="https://finance.yahoo.com/quote/TSLA?p=TSLA&.tsrc=fin-srch">Yahoo Finance</a>, which means that TSLA overreacts to the changes in the market.

Moreover, the **returns over time** for each stock are shown below:

<p align="center">
	<img src="Images/Fig2_HistStockReturns.png?raw=true" width=80% height=80%>
</p>

Indeed, from the figures above, **TSLA** and **NVDA** daily returns are **the most volatile**; whereas the **2222.SR** and **BRK-B** ones are **the less**.

Then, once the daily returns were calculated, the weight of each stock was optimized in the portfolio to maximize the return thereof using Convex optimization. This part of the analysis is based on <a href="https://github.com/quantopian/research_public/blob/master/research/Markowitz-blog.ipynb">Starke, Edwards & Wiecki (2016)</a>. The **Efficient Frontier** chart is as follows:

<p align="center">
	<img src="Images/Fig3_EfficientFrontier.png?raw=true" width=60% height=60%>
</p>

And the **weights of each asset** in the portfolio are shown below:

<p align="center">
	<img src="Images/Fig4_PortfolioComposition.png?raw=true" width=60% height=60%>
</p>

Interestingly, the portfolio optimization yielded that most of the portfolio should be **TSLA** stocks (about 99%). 

Even though these results contradicts the idea of having a diversified portfolio, for the purposes of the present analysis the outcome from the optimization algorithm were used in the later steps of this project.

After the portfolio was optimized, the historical returns thereof were calculated based on the stock close value over time and the optimized weights. So, the historical returns of the optimized portfolio are as follows:

<p align="center">
	<img src="Images/Fig5_HistReturnsOptPort.png?raw=true" width=60% height=60%>
</p>

As expectable, the historical returns of the optimized portfolio mirror the behavior of **TSLA** as most of the portfolio comprises such asset.

#### **6.4 Data Modeling**
A Recurrent Neural Network was built and trained to predict the return of the optimized stocks portfolio. The following architecture was proposed:

1. A first **Lambda layer** to convert the input windowed dataset into a tensor.
2. A first **Long Short-Term Memory layer** with 256 units.
3. A second **Long Short-Term Memory layer** with 128 units.
4. A third **Long Short-Term Memory layer** with 64 units.
5. A **full connection layer** with 32 nodes and a ReLU activation function, and a dropout of 15%.
6. A **full connection layer** with 128 nodes and a ReLU activation function, and a dropout of 15%.
7. A **full connection layer** with 64 nodes and a ReLU activation function, and a dropout of 15%.
8. An **output layer** of 1 node.
9. An **Lambda layer** to scale the output by 100 units.

The code of the model in Python is as follows:
```python
# Model

# Initialization of the RNN
model = Sequential()

# Transformation of the input dataset into a tensor
model.add(Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]))

# LSTM
model.add(Bidirectional(LSTM(256, return_sequences=True)))

# LSTM
model.add(Bidirectional(LSTM(128, return_sequences=True)))

# LSTM
model.add(Bidirectional(LSTM(64)))

# Full Connection
model.add(Dense(32, input_shape=[window_size], activation="relu"))
model.add(Dropout(0.15))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.15))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.15))
model.add(Dense(1))

# Output Scaling
model.add(Lambda(lambda x: x * 100.0))

# Model Summary
model.summary()
```

```bash
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lambda_4 (Lambda)           (None, None, 1)           0         
                                                                 
 bidirectional_4 (Bidirectio  (None, None, 512)        528384    
 nal)                                                            
                                                                 
 bidirectional_5 (Bidirectio  (None, None, 256)        656384    
 nal)                                                            
                                                                 
 bidirectional_6 (Bidirectio  (None, 128)              164352    
 nal)                                                            
                                                                 
 dense_7 (Dense)             (None, 32)                4128      
                                                                 
 dropout_5 (Dropout)         (None, 32)                0         
                                                                 
 dense_8 (Dense)             (None, 128)               4224      
                                                                 
 dropout_6 (Dropout)         (None, 128)               0         
                                                                 
 dense_9 (Dense)             (None, 64)                8256      
                                                                 
...
Total params: 1,365,793
Trainable params: 1,365,793
Non-trainable params: 0
```

Furthermore, in order to get better predictions, the **learning rate was optimized**. To do so, a pivot or initial learning rate of $1\times10^{-8}$ was defined, which was slightly incremented by each epoch with the help of a learning rate scheduler from Tensorflow. 

On the other hand, as optimizer, the **Stochastic Gradient Descent** algorithm was used with an arbitrarily momentum of 0.9 in order to help accelerate the gradient vectors in the right directions and lead the model to a faster learning.

Moreover, in order to reduce the sensitivity to outliers wich are present in the dataset, the <a href="https://en.wikipedia.org/wiki/Huber_loss">**Huber loss function**</a> was used in the model compilation. 

The Learning Rate vs. Loss plot is shown below:

<p align="center">
	<img src="Images/Fig7_LearningRateOpt.png?raw=true" width=60% height=60%>
</p>

Thus, from the plot above,  the approximate best learning rate is $5\times10^{-5}$.

Once the architecture of the RNN was defined and an optimal learning rate was selected, the model was trained using the **Huber loss** as the loss function, the **Stochastic Gradient Descent with Momentum** as the optimizer and keeping track of the **MSE**, **MAE**, and $r^2$ metrics too. 300 epochs were used.

```python
# Model compilation
model.compile(loss = tf.keras.losses.Huber(), 
              optimizer = tf.keras.optimizers.SGD(learning_rate = lr, momentum= 0.9), 
              metrics=["mse","mae", tfa.metrics.RSquare()]

# Model Fitting
history = model.fit(dataset, epochs = 300, verbose = 1)

```

After the model was trained, to generate the predictions, the original time series data was first split into a training and a testing set. To do so, the first 80% of the data was used for the training set and the remaining 20% for the testing set. Then, the forecasting was performed with predict method. The results are shown below:

<p align="center">
	<img src="Images/Fig8_ReturnPredictions.png?raw=true" width=60% height=60%>
</p>

In view of the plot above, the RNN has predicted a constant return of 0.7362% for all the time points, which suggests that the model was not able to catch a pattern from the data. In comparison, the mean return of the portfolio was 0.292%. So, the RNN yielded a higher prediction. 


#### **6.5 Evaluation**
The predictions of the Recurrent Neural Network were evaluated using the Root Mean Squared Error (RMSE), the Mean Absolute Error (MAE), and Coefficient of Determination
( $r^{2}$ ).

Metric | Value
:---: | :---:
Mean Absolute Error | 0.03362
Root Mean Squared Error | 0.04421
$r^{2}$ | -0.20520

Even though the MAE and the RMSE are **low** in absolute terms, it is insightful that the $r^{2}$ is **negative**, which implies that the model yields worst results than the mean of the test set. So, as suggested by the plot of the predicted versus the actual returns, the model cannot be deemed as satisfactory.

On the other hand, the plot of the metrics over epochs is as follows:

<p align="center">
	<img src="Images/Fig9_MetricsReturns.png?raw=true" width=60% height=60%>
</p>

As expectable, the RMSE and MAE stabilized after some epochs. However, it is noteworthy that the $r^{2}$ is always negative and have a considerable fluctuation over the epochs. Again, this suggests that the model is unsatisfactory.

___
### **7. Conclusions**
The trained model was unable to recognize a pattern from the historical returns of the optimized portfolio. In this sense, even though low values of RMSE and MAE were obtained due to the scale of the returns, a negative $r^{2}$ was yielded by the model. In consequence, the model generated worse predictions than the mean of the historical returns. 

In this sense, it is suggested to train the model using the historical values of the optimized portfolio instead of the returns. Other options to explore are to change the size of the dataset windows and the architecture of the model.

___
### **8. Bibliography**
- **Baca-Urbina, G., & Marcelino-Aranda, M. (2016)**. *Ingeniería financiera*. Mexico City: Grupo Editorial Patria.
- **DiPietro, R. & Hager, G. D. (2020)**. Deep learning: RNNs and LSTM. In S. K. Zhou, D. Rueckert & G. Fichtinger (Eds.), *Handbook of Medical Image Computing and Computer Assisted Intervention* (pp. 503-519). The Elsevier and MICCAI Society Book Series. https://doi.org/10.1016/B978-0-12-816176-0.00026-0
- **Freitas, F. D., De Souza, A. F. & de Almeida, A. R. (2009)**. Prediction-based portfolio optimization model using neural networks. *Neurocomputing*, 72(10–12): 155-2170. https://doi.org/10.1016/j.neucom.2008.08.019.
- **Gupta, T. K. & Raza, K. (2019)**. Optimization of ANN Architecture: A Review on Nature-Inspired Techniques. In N. Dey, S. Borra, A. S. Ashour & F. Shi (Eds.), *Machine Learning in Bio-Signal Analysis and Diagnostic Imaging*. Academic Press. https://doi.org/10.1016/B978-0-12-816086-2.00007-2
- **Johnston, M. (2022)**. *Biggest Companies in the World by Market Cap*. https://www.investopedia.com/biggest-companies-in-the-world-by-market-cap-5212784
- **Rollins, J. B. (2015)**. *Metodología Fundamental para la Ciencia de Datos. Somers: IBM Corporation.* https://www.ibm.com/downloads/cas/WKK9DX51
- **Starke, T., Edwards, D. & Wiecki, T. (2016)**. *A tutorial on Markowitz portfolio optimization in Python using cvxopt*. https://github.com/quantopian/research_public/blob/master/research/Markowitz-blog.ipynb

___
### **9. Description of Files in Repository**
File | Description 
--- | --- 
1_PortfolioReturn_DataCollectionPrep.ipynb | Notebook with the Python code for collecting the stock value data, calculating the daily returns and optimizing the portfolio.
1_PortfolioReturn_DataCollectionPrep.html | Notebook in HTML format.
2_PortfolioReturn_DataModelingEval.ipynb | Notebook with the Python code for modeling and evaluating the RNN for predicting the optimized portfolio returns.
2_PortfolioReturn_DataModelingEval.html | Notebook in HTML format.
Hist_Opt_Returns.csv | Historical returns of the optimized portfolio in a CSV format.
requirements.csv | Python requirements file.
