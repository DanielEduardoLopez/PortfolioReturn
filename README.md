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

For simplicity, in the present project, only 10 stocks selected from the biggest companies in the world by market capitalization (Johnston, 2022) were used to build the portfolio, with the goal of using RNN to predict its return and assess the accuracy of said prediction.

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
2. **Data requirements**: Stock daily values from Jan 2019 to Dec 2022.
3. **Data collection**: Data was retrieved from the [Yahoo Finance's API for Ptyhon](https://pypi.org/project/yfinance/)
4. **Data exploration**: Data was explored with Python 3 and its libraries Numpy, Pandas, Matplotlib and Seaborn.
5. **Data preparation**: The calculation of the daily returns and the optimization of the portfolio using the Markowitz's methodology (Starke, Edwards & Wiecki, 2016) was carried out with Python 3 and its libraries Numpy, Pandas, Matplotlib and Cvxopt.
5. **Data modeling**: A RNN was created and trained in Python 3 and its libraries Numpy, Pandas, Sklearn, Keras and Tensorflow were utilized.
6. **Evaluation**: The model predictions were primarily evaluated through the R<sup>2</sup>, RMSE and MAE.

___
### **6. Main Results**

#### **6.1 Data Collection**
Pending...

#### **6.2 Data Exploration**
Pending...

#### **6.3 Data Preparation**
Pending...

#### **6.4 Data Modeling**
Pending...

#### **6.5 Evaluation**
Pending...

___
### **7. Conclusions**
Pending...

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
1_PortfolioReturn_DataCollection.ipynb | Notebook with the Python code for collecting the stock value data.

