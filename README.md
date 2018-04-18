Slides and demonstration examples for the talk at PyCon UA, 2018


## An Introduction to Time Series Forecasting with Python

* Andrii Gakhov, ferret go GmbH
* https://www.gakhov.com
* PyCon UA, Kharkiv, April 28-29, 2018


### Talk Abstract


Time series is an important instrument to model, analyze and predict data collected over time. In this talk, we learn the basic theoretical concepts without going deep into mathematical aspects, study different models, and try them in practice using StatsModels, Prophet, scikit-learn, and keras.

### Talk Description

Nowadays, it is hard to find a company that doesn’t collect various time-dependent data in different forms, for instance, it can be a daily number of visitors and monthly sales for online stores, available resources and stock for factories, number of food poisoning cases for hospitals, and so on. And the reason why all that data is carefully collected, because it can provide meaningful insides not only about the past but can be used to predict and prepare for the future.

In this presentation, we discuss how to analyze and forecast those data, that is called time series. Many people already did that many times while trying to predict the weather on the weekend, guessing the currency exchange rate for tomorrow, or just by expecting great discounts on Christmas sales. Of course, some patterns are truly obvious, like weekly or monthly changes, and overall tendency, others are not. However, all these aspects can be formalized and learned automatically using the power of mathematics and computer science.

The first part is dedicated to the theoretical introduction of time series, where listeners can learn or refresh in memory the essential aspects of time series’ representations, modeling, and forecasting. In the second part, we dive into the most popular time series forecast models - stochastic models (e.g., Autoregressive integrated moving average (ARIMA)), artificial neural networks (e.g., seasonal recurrent neural network) and Support Vector Machines (SVR). Along the way, we show at practice how these models can be applied to a real-world dataset of UK visits by providing examples using such popular Python libraries as StatsModels, Prophet, scikit-learn, and keras.

With these guidelines in mind, you should be better equipped to deal with time series in your everyday work and opt-in for the right tools to analyze them.

To follow the talk it's not required any prior knowledge of time series analysis, but the basic understanding of mathematics and machine learning approaches could be quite helpful.


GitHub repository: https://github.com/gakhov/pycon-ua-2018/


## Structure

* PDF slides: https://github.com/gakhov/pycon-ua-2018/slides/
* Datasets: https://github.com/gakhov/pycon-ua-2018/data/

### Generic dataset exploration

To demonstrate the mentioned in the presentation models, I use the following dataset:

#### OS visits to UK (All visits)
The dataset represents the monthly total number of visits to the UK by overseas residents (in thousands)<br>from January 1980 to October 2017.
Source: [Office for National Statistics](https://www.ons.gov.uk/peoplepopulationandcommunity/leisureandtourism/timeseries/gmaa/ott)

* Jupyter Notebook: https://github.com/gakhov/pycon-ua-2018/look-into-the-data.ipynb

### Stochastic Models

As a stochastic model for seasonal time series, in the presentation, I describe the Seasonal Autoregressive Integrated Moving Average (SARIMA) model.

* Jupyter Notebook: https://github.com/gakhov/pycon-ua-2018/stochastic-models.ipynb
* Python package: [Statsmodels](https://www.statsmodels.org/)


Another example of this family is Facebook Prophet model, that based on the additive time series decomposition, some ideas from ARIMA and Bayesian analysis.

* Jupyter Notebook: https://github.com/gakhov/pycon-ua-2018/prophet.ipynb
* Python package: [Facebook Prophet](https://github.com/facebook/prophet)

### Artificial Neural Networks

I consider recurrent artificial neural networks with Long Short-Term Memory (LSTM) architecture and demonstrate how to create and fit Seasonal Artificial Neural Network (SANN).

* Jupyter Notebook: https://github.com/gakhov/pycon-ua-2018/artificial-neural-networks.ipynb
* Python package: [keras](https://keras.io/)

### Support Vector Machines

Support Vector Machines can be used to forecast time series, particularly the Support Vector Machine Regressors (SVMR) that are demonstrated in the presentation.

* Jupyter Notebook: https://github.com/gakhov/pycon-ua-2018/support-vector-machines.ipynb
* Python package: [Scikit-learn](http://scikit-learn.org)


Dependencies
---------------------

* Python 3.3+ (http://python.org/download/)


Install with pip
--------------------

Installation requires a working build environment that can be build automatically using `make` utility:

.. code:: bash

    $ make
    $ make run

After these commands your default browser should open a Jupyter notebook's index page.


License
-------

MIT License


Source code
-----------

* https://github.com/gakhov/pycon-ua-2018/


Author
-------

* `Andrii Gakhov <andrii.gakhov@gmail.com>`

