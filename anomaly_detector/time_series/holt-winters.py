import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

'''
https://machinelearningmastery.com/exponential-smoothing-for-time-series-forecasting-in-python/

Alpha: Smoothing factor for the level.
Beta: Smoothing factor for the trend.
Gamma: Smoothing factor for the seasonality.
Trend Type: Additive or multiplicative.
Dampen Type: Additive or multiplicative.
Phi: Damping coefficient.
Seasonality Type: Additive or multiplicative.
Period: Time steps in seasonal period.

- trend: The type of trend component, as either “add” for additive or “mul” 
 for multiplicative. Modeling the trend can be disabled by setting it to None.
- damped: Whether or not the trend component should be damped,
 either True or False.
- seasonal: The type of seasonal component, as either “add” for additive 
 or “mul” for multiplicative. Modeling the seasonal component 
 can be disabled by setting it to None.
- seasonal_periods: The number of time steps in a seasonal period,
 e.g. 12 for 12 months in a yearly seasonal structure (more here).
 The model can then be fit on the training data by calling the fit() function.
 This function allows you to either specify the smoothing coefficients 
 of the exponential smoothing model or have them optimized. By default,
 they are optimized (e.g. optimized=True). These coefficients include:
- smoothing_level (alpha): the smoothing coefficient for the level.
- smoothing_slope (beta): the smoothing coefficient for the trend.
- smoothing_seasonal (gamma): the smoothing coefficient 
 for the seasonal component.
- damping_slope (phi): the coefficient for the damped trend.
  Additionally, the fit function can perform basic data preparation
   prior to modeling; specifically:
- use_boxcox: Whether or not to perform a power transform of 
 the series (True/False) or specify the lambda for the transform.
'''

df = pd.read_csv('data/cpu_util.csv')
df.head()

# df.plot.line(x='YEAR_MONTH_SALE_DATE', y='COUNT_YEAR_MONTH_SALE_SAMPLE')
data, index = df.values[:, 1], pd.DatetimeIndex(df.values[:, 0])
origin_series = pd.Series(data.astype('float64'), index)

figsize = (20, 10)
origin_series.plot(marker=None, color='orange', legend=True, figsize=figsize)


# plt.xticks(pd.date_range('2019-09-17', '2019-09-20', periods=12))
# plt.show()
# pass


def ses():
    # Simple Exponential Smoothing
    ses_obj = SimpleExpSmoothing(origin_series)

    for color, smoothing_level in [('blue', 0.2), ('red', 0.6),
                                   ('green', None)]:
        if not smoothing_level:
            fit = ses_obj.fit(optimized=True)
            smoothing_level = fit.model.params['smoothing_level']
        else:
            fit = ses_obj.fit(smoothing_level=smoothing_level, optimized=False)
        forecast = fit.forecast(12).rename(rf'$\alpha={smoothing_level}$')

        # plot
        forecast.plot(marker='o', color=color, legend=True)
        # plt.show()

        fitted = pd.Series(data=fit.fittedvalues.values, index=index)
        fitted.plot(marker='o', color=color)

        # plt.show()

    plt.show()


def holt():
    forecast_steps = 100
    fit1 = Holt(origin_series).fit(
        smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
    forecast1 = fit1.forecast(forecast_steps).rename("Holt's linear trend")

    fit2 = Holt(origin_series, exponential=True).fit(
        smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
    forecast2 = fit2.forecast(forecast_steps).rename("Exponential trend")

    fit3 = Holt(origin_series, damped=True).fit(
        smoothing_level=0.8, smoothing_slope=0.2, damping_slope=0.8)
    forecast3 = fit3.forecast(forecast_steps).rename("Additive damped trend")

    fit1.fittedvalues.plot(marker="o", color='blue')
    forecast1.plot(color='blue', marker="o", legend=True)
    # plt.show()

    fit2.fittedvalues.plot(marker="o", color='red')
    forecast2.plot(color='red', marker="o", legend=True)
    # plt.show()

    fit3.fittedvalues.plot(marker="o", color='green')
    forecast3.plot(color='green', marker="o", legend=True)
    plt.show()


def holt_winters():
    forecast_steps = 1000
    fit1 = ExponentialSmoothing(origin_series, seasonal_periods=365,
                                trend='add',
                                seasonal='add').fit(use_boxcox=True)
    fit2 = ExponentialSmoothing(origin_series, seasonal_periods=365,
                                trend='add',
                                seasonal='mul').fit(use_boxcox=True)
    fit3 = ExponentialSmoothing(origin_series, seasonal_periods=365,
                                trend='add',
                                seasonal='add', damped=True).fit(
        use_boxcox=True, damping_slope=0.8)
    fit4 = ExponentialSmoothing(origin_series, seasonal_periods=365,
                                trend='add',
                                seasonal='mul', damped=True).fit(
        use_boxcox=True, damping_slope=0.8)

    fit1.fittedvalues.plot(style='--', color='red', figsize=figsize)
    fit3.fittedvalues.plot(style='--', color='green', figsize=figsize)

    fit1.forecast(forecast_steps).plot(style='--', marker=None, color='red',
                                       figsize=figsize,
                                       legend=True, linewidth=1)
    fit3.forecast(forecast_steps).plot(style='--', marker=None, color='green',
                                       legend=True, linewidth=1,
                                       figsize=figsize)

    plt.show()


if __name__ == "__main__":
    # ses()
    # holt()
    holt_winters()
    pass
