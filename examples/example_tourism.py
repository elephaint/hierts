"""
   Copyright (c) 2022- Olivier Sprangers as part of Airlab Amsterdam

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   https://github.com/elephaint/hierts/blob/main/LICENSE

"""
# This example is based on chapter 11 of:
#
# Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and practice, 
# 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3. Accessed on 05/07/2022.
#
# The data was taken from R's tsibble package:
# 
# Wang, E, D Cook, and RJ Hyndman (2020). A new tidy data structure to support exploration 
# and modeling of temporal data, Journal of Computational and Graphical Statistics, 29:3, 
# 466-478, doi:10.1080/10618600.2019.1695624.
# 
# https://tsibble.tidyverts.org/
#
# Download the .csv file at: https://github.com/elephaint/hierts/tree/main/examples/data
#
#%% Read packages
import pandas as pd
import numpy as np
from hierts.reconciliation import apply_reconciliation_methods, aggregate_bottom_up_forecasts, calc_level_method_rmse
from sktime.forecasting.ets import AutoETS
#%% Read data
df = pd.read_csv('data/tourism.csv', index_col=0)
df['Quarter'] = pd.PeriodIndex(df['Quarter'].str[0:4] + '-' + df['Quarter'].str[5:], freq='q')
#%% Set aggregations and calculate summing matrix
aggregation_cols = ['State', 'Region', 'Purpose']
# Define the aggregations. Don't include the top (total) level and bottom-level: these will be added automatically
aggregations = [['State'],
                ['State', 'Region'],
                ['State', 'Purpose'],
                ['Purpose']]
# Calculate summing matrix
df_S = calc_summing_matrix(df, aggregation_cols, aggregations)
#%% Create a forecasting model for each time series in the aggregation matrix df_S
# Set target, time_index and split of train and test.
target = 'Trips'
time_index = 'Quarter'
end_train = '2015Q4'
start_test = '2016Q1'
# Add bottom_timeseries identifier and create actuals dataframe for all aggregations
df['bottom_timeseries'] = df[aggregation_cols].agg('-'.join, axis=1)
actuals_bottom_timeseries = df.set_index(['bottom_timeseries', time_index])[target]\
                              .unstack(1)\
                              .loc[df_S.columns]
actuals = df_S @ actuals_bottom_timeseries
# Create forecasts: a simple ETS model per timeseries
forecasts = pd.DataFrame(index=actuals.index, columns = actuals.columns, dtype=np.float32)
for index, series in actuals.iterrows():
    # Fit model and predict (we need to clip because otherwise there's a convergence error)
    model = AutoETS(auto=True, n_jobs=1, random_state=0)
    model.fit(np.clip(series.loc[:end_train], 1e-3, 1e16))
    forecast = model.predict(series.index)
    # Store to forecasts/actuals array
    forecasts.loc[index] = forecast.values

# Calculate residuals (both in- and out-of-sample residuals)
residuals = (forecasts - actuals)
#%% Reconciliation
# Use residuals on training set 
residuals_train = residuals.loc[:, :end_train]
forecasts_test = forecasts.loc[:, start_test:]
forecasts_methods = apply_reconciliation_methods(forecasts_test, df_S, residuals_train, methods=['ols', 'wls_struct', 'wls_var', 'mint_shrink'])
# Bottom-up forecasts
forecasts_bu_bottom_level = forecasts.loc['Bottom level']
forecasts_bu = aggregate_bottom_up_forecasts(forecasts_bu_bottom_level, df_S)
forecasts_bu_test = forecasts_bu.loc[:, start_test:]
forecasts_method = pd.concat({'bottom-up': forecasts_bu_test}, names=['Method'])
forecasts_methods = pd.concat((forecasts_method, forecasts_methods), axis=0)
# Calculate error for all levels and methods. We set bottom-up as the base method to compare against
# in the relative rmse
rmse, rel_rmse = calc_level_method_rmse(forecasts_methods, actuals, base='bottom-up')