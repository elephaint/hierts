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
#%% Read packages
import pandas as pd
import numpy as np
from src import calc_summing_matrix, apply_reconciliation_methods, aggregate_bottom_up_forecasts, calc_level_method_rmse
from sktime.forecasting.ets import AutoETS
#%% Read data
df = pd.read_csv('examples/data/tourism.csv', index_col=0)
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
df['bottom_timeseries'] = df[aggregation_cols].agg('-'.join, axis=1)
df_target = df.set_index(['bottom_timeseries', time_index])[target].unstack(0)
df_target = df_target[df_S.columns]
forecasts = pd.DataFrame(index=df_S.index, columns = df_target.index, dtype=np.float32)
actuals = pd.DataFrame(index=df_S.index, columns = df_target.index, dtype=np.float32)
for aggregate, summing_vector in df_S.iterrows():
    # Get series
    series = df_target @ summing_vector
    # Fit model and predict (we need to clip because otherwise there's a convergence error)
    model = AutoETS(auto=True, n_jobs=1, random_state=0)
    model.fit(np.clip(series.loc[:end_train], 1e-3, 1e16))
    forecast = model.predict(series.index)
    # Store to forecasts/actuals array
    forecasts.loc[aggregate] = forecast.values
    actuals.loc[aggregate] = series.values

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