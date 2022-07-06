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
# The data comes from the online book of Hyndman, see link below.
#
#%% Read packages
import pandas as pd
import numpy as np
from hierts.reconciliation import calc_summing_matrix, apply_reconciliation_methods, calc_level_method_rmse
#%% Read data
df = pd.read_csv("https://OTexts.com/fpp3/extrafiles/prison_population.csv")
#%% Set aggregations and calculate summing matrix
aggregation_cols = ['State', 'Gender', 'Legal', 'Indigenous']
# Define the aggregations. Don't include the top (total) level and bottom-level: these will be added automatically
aggregations = [['State'],
                ['State', 'Gender'],
                ['State', 'Legal'],
                ['State', 'Indigenous'],
                ['Gender', 'Legal']]
# Calculate summing matrix
df_S = calc_summing_matrix(df, aggregation_cols, aggregations)
#%% We randomly generate forecasts using a poisson sampling on the actual values
target = 'Count'
time_index = 'Date'
end_train = '2015-12-31'
start_test = '2016-01-01'
rng = np.random.default_rng(seed=0)
df[f'{target}_predicted'] = rng.poisson(lam=df[f'{target}'])
#%% Create actuals and forecast dataframes for all aggregations
df['bottom_timeseries'] = df[aggregation_cols].agg('-'.join, axis=1)
actuals_bottom_timeseries = df.set_index(['bottom_timeseries', time_index])[target]\
                              .unstack(1)\
                              .loc[df_S.columns]
forecasts_bottom_timeseries = df.set_index(['bottom_timeseries', time_index])[f'{target}_predicted']\
                                .unstack(1)\
                                .loc[df_S.columns]
actuals = df_S @ actuals_bottom_timeseries
forecasts = df_S @ forecasts_bottom_timeseries
residuals = (forecasts - actuals)
#%% Reconciliation
forecasts_test = forecasts.loc[:, start_test:]
residuals_train = residuals.loc[:, :end_train]
forecasts_reconciled = apply_reconciliation_methods(forecasts_test, df_S, residuals_train, methods=['ols', 'wls_var', 'mint_shrink'])
rmse, rel_rmse = calc_level_method_rmse(forecasts_reconciled, actuals, base='base')