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
#%% Import packages
import numpy as np 
import pandas as pd
import time
from numba import njit, prange
from typing import List, Tuple
#%% Functions to perform forecast reconciliation
def reconcile_forecasts(yhat: np.ndarray, S: np.ndarray, residuals: np.ndarray=None, method: str='ols') -> np.ndarray:
    """Optimal reconciliation of hierarchical forecasts using various approaches.
    
        Based on approaches from:
    
        Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). 
        Optimal forecast reconciliation for hierarchical and grouped time series through 
        trace minimization. Journal of the American Statistical Association, 114(526), 804-819.

        :param yhat: forecasts for each time series for each timestep of size [n_timeseries x n_timesteps]
        :type yhat: numpy.ndarray
        :param S: summing matrix detailing the hierarchical tree of size [n_timeseries x n_bottom_timeseries]
        :type S: numpy.ndarray
        :param residuals: residuals (i.e. forecast errors) for each time series for a set of historical timesteps of 
        size [n_timeseries x n_timesteps_residuals]. Required when using 'wls_var', 'mint_cov', 'mint_shrink'
        :type residuals: numpy.ndarray, optional
        :param method: reconciliation method, defaults to 'ols'. Options are: 'ols', 'wls_var', 'wls_struct', 
        'mint_cov', 'mint_shrink'
        :type method: str, optional

        :return: ytilde, reconciled forecasts for each time series for each timestep of size [n_timeseries x n_timesteps]
        :rtype: numpy.ndarray

    """
    n_timeseries = S.shape[0]
    n_bottom_timeseries = S.shape[1]
    ms = n_timeseries - n_bottom_timeseries
    assert yhat.shape[0] == n_timeseries, "Forecasts and summing matrix S do not contain the same amount of time series"
    if method in ['wls_var', 'mint_cov', 'mint_shrink']:
        assert residuals is not None, f"Method {method} requires you to provide a residuals matrix"
        assert residuals.shape[0] == n_timeseries, "Residuals and summing matrix S should contain the same amount of time series"

    # Prepare arrays for reconciliation
    J = np.concatenate((np.zeros((n_bottom_timeseries, ms), dtype=np.float32), S[ms:]), axis=1)
    Ut = np.concatenate((np.eye(ms, dtype=np.float32), -S[:ms]), axis=1)
    # Select correct weight matrix W according to specified reconciliation method
    if method == 'ols':
        # Ordinary least squares, default option. W = np.eye(n_timeseries), thus UtW = Ut @ W = Ut * Wdiag = Ut
        UtW = Ut
    elif method == 'wls_struct':
        # Weighted least squares using structural scaling. W matrix has non-zero elements on diagonal only.
        Wdiag = np.sum(S, axis=1)
        UtW = Ut * Wdiag
    elif method == 'wls_var':
        # Weighted least squares using variance scaling. W matrix has non-zero elements on diagonal only.
        Wdiag = np.sum(residuals**2, axis=1) / residuals.shape[1]
        UtW = Ut * Wdiag
    elif method == 'mint_cov':
        # Trace minimization using the empirical error covariance matrix
        W = np.cov(residuals)
        UtW = Ut @ W
    elif method == 'mint_shrink':
        # Trace minimization using the shrunk empirical covariance matrix
        residuals_mean = np.mean(residuals, axis=1)
        residuals_std = np.std(residuals, axis=1)
        W = shrunk_covariance_schaferstrimmer(residuals, residuals_mean, residuals_std)
        UtW = Ut @ W
    else:
        raise NotImplementedError("Method not implemented. Options are: ['ols', 'wls_var', 'wls_struct', 'mint_cov', 'mint_shrink']")
        
    # Compute reconciled forecasts
    G = (J - np.linalg.solve(UtW[:, ms:] @ Ut.T[ms:] + UtW[:, :ms], UtW[:, ms:] @ J.T[ms:]).T @ Ut)
    ytilde = (yhat.T @ G.T @ S.T).T

    return ytilde

@njit(parallel=True, fastmath=True, cache=True)
def shrunk_covariance_schaferstrimmer(residuals, residuals_mean, residuals_std):
    # Shrink empirical covariance according to the following method:
    #   Schäfer, Juliane, and Korbinian Strimmer. 
    #   ‘A Shrinkage Approach to Large-Scale Covariance Matrix Estimation and 
    #   Implications for Functional Genomics’. Statistical Applications in 
    #   Genetics and Molecular Biology 4, no. 1 (14 January 2005). 
    #   https://doi.org/10.2202/1544-6115.1175.

    n_timeseries = residuals.shape[0]
    n_samples = residuals.shape[1]
    # We need the empirical covariance, the off-diagonal sum of the variance of 
    # the empirical correlation matrix and the off-diagonal sum of the squared 
    # empirical correlation matrix.
    emp_cov = np.zeros((n_timeseries, n_timeseries), dtype=np.float32)
    sum_var_emp_corr = 0
    sum_sq_emp_corr = -n_timeseries
    factor_emp_corr = n_samples / (n_samples - 1)
    factor_var_emp_cor = n_samples / (n_samples - 1)**3
    for i in range(n_timeseries):
        # Calculate standardized residuals
        X_i = (residuals[i] - residuals_mean[i]) 
        Xs_i = X_i / residuals_std[i]
        Xs_i_mean = np.mean(Xs_i)
        for j in prange(n_timeseries):
            # Calculate standardized residuals
            X_j = (residuals[j] - residuals_mean[j]) 
            Xs_j = X_j / residuals_std[j]
            Xs_j_mean = np.mean(Xs_j)
            # Empirical covariance
            emp_cov[i, j] = factor_emp_corr * np.mean(X_i * X_j)
            # Sum off-diagonal variance of empirical correlation
            w = (Xs_i - Xs_i_mean) * (Xs_j - Xs_j_mean)
            w_mean = np.mean(w)
            sum_var_emp_corr += (i != j) * factor_var_emp_cor * np.sum(np.square(w - w_mean))
            # Sum squared empirical correlation (off-diagonal correction made by initializing 
            # with -n_timeseries, so (i != j) not necessary here)
            sum_sq_emp_corr += np.square(factor_emp_corr * w_mean)

    # Calculate shrinkage intensity 
    shrinkage = sum_var_emp_corr / sum_sq_emp_corr
    # Calculate shrunk covariance estimate
    emp_cov_diag = np.diag(emp_cov)
    W = (1 - shrinkage) * emp_cov
    # Fill diagonal with original empirical covariance diagonal
    np.fill_diagonal(W, emp_cov_diag)

    return W

def calc_summing_matrix(df: pd.DataFrame, aggregation_cols: List[str], aggregations: List[List[str]] = None) -> pd.DataFrame:
    """Given a dataframe of timeseries and columns indicating their groupings, this function
    calculates a summing matrix according to a set of specified aggregations for the time series. 

        :param df: DataFrame containing information about time series and their groupings
        :type df: pd.DataFrame
        :param aggregation_cols: List containing all the columns that contain categorization of the timeseries. 
        :type df: List[str]
        :param aggregations: List of Lists containing the aggregations required, defaults to None. In case of None, 
        the summing matrix will only contain (i) the summation vector for the total series (i.e. a row vector of ones
         of length n_bottom_series), and (ii) the summation matrix for the bottom level series (i.e. the identity matrix 
         for the amount of bottom level time series). Hence, in the case of None, the output df_S will have
         shape [n_bottom_series + 1, n_bottom_series]
        :type df: List[List[str]]
        
        :return: df_S, output dataframe containing the summing matrix of shape [(n_bottom_timeseries + n_aggregate_timeseries) x n_bottom_timeseries]
        The number of aggregate time series is the result of applying all the required aggregations.
        :rtype: pd.DataFrame

        Example:
		
		.. code-block:: python
			
            df = pd.DataFrame([])
            aggregation_cols = ['Level1', 'Level2']
            aggregations = [['Level1'], ['Level1', 'Level2']]
            df_S = calc_summing_matrix(df, aggregations, aggregation_cols)
    
    """
    # Set aggregations to empty list in case it is not provided
    if aggregations is None:
        aggregations = []
    # Check whether aggregations are in the aggregation_cols
    aggregation_cols_in_aggregations = list(set([col for cols in aggregations for col in cols]))
    for col in aggregation_cols_in_aggregations:
        assert col in aggregation_cols, f"Column {col} in aggregations not present in aggregation_cols"
    # Find the unique aggregation columns from the given set of aggregations
    levels = df[aggregation_cols].drop_duplicates().reset_index(drop=True)
    # Create summing matrix for all aggregation levels
    df_S_aggs = pd.DataFrame()
    for aggregation in aggregations:
        aggregation_name = '-'.join(aggregation)
        S_agg = pd.get_dummies(levels[aggregation].astype(str).agg('-'.join, axis=1).sort_values().reset_index(drop=True)).T    
        S_agg = pd.concat({f'{aggregation_name}': S_agg}, names=['Aggregation', 'Value'])
        df_S_aggs = pd.concat((df_S_aggs, S_agg))
    
    # Create summing matrix for bottom level series
    bottom_timeseries = levels[aggregation_cols].astype(str).agg('-'.join, axis=1).sort_values().reset_index(drop=True)
    df_S_bottom = pd.get_dummies(bottom_timeseries).T 
    df_S_bottom = pd.concat({'Bottom level': df_S_bottom}, names=['Aggregation', 'Value'])
    # Create summing matrix (=row vector) for top level (=total) series
    n_bottom_timeseries = len(levels)
    S_top = np.ones((1, n_bottom_timeseries))
    df_S_top = pd.DataFrame(S_top, index=['Total'])
    df_S_top = pd.concat({'Total': df_S_top}, names=['Aggregation', 'Value'])
    # Stack all summing matrices: top, aggregations, bottom
    df_S = pd.concat((df_S_top, df_S_aggs, df_S_bottom))
    df_S.columns = bottom_timeseries

    return df_S

def apply_reconciliation_methods(forecasts: pd.DataFrame, df_S: pd.DataFrame, residuals: pd.DataFrame,
                                methods: List[str] = None) -> pd.DataFrame:
    """Apply all hierarchical forecasting reconciliation methods to a set of forecasts.

        :param forecasts: dataframe containing forecasts for all aggregations
        :type forecasts: pd.DataFrame
        :param df_S: Dataframe containing the summing matrix for all aggregations in the hierarchy.
        :type df_S: pd.DataFrame
        :param residuals: dataframe containing the residuals on the training set for all timeseries.
        :type residuals: pd.DataFrame
        :param methods: list containing which reconciliation methods to be applied, defaults to None. 
        Choose from: 'ols', 'wls_var', 'wls_struct', 'mint_cov', 'mint_shrink'. None means all methods 
        will be applied.
        :type methods: List[str]
        
        :return: forecasts_methods, dataframe containing forecasts for all reconciliation methods
        :rtype: pd.DataFrame  
    
    """
    forecasts_method = pd.concat({'base': forecasts}, names=['Method'])
    cols = forecasts_method.columns
    # Convert to float32
    yhat = forecasts_method.values.astype('float32')
    S = df_S.values.astype('float32')
    residuals = residuals.values.astype('float32')
    # Apply all reconciliation methods
    if methods == None:
        methods = ['ols', 'wls_struct', 'wls_var', 'mint_cov', 'mint_shrink']
    forecasts_methods = []
    forecasts_methods.append(forecasts_method)
    for method in methods:
        t0 = time.perf_counter()
        ytilde = reconcile_forecasts(yhat, S, residuals, method=method)
        t1 = time.perf_counter()
        print(f'Method {method}, reconciliation time: {t1-t0:.4f}s')
        forecasts_method = pd.DataFrame(data=ytilde,
                                        index=forecasts.index, 
                                        columns=cols)
        forecasts_method = pd.concat({f'{method}': forecasts_method}, names=['Method'])
        forecasts_methods.append(forecasts_method)

    forecasts_methods = pd.concat(forecasts_methods)
    
    return forecasts_methods

def aggregate_bottom_up_forecasts(forecasts: pd.DataFrame, df_S: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a set of bottom-level forecasts according to a specified summing matrix df_S

        :param forecasts: dataframe containing bottom-level forecasts
        :type forecasts: pd.DataFrame
        :param df_S: Dataframe containing the summing matrix for all aggregations in the hierarchy.
        :type df_S: pd.DataFrame
        
        :return: forecasts_methods, dataframe containing forecasts for all reconciliation methods
        :rtype: pd.DataFrame      
    
    """
    # Check to ensure correct inputs are given
    assert set(df_S.columns) == set(forecasts.index), 'Index of forecasts should match columns of df_S'
    # Bottom-up forecasts
    all_aggregations = df_S.index.get_level_values('Aggregation').unique()
    all_aggregations = all_aggregations.drop('Bottom level')
    forecasts_bu = pd.DataFrame(index=df_S.index, columns=forecasts.columns)
    forecasts_bu.loc['Bottom level'] = forecasts.values.astype('float32')
    for agg in all_aggregations:
        forecasts_bu.loc[agg] = (df_S.loc[agg] @ forecasts).values.astype('float32')

    return forecasts_bu


def calc_level_method_rmse(forecasts_methods: pd.DataFrame, actuals: pd.DataFrame, 
                            base: str = 'base') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate RMSE for each level, for each method for a set of forecasts.
    
        :param forecasts_methods: dataframe containing forecasts for all reconciliation methods
        :type forecasts_methods: pd.DataFrame
        :param actuals: Dataframe containing the ground truth for all time series
        :type actuals: pd.DataFrame
        :param base: base to compare rmse against for the `rel_rmse` output. 
        :type base: str
        
        :return: Tuple[rmse, rel_rmse], tuple containing (i) rmse for all methods, across all levels, and (ii)
        relative rmse for all methods, across all levels. 
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    
    """
    # Input verification
    assert base in forecasts_methods.index, f'Chosen base {base} not in index of forecasts_methods'
    
    # Compute rmse for all methods & levels
    rmse_index = forecasts_methods.index.droplevel(['Value']).drop_duplicates()
    rmse = pd.DataFrame(index=rmse_index, columns=['RMSE'], dtype=np.float64)
    # rmse = pd.DataFrame()
    methods = forecasts_methods.index.get_level_values('Method').unique()
    for method in methods:
        forecasts_method = forecasts_methods.loc[method]
        sq_error = ((forecasts_method - actuals.loc[:, forecasts_method.columns])**2).stack()
        rmse_current = np.sqrt(sq_error.groupby(['Aggregation']).mean())
        rmse.loc[(method, slice(None)), 'RMSE'] = rmse_current.loc[rmse.loc[method, 'RMSE'].index].values
        rmse.loc[(method, 'All series'), 'RMSE'] = np.sqrt(sq_error.mean())
        
    rmse = rmse.sort_index().unstack(0)
    # Sort by base, then put in order: Total first, bottom-level series (i.e. None) penultimate, aggregate
    # over all time series + aggregates ('All series') last
    rmse.columns = rmse.columns.droplevel(0)
    rmse = rmse[methods].sort_values(by=base, ascending=False)
    index_cols = list(rmse.index.drop('All series')) + ['All series']
    rmse = rmse.reindex(index = index_cols)
    rel_rmse = rmse.div(rmse[base], axis=0)

    return rmse, rel_rmse