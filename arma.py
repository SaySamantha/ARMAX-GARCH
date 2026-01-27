import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from armadata import ARMAVolumeDataset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from arch import arch_model
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#TO CHECK USING VIF LATER
sing_exog_vars = ['log_return', 'relative_open', 'relative_high', 'relative_low', 'relative_close']

#TO TEST EXOG COMBINATIONS AFTER VIF
exog_vars = [
    'log_return', 'relative_open', 'relative_close',
    ['log_return', 'relative_open'],
    ['log_return', 'relative_close'],
    ['relative_open', 'relative_close'],
    ['log_return', 'relative_open', 'relative_close']
]

train_dataset = ARMAVolumeDataset(dt.datetime(2022, 9, 30), dt.datetime(2023, 5, 8))
val_dataset = ARMAVolumeDataset(dt.datetime(2023, 5, 9), dt.datetime(2023, 7, 19))
test_dataset = ARMAVolumeDataset(dt.datetime(2023, 7, 20), dt.datetime(2023, 9, 30))

train_data = train_dataset.process_data()
val_data = val_dataset.process_data()
test_data = test_dataset.process_data()

train_series = pd.Series(train_data['log_diff_volume'].dropna().values)
val_series = pd.Series(val_data['log_diff_volume'].dropna().values)
test_series = pd.Series(test_data['log_diff_volume'].dropna().values)

train_diff = train_series.values
val_diff = val_series.values
test_diff = test_series.values

#STATIONARY, AUTOCORR TESTS
adf_result = adfuller(train_diff)
print("\n[ADF Test - train_diff]")
print(f"ADF Statistic: {adf_result[0]:.6f}")
print(f"p-value: {adf_result[1]:.6f}")

lb_test = acorr_ljungbox(train_diff, lags=[20], return_df=True)
p_value = lb_test['lb_pvalue'].iloc[0]
print(f"Ljung-Box p-value at lag 20: {p_value:.6f}")

#ACF AND PACF GRAPHS
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_acf(train_series, lags=15, zero=False, ax=plt.gca())
plt.title("ACF - Train Log Diff Volume")
plt.subplot(1, 2, 2)
plot_pacf(train_series, lags=15, zero=False, ax=plt.gca())
plt.title("PACF - Train Log Diff Volume")
plt.tight_layout()
plt.show()

#CHOOSE THE BEST ARIMAX MODEL
orders = {
    "MA(1)": (0, 0, 1),
    "AR(1)": (1, 0, 0),
    "MA(2)": (0, 0, 2),
    "ARMA(1,1)": (1, 0, 1),
    "ARMA(1,2)": (1, 0, 2),
    "ARIMA(1, 1, 1)": (1, 1, 1),
    "ARIMA(1, 1, 2)": (1, 1, 2),
}

results = []
best_rmse = np.inf
best_model = None

for name, order in orders.items():
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    aic_val = model_fit.aic
    val_series_aligned = pd.Series(val_series.values, index=pd.RangeIndex(start=len(train_series), stop=len(train_series) + len(val_series)))
    updated_model = model_fit.append(val_series_aligned)
    forecast_series = updated_model.predict(start=len(train_series), end=len(train_series) + len(val_series) - 1, dynamic=False)
    forecast = np.nan_to_num(forecast_series.values, nan=0.0)
    rmse = np.sqrt(mean_squared_error(val_diff, forecast))
    results.append({'model': name, 'order': order, 'AIC': aic_val, 'RMSE': rmse})
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = name

results_df = pd.DataFrame(results).sort_values('RMSE')
print("\n[MODEL COMPARISON SUMMARY]")
print(results_df.to_string(index=False))
print(f"\n[BEST MODEL BY RMSE]")
print(f"{best_model} with RMSE = {best_rmse:.6f}")

#CHECK VIF OF EXOG
vif_data = train_data[sing_exog_vars].dropna()
X_vif = sm.add_constant(vif_data)
vif_results = pd.DataFrame()
vif_results["Variable"] = X_vif.columns
vif_results["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print("\n[VIF CHECK]")
print(vif_results.to_string(index=False))

#CHECK RMSE AND AIC OF EXOG COMBINATIONS ON VAL
results_exog = []
best_rmse_exog = np.inf
best_var = "log_return"

for var in exog_vars:
    cols = var if isinstance(var, list) else [var]
    name = "+".join(cols)

    exog_train = train_data[cols].iloc[:len(train_series)].values
    exog_val = val_data[cols].iloc[:len(val_series)].values

    model = ARIMA(train_series, order=(1,0,2), exog=exog_train)
    model_fit = model.fit()

    val_indices = pd.RangeIndex(start=len(train_series), stop=len(train_series) + len(val_series))
    val_series_clean = pd.Series(val_series.values, index=val_indices)
    exog_val_clean = pd.DataFrame(exog_val, index=val_indices)

    updated_model = model_fit.append(val_series_clean, exog=exog_val_clean)
    forecast_series = updated_model.predict(start=val_indices[0], end=val_indices[-1], exog=exog_val_clean)
    forecast = np.nan_to_num(forecast_series.values, nan=0.0)

    rmse = np.sqrt(mean_squared_error(val_diff, forecast))
    results_exog.append({'exog_set': name, 'AIC': model_fit.aic, 'RMSE': rmse})

    if rmse < best_rmse_exog:
        best_rmse_exog = rmse
        best_var = name

exog_results_df = pd.DataFrame(results_exog).sort_values('RMSE')
print("\n[EXOGENOUS SELECTION SUMMARY]")
print(exog_results_df.to_string(index=False))

best_exog_cols = best_var.split('+')
exog_train_best = train_data[best_exog_cols].iloc[:len(train_series)].values
final_arimax_for_diag = ARIMA(train_series, order=(1,0,2), exog=exog_train_best).fit()
train_residuals = final_arimax_for_diag.resid

#CHECK FOR GARCH EFFECTS
arch_lm_result = het_arch(train_residuals)
print("\n[ARCH-LM Test - train_residuals]")
print(f"ARCH-LM Statistic: {arch_lm_result[0]:.6f}")
print(f"p-value: {arch_lm_result[1]:.6f}")

#CHECK CANDIDATE LAGS FOR GARCH
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_acf(np.square(train_residuals), lags=15, zero=False, ax=plt.gca())
plt.title("ACF of Squared Train Residuals")
plt.subplot(1, 2, 2)
plot_pacf(np.square(train_residuals), lags=15, zero=False, ax=plt.gca())
plt.title("PACF of Squared Train Residuals")
plt.tight_layout()
plt.show()

vol_models = ["GARCH", "EGARCH", "GJR-GARCH"]
vol_orders = [(1,1), (2,1)]
distributions = ["normal", "t", "ged"]

exog_val_best = val_data[best_exog_cols].iloc[:len(val_series)].values
val_forecast_mean = final_arimax_for_diag.forecast(steps=len(val_series), exog=exog_val_best)
val_resids = val_diff - val_forecast_mean

#FIND THE BEST GARCH MODEL
arch_results = []
best_rmse_vol = np.inf
best_params_vol = None

scaling_factor = 100.0 if np.abs(train_residuals).mean() < 1e-3 else 1.0
scaled_train_residuals = train_residuals * scaling_factor

for vol in vol_models:
    for p, q in vol_orders:
        for dist in distributions:
            try:
                am = arch_model(scaled_train_residuals, vol=vol, p=p, q=q, dist=dist, mean="Zero")
                res = am.fit(disp="off", show_warning=False)
                
                vol_preds = res.conditional_volatility
                actual_vol = np.abs(scaled_train_residuals) 
                v_rmse = np.sqrt(mean_squared_error(actual_vol, vol_preds))
                
                arch_results.append({
                    'model': f"{vol}({p},{q})", 
                    'dist': dist, 
                    'RMSE': v_rmse / scaling_factor
                })
                
                if v_rmse < best_rmse_vol:
                    best_rmse_vol = v_rmse
                    best_params_vol = {'type': vol, 'order': (p,q), 'dist': dist}
            except:
                continue

arch_summary_df = pd.DataFrame(arch_results).sort_values('RMSE')
print("\n[VOLATILITY MODEL SUMMARY (BY RMSE)]")
print(arch_summary_df.head(10).to_string(index=False))

#FORECAST THE MEAN
final_var = best_var.split('+')
exog_train = train_data[final_var].iloc[:len(train_series)].values
exog_test = test_data[final_var].iloc[:len(test_series)].values

arimax_fit = ARIMA(train_series, order=(1,0,2), exog=exog_train).fit()

test_dates = test_data.index[-len(test_diff):]
test_indices = pd.RangeIndex(start=len(train_series), stop=len(train_series) + len(test_diff))
exog_test_df = pd.DataFrame(exog_test, index=test_indices)

updated_arimax = arimax_fit.append(pd.Series(test_diff, index=test_indices), exog=exog_test_df)
mean_forecast = updated_arimax.predict(start=test_indices[0], end=test_indices[-1], exog=exog_test_df).values

#FIT THE GARCH MODEL
final_garch_model = arch_model(
    scaled_train_residuals, 
    vol=best_params_vol['type'], 
    p=best_params_vol['order'][0], 
    q=best_params_vol['order'][1], 
    dist=best_params_vol['dist'], 
    mean="Zero"
)
garch_fitted = final_garch_model.fit(disp="off")

#GENERATE USING SIMULATION
forecast_horizon = len(test_diff)
garch_forecast = garch_fitted.forecast(
    horizon=forecast_horizon,
    method='simulation',
    simulations=100
)

forecasted_variance = garch_forecast.variance.values[-1]
forecasted_stddev = np.sqrt(forecasted_variance) / scaling_factor

rs = np.random.RandomState(42)
nu = garch_fitted.params.get('nu', 4.0)
dist = best_params_vol['dist']

if dist == "normal":
    simulated_z = rs.normal(0, 1, size=forecast_horizon)
elif dist == "t":
    simulated_z = rs.standard_t(df=nu, size=forecast_horizon)
elif dist == "ged":
    from scipy.stats import gennorm
    simulated_z = gennorm.rvs(beta=nu, size=forecast_horizon, random_state=rs)

predicted_et = forecasted_stddev * simulated_z
combined_predictions = mean_forecast + predicted_et

final_rmse = np.sqrt(mean_squared_error(test_diff, combined_predictions))
mean_only_rmse = np.sqrt(mean_squared_error(test_diff, mean_forecast))

#PLOT THE RESULTS
fig = make_subplots(
    rows=2, cols=1, 
    subplot_titles=(
        f"ARIMAX Point Forecast (RMSE: {mean_only_rmse:.6f})", 
        f"Combined {best_params_vol['type']} Simulation (RMSE: {final_rmse:.6f})"
    ),
    vertical_spacing=0.12
)

fig.add_trace(
    go.Scatter(x=test_dates, y=test_diff, name="Actual", 
               line=dict(color='black', width=1), opacity=0.8),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=test_dates, y=mean_forecast, name="Prediction", 
               line=dict(color='blue', width=1.5)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=test_dates, y=test_diff, name="Actual", 
               line=dict(color='black', width=1), opacity=0.8, showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=test_dates, y=combined_predictions, name="Prediction (Combined)", 
               line=dict(color='blue', width=1.5), showlegend=False),
    row=2, col=1
)

fig.update_layout(
    template="plotly_white",
    height=900,
    showlegend=True,
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="right",
        x=1
    ),
    margin=dict(l=50, r=50, t=80, b=50)
)

fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f2f6', tickformat='%b %d %Y')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f2f6', zeroline=True, zerolinecolor='lightgrey')
fig.show()

#PRINT RESULTS
print(f"\n[FINAL PERFORMANCE COMPARISON]")
print(f"Mean-Only RMSE (Point Accuracy):     {mean_only_rmse:.6f}")
print(f"Combined RMSE (Stochastic Realism):  {final_rmse:.6f}")
