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

#FIND BEST GARCH MODEL
arch_results = []
best_aic_vol = np.inf
best_params_vol = None

scaling_factor = 100.0 if np.abs(train_residuals).mean() < 1e-3 else 1.0
scaled_train_residuals = train_residuals * scaling_factor

arch_results = []
best_aic_vol = np.inf
best_params_vol = None

for vol in vol_models:
    for p, q in vol_orders:
        for dist in distributions:
            try:
                am = arch_model(scaled_train_residuals, vol=vol, p=p, q=q, dist=dist, mean="Zero")
                res = am.fit(disp="off", show_warning=False)
                current_aic = res.aic
                arch_results.append({'model': f"{vol}({p},{q})", 'dist': dist, 'AIC': current_aic, 'Log-Likelihood': res.loglikelihood})
                if current_aic < best_aic_vol:
                    best_aic_vol = current_aic
                    best_params_vol = {'type': vol, 'order': (p,q), 'dist': dist, 'scaling': scaling_factor}
            except:
                continue

arch_summary_df = pd.DataFrame(arch_results).sort_values('AIC')

print("\n[VOLATILITY MODEL SUMMARY]")
print(arch_summary_df.head(10).to_string(index=False))
if best_params_vol:
    print(f"\nBEST: {best_params_vol['type']}({best_params_vol['order']}) with {best_params_vol['dist']} distribution.")
else:
    print("\nERROR: No models converged.")

# 1. Final Mean Model (ARIMAX)
final_var = best_var.split('+')
exog_train = train_data[final_var].iloc[:len(train_series)].values
exog_test = test_data[final_var].iloc[:len(test_series)].values

arimax_fit = ARIMA(train_series, order=(1,0,2), exog=exog_train).fit()

# 2. Best GARCH Model (For parameter identification only)
train_resids = arimax_fit.resid
scaling_factor = 100.0 if np.abs(train_resids).mean() < 1e-3 else 1.0
scaled_train_resids = train_resids * scaling_factor

final_garch_model = arch_model(
    scaled_train_resids, 
    vol=best_params_vol['type'], 
    p=best_params_vol['order'][0], 
    q=best_params_vol['order'][1], 
    dist=best_params_vol['dist'], 
    mean="Zero"
)
garch_fitted = final_garch_model.fit(disp="off")

test_dates = test_data.index[-len(test_diff):]
test_indices = pd.RangeIndex(start=len(train_series), stop=len(train_series)+len(test_diff))
exog_test_df = pd.DataFrame(exog_test, index=test_indices)

updated_arimax = arimax_fit.append(pd.Series(test_diff, index=test_indices), exog=exog_test_df)
mean_forecast_test = updated_arimax.predict(start=test_indices[0], end=test_indices[-1], exog=exog_test_df)

final_preds = mean_forecast_test.values
final_rmse = np.sqrt(mean_squared_error(test_diff, final_preds))

#PLOT THE FORECASTS
plt.figure(figsize=(12, 6))
plt.plot(test_dates, test_diff, label='Actual Volume (Log Diff)', color='black', alpha=0.3, linewidth=1)
plt.plot(test_dates, final_preds, label='Predicted (ARIMAX Mean)', color='crimson', linewidth=1.5)
plt.title("Test Forecast: ARMAX Model")
plt.xlabel("Date")
plt.ylabel("Log Difference of Volume")
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.4)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#PRINT RESULTS
print("\n[FINAL MODEL PERFORMANCE]")
print(f"ARMAX Model: ARIMAX(1,0,2) with {best_var}")
print(f"GARCH Model: {best_params_vol['type']}({best_params_vol['order'][0]},{best_params_vol['order'][1]})")
print(f"Distribution: {best_params_vol['dist']}")
print(f"Final Test RMSE: {final_rmse:.6f}")
