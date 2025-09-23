# [Imputer Module](@id imputer-guide)

`SynthPred.Imputer` provides both simple, column-wise strategies and more
advanced time-series-aware techniques to fill missing values while keeping
track of what was modified.

## Overview

- [`Imputer.impute_simple`](@ref) supports quick replacements using
  statistics (`"mean"`, `"median"`, `"mode"`) and directional fill
  (`"ffill"`, `"bfill"`).
- [`Imputer.impute_advanced`](@ref) targets numeric series with a high
  fraction of missing values, offering distribution sampling, ARIMA-based
  forecasting, or RNN-driven predictions.
- [`Imputer.save_imputation_report`](@ref) persists the summary table
  returned by `impute_advanced` as CSV or JSON for downstream auditing.

## Choosing a strategy

1. Start with [`Exploration.missing_report`](@ref) to understand missingness.
2. Apply `impute_simple` when each column can be treated independently or
   when the data lacks a clear temporal axis.
3. Switch to `impute_advanced` once you have a time stamp column and at
   least a moderate level of missingness (controlled by the `threshold`
   keyword).
4. Persist the generated report for reproducibility with
   `save_imputation_report`.

## Usage example

```julia
using DataFrames, Dates
using SynthPred

# Synthetic time series with gaps
traffic = DataFrame(
    ts = Date(2024, 1, 1):Day(1):Date(2024, 1, 10),
    visits = [120, 115, missing, 130, missing, 142, 138, missing, 150, 154],
)

# Fill sparse gaps via forward-fill for a quick preview
filled = Imputer.impute_simple(traffic, "ffill")

# Re-run with an RNN strategy for more principled estimates
df_clean, report = Imputer.impute_advanced(traffic, "rnn"; threshold=0.1)

# Persist the audit trail for transparency
Imputer.save_imputation_report(report, "reports/traffic_imputation.json")
```

## Related documentation

- Review the [Exploration guide](@ref exploration-guide) for the
  pre-imputation diagnostics that pair with these tools.
- Consult the [API Reference](@ref api-reference) for the exact method
  signatures and keyword arguments exposed by `Imputer`.
