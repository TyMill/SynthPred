# SynthPred.jl

**SynthPred.jl** is a Julia package for synthetic data analysis, imputation, AutoML, and model blending.

## 🚀 Features
- Missing data imputation: simple (mean, median), ARIMA, RNN
- Statistical data profiling
- AutoML pipeline with model evaluation
- Ensemble learning (top 2 models)
- JSON/CSV reports

## 📦 Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/TyMill/SynthPred.jl")
```
## 🧪 Quick Example

```julia
using SynthPred
using CSV, DataFrames

# 1. Load training data
df = CSV.read("data/example.csv", DataFrame)

# 2. Explore data: summary stats, missing values
SynthPred.Exploration.describe_data(df)

# 3. Impute missing values (advanced RNN strategy)
df_clean, report = SynthPred.Imputer.impute_advanced(df, "rnn", threshold=0.1)

# 4. Save imputation report
mkpath("reports")  # ensure directory exists before saving
SynthPred.Imputer.save_imputation_report(report, "reports/imputation_report.json")

# 5. Run AutoML to find top 2 models
top_models, scores = SynthPred.AutoML.run_automl(df_clean, :target)

# 6. Train ensemble model from top 2
X = select(df_clean, Not(:target))
y = df_clean[:, :target]
ensemble_model = SynthPred.AutoML.blend_top_models(top_models, X, y)

# 7. Predict on new data
Xnew = CSV.read("data/new_data.csv", DataFrame)
predictions = SynthPred.AutoML.predict_ensemble(ensemble_model, Xnew)

println(predictions)
```

## 📚 Documentation roadmap

- Dive into the [Exploration module guide](@ref exploration-guide) for
  profiling workflows and data diagnostics.
- Learn how to handle missing values in the [Imputer module guide](@ref
  imputer-guide).
- Automate model selection and blending with the [AutoML module guide](@ref
  automl-guide).
- Browse every exported symbol in the [API Reference](@ref api-reference).
