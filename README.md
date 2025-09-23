# SynthPred.jl
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://tymill.github.io/SynthPred/)

[![DOI](https://zenodo.org/badge/955290469.svg)](https://doi.org/10.5281/zenodo.15090892)

![GitHub all releases](https://img.shields.io/github/downloads/TyMill/SynthPred/total?label=📦%20Downloads&style=plastic)


**SynthPred.jl** is a Julia package for synthetic data analysis, advanced imputation (ARIMA, RNN), AutoML, and ensemble modeling.

---

## 🚀 Features

- 🔍 Descriptive statistics and missing data reporting
- 🧼 Simple and advanced imputation:
  - Mean, median, mode
  - Forward/backward fill
  - Gaussian distribution sampling
  - Time series-based: ARIMA
  - Sequence learning-based: RNN (Flux.jl)
- 🤖 AutoML for classification (MLJ.jl-based)
- ⚖️ Blending top-performing models via ensembling
- 📊 Predictions on new data
- 📑 JSON/CSV imputation reports

---

## 📦 Installation

```julia
using Pkg
Pkg.add(url="https://github.com/TyMill/SynthPred.jl")
```

---

## 🧪 Quick Example

```julia
using SynthPred
using CSV, DataFrames

# Load training data
df = CSV.read("data/example.csv", DataFrame)

# Explore data
SynthPred.Exploration.describe_data(df)

# Impute missing values (e.g. RNN strategy)
df_clean, report = SynthPred.Imputer.impute_advanced(df, "rnn", threshold=0.1)
mkpath("reports")  # ensure directory exists before saving
SynthPred.Imputer.save_imputation_report(report, "reports/imputation_report.json")

# Run AutoML pipeline
top_models, scores = SynthPred.AutoML.run_automl(df_clean, :target)
X = select(df_clean, Not(:target))
y = df_clean[:, :target]
ensemble = SynthPred.AutoML.blend_top_models(top_models, X, y)

# Predict on new data
Xnew = CSV.read("data/new_data.csv", DataFrame)
preds = SynthPred.AutoML.predict_ensemble(ensemble, Xnew)
println(preds)
```

---

## 📚 Documentation

- 📘 Module walkthroughs: [Exploration](https://tymill.github.io/SynthPred.jl/exploration/), [Imputer](https://tymill.github.io/SynthPred.jl/imputer/), [AutoML](https://tymill.github.io/SynthPred.jl/automl/)
- 📖 Full API: [https://tymill.github.io/SynthPred.jl/api/](https://tymill.github.io/SynthPred.jl/api/)
- 🌐 Documentation hub: [https://tymill.github.io/SynthPred.jl](https://tymill.github.io/SynthPred.jl)

---

## 🧪 Project Structure

```
SynthPred/
├── Project.toml
├── src/
│   ├── SynthPred.jl
│   ├── Exploration.jl
│   ├── Imputer.jl
│   └── AutoML.jl
├── data/
│   ├── example.csv
│   └── new_data.csv
├── reports/
│   └── imputation_report.json
├── docs/
│   └── src/index.md
├── test/
│   └── runtests.jl
└── main.jl
```

---

## 📌 Roadmap

- [x] Core modules: Exploration, Imputer, AutoML
- [x] ARIMA and RNN-based imputations
- [x] AutoML + model blending with MLJ.jl
- [x] Imputation reports (CSV/JSON)
- [x] Documentation (Documenter.jl + GitHub Pages)
- [ ] Exporting trained models (`JLD2`, `BSON`)
- [ ] Web GUI with Pluto.jl or Dash.jl
- [ ] Integration with JuliaHub and Zenodo DOI

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss your proposal.

---

## 📜 License

MIT License © 2025 Tymoteusz Miller

---

## 📬 Contact

📧 me@tymoteuszmiller.dev


---

Built with ❤️ in Julia for real-world ML and scientific discovery.

