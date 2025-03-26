using CSV, DataFrames
using SynthPred.Exploration
using SynthPred.Imputer
using SynthPred.AutoML

const DATA_PATH = "data/example.csv"
const NEW_DATA_PATH = "data/new_data.csv"
const REPORT_PATH = "reports/imputation_report.json"

# ----------------------------------------
println("📥 Wczytywanie danych z pliku: $DATA_PATH")
df = CSV.read(DATA_PATH, DataFrame)

# ----------------------------------------
println("📊 ANALIZA DANYCH")
describe_data(df)
run_basic_tests(df)

# ----------------------------------------
println("🧼 IMPUTACJA DANYCH (RNN/ARIMA/distribution)")
df_filled, report = impute_advanced(df, "rnn", threshold=0.1)
save_imputation_report(report, REPORT_PATH)

# ----------------------------------------
target_col = :target
println("🧠 AutoML – wybór top 2 modeli dla kolumny $target_col")
top_models, scores = run_automl(df_filled, target_col)

X = select(df_filled, Not(target_col))
y = df_filled[:, target_col]

println("⚖️ Tworzenie modelu ensemble z top 2 modeli")
ensemble_model = blend_top_models(top_models, X, y)

# ----------------------------------------
println("📈 Predykcja na nowych danych z pliku: $NEW_DATA_PATH")
Xnew = CSV.read(NEW_DATA_PATH, DataFrame)
predictions = predict_ensemble(ensemble_model, Xnew)

println("📢 Wyniki predykcji:")
println(predictions)
