# [AutoML Module](@id automl-guide)

`SynthPred.AutoML` orchestrates model training, evaluation, and ensembling
using built-in implementations of Gaussian Naive Bayes and k-nearest
neighbour classifiers. It aims to provide a quick baseline as well as a path
to stronger ensemble predictors.

## Core capabilities

- [`AutoML.run_automl`](@ref) evaluates a configurable list of classifiers
  using cross-validation and returns the top performers.
- [`AutoML.blend_top_models`](@ref) builds a majority-vote ensemble when at
  least two trained models are available, or fits the single model directly
  otherwise.
- [`AutoML.predict_ensemble`](@ref) makes predictions on new data frames using
  the fitted ensemble.

## Recommended workflow

1. Prepare clean, imputed data (`DataFrame`) and identify the target column.
2. Call `run_automl` to rank candidate learners. You can limit the search
   with the `max_models` keyword if you only want to evaluate a subset of
   models.
3. Pass the selected models, along with the features and target, to
   `blend_top_models` to fit an ensemble container.
4. Use the returned ensemble with `predict_ensemble` for inference on
   hold-out or production datasets.

## Usage example

```julia
using DataFrames, Random
using SynthPred

# Create a simple classification dataset
Random.seed!(42)
df_clean = DataFrame(
    feature1 = rand(100),
    feature2 = rand(["A", "B"], 100),
    target = rand(["yes", "no"], 100),
)

# Split into training/testing partitions
train_idx = 1:80
test_idx = 81:100
train_df = df_clean[train_idx, :]
test_df = df_clean[test_idx, :]

# Run the AutoML search
top_models, scores = AutoML.run_automl(train_df, :target)

# Train a blended model on the training data
X = select(train_df, Not(:target))
y = train_df[:, :target]
mach = AutoML.blend_top_models(top_models, X, y)

# Generate predictions for unseen rows
preds = AutoML.predict_ensemble(mach, select(test_df, Not(:target)))
```

## Further reading

- Pair this with the [Imputer guide](@ref imputer-guide) to ensure the
  dataset passed into `AutoML` has reliable, gap-free inputs.
- Dive into the [API Reference](@ref api-reference) for argument details
  and notes on the lightweight classifiers powering the search.
