# [Exploration Module](@id exploration-guide)

The `SynthPred.Exploration` module contains utilities for understanding a
dataset before modelling. It focuses on column typing, descriptive
statistics, and sanity checks so that later imputation or modelling steps
can be chosen appropriately.

## Typical workflow

1. **Infer column semantics** with [`Exploration.infer_column_types`](@ref)
   to see how SynthPred categorises each feature (continuous, categorical,
   datetime, missing-only, or unknown).
2. **Summarise the dataset** via [`Exploration.describe_data`](@ref), which
   prints descriptive statistics alongside a missing-value table.
3. **Inspect missingness** directly using [`Exploration.missing_report`](@ref)
   when you want a tidy `DataFrame` for further processing or reporting.
4. **Run quick distribution checks** with
   [`Exploration.run_basic_tests`](@ref) to identify columns that deviate
   from normality using the Shapiro-Wilk test.

These steps help you decide which imputation or modelling strategy is the
best fit for your data.

## Usage example

```julia
using DataFrames
using Dates
using SynthPred

# Load or construct your dataset
customers = DataFrame(
    id = 1:6,
    joined_at = Date(2024, 1, 1):Day(15):Date(2024, 3, 11),
    spend = [12.5, 15.9, missing, 22.1, 18.4, missing],
    segment = ["A", "A", "B", "B", "B", missing],
)

# Inspect structure and missingness
types = Exploration.infer_column_types(customers)
Exploration.describe_data(customers)

# Work with the missing-value report programmatically
report = Exploration.missing_report(customers)
first(report, 5)

# Spot columns with non-normal behaviour before modelling
Exploration.run_basic_tests(customers)
```

## Next steps

- Continue to the [Imputation guide](@ref imputer-guide) when you are ready
  to fix missing values based on the insights gathered here.
- Reference the [API Reference](@ref api-reference) for the full set of
  docstrings and method signatures in `Exploration`.
