module Exploration

using DataFrames
using Statistics
using StatsBase
using HypothesisTests
using Missings
using PrettyTables

export describe_data, infer_column_types, missing_report, run_basic_tests

"""
    infer_column_types(df::DataFrame)::Dict{Symbol, String}

Infers the type of each column: :Continuous, :Categorical, :Datetime, :MissingOnly
"""
function infer_column_types(df::DataFrame)::Dict{Symbol, String}
    types = Dict{Symbol, String}()
    for col in names(df)
        col_data = df[!, col]
        if eltype(col_data) <: Union{Missing, Float64, Int}
            unique_vals = length(unique(skipmissing(col_data)))
            types[col] = unique_vals < 10 ? "Categorical" : "Continuous"
        elseif eltype(col_data) <: Union{Missing, AbstractString}
            types[col] = "Categorical"
        elseif eltype(col_data) <: Union{Missing, Date, DateTime}
            types[col] = "Datetime"
        elseif all(ismissing, col_data)
            types[col] = "MissingOnly"
        else
            types[col] = "Unknown"
        end
    end
    return types
end

"""
    describe_data(df::DataFrame)

Prints descriptive statistics and missing value summary.
"""
function describe_data(df::DataFrame)
    println("📊 Descriptive Statistics:")
    println("--------------------------------------------------")
    println(describe(df))
    println("--------------------------------------------------")
    println("\n🧩 Missing Values Report:")
    report = missing_report(df)
    pretty_table(report)
end

"""
    missing_report(df::DataFrame)::DataFrame

Returns a DataFrame with the count and percentage of missing values per column.
"""
function missing_report(df::DataFrame)::DataFrame
    total = nrow(df)
    colnames = names(df)
    miss_count = [count(ismissing, df[!, col]) for col in colnames]
    miss_percent = [round(100 * c / total, digits=2) for c in miss_count]

    return DataFrame(Column=colnames, MissingCount=miss_count, MissingPercent=miss_percent)
end

"""
    run_basic_tests(df::DataFrame)

Runs Shapiro-Wilk test for normality and prints summary.
"""
function run_basic_tests(df::DataFrame)
    println("\n🧪 Shapiro-Wilk Normality Test (numeric columns):")
    for col in names(df)
        col_data = skipmissing(df[!, col])
        if eltype(col_data) <: Real && length(col_data) >= 3
            try
                result = HypothesisTests.shapiro_wilk_test(collect(col_data))
                println(" - $(col): p = $(round(result.pvalue, digits=4)) → $(result.pvalue < 0.05 ? "NOT normal" : "normal")")
            catch
                println(" - $(col): Could not compute test (likely constant values)")
            end
        end
    end
end

end # module
