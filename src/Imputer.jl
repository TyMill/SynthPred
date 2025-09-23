module Imputer

using DataFrames, Statistics, StatsBase, Distributions
using Impute, Dates, JSON, CSV
using Flux
using StateSpaceModels

export impute_simple, impute_advanced, save_imputation_report

# ----------------------------
# Simple Imputation
# ----------------------------
"""
    impute_simple(df::DataFrame, method::String)::DataFrame

Simple methods: "mean", "median", "mode", "ffill", "bfill"
"""
function impute_simple(df::DataFrame, method::String)::DataFrame
    df_clean = deepcopy(df)

    for col in names(df)
        col_data = df[!, col]

        if eltype(col_data) <: Union{Missing, Number}
            if method == "mean"
                df_clean[!, col] = Impute.impute(col_data, Impute.mean)
            elseif method == "median"
                df_clean[!, col] = Impute.impute(col_data, Impute.median)
            elseif method == "mode"
                df_clean[!, col] = Impute.impute(col_data, Impute.mode)
            end
        elseif method == "mode" && eltype(col_data) <: Union{Missing, AbstractString}
            df_clean[!, col] = Impute.impute(col_data, Impute.mode)
        elseif method in ["ffill", "bfill"]
            df_clean[!, col] = Impute.locf(df[!, col]; rev=(method == "bfill"))
        end
    end

    return df_clean
end

# ----------------------------
# Helpers
# ----------------------------
function detect_time_column(df::DataFrame)
    for col in names(df)
        if eltype(df[!, col]) <: Union{Date, DateTime}
            return col
        end
    end
    return nothing
end

# ----------------------------
# ARIMA Imputation
# ----------------------------
function arima_impute(df::DataFrame, col::Symbol, time_col::Symbol)::Vector
    perm = sortperm(df[!, time_col])
    df_sorted = df[perm, :]
    y_sorted = df_sorted[!, col]
    observed = collect(skipmissing(y_sorted))
    result = Vector{eltype(df[!, col])}(undef, nrow(df))

    if length(observed) < 10
        println("⚠️ Too few values for ARIMA.")
        result[perm] = y_sorted
        return result
    end

    history = Float64.(observed)
    y_filled = copy(y_sorted)

    for i in eachindex(y_sorted)
        if ismissing(y_sorted[i])
            model = SARIMA(history, order=(1, 0, 0), seasonal=(0, 0, 0, 0))
            results = fit(model)
            prediction = forecast(results, 1)

            value = if prediction isa AbstractVector
                prediction[1]
            elseif hasproperty(prediction, :predicted_mean)
                getproperty(prediction, :predicted_mean)[1]
            elseif hasproperty(prediction, :forecast)
                getproperty(prediction, :forecast)[1]
            else
                error("Unsupported forecast result from StateSpaceModels")
            end

            y_filled[i] = convert(eltype(y_filled), value)
            push!(history, Float64(value))
        end
    end

    result[perm] = y_filled
    return result
end

# ----------------------------
# RNN Imputation
# ----------------------------
function rnn_impute(df::DataFrame, col::Symbol, time_col::Symbol; window=5, epochs=50)
    perm = sortperm(df[!, time_col])
    df_sorted = df[perm, :]
    y_sorted = df_sorted[!, col]
    values = skipmissing(y_sorted) |> collect |> Float32.
    result = Vector{eltype(df[!, col])}(undef, nrow(df))

    if length(values) < window + 5
        println("⚠️ Too few values for RNN.")
        result[perm] = y_sorted
        return result
    end

    X = [values[i:i+window-1] for i in 1:(length(values) - window)]
    Y = [values[i+window] for i in 1:(length(values) - window)]

    x_data = hcat(X...) |> reshape(_, window, 1, :)
    y_data = Float32.(Y) |> reshape(1, 1, :)

    model = Chain(RNN(window, 16, tanh), Dense(16, 1))
    opt = ADAM()
    loss(x, y) = Flux.mse(model(x), y)
    ps = Flux.params(model)

    for _ in 1:epochs
        Flux.train!(loss, ps, [(x_data, y_data)], opt)
    end

    y_filled = copy(y_sorted)
    input_seq = values[end-window+1:end]

    for i in eachindex(y_sorted)
        if ismissing(y_sorted[i])
            x = reshape(Float32.(input_seq), window, 1, 1)
            prediction = model(x)[1]
            y_filled[i] = prediction
            push!(input_seq, prediction)
            popfirst!(input_seq)
        end
    end

    result[perm] = y_filled
    return result
end

# ----------------------------
# Advanced Imputation
# ----------------------------
function impute_advanced(df::DataFrame, method::String; threshold=0.1)
    df_clean = deepcopy(df)
    nrows = nrow(df)
    time_col = detect_time_column(df)
    report = DataFrame(Column=String[], Method=String[], ImputedCount=Int[])

    for col in names(df)
        if col == time_col
            continue
        end

        col_data = df[!, col]
        n_missing = count(ismissing, col_data)
        frac_missing = n_missing / nrows

        if frac_missing >= threshold && eltype(col_data) <: Union{Missing, Number}
            imputed_count = 0
            performed_imputation = false

            if method == "distribution"
                vals = collect(skipmissing(col_data))

                if !isempty(vals)
                    μ = mean(vals)
                    σ = std(vals)
                    if !isfinite(σ) || σ == 0
                        σ = eps(typeof(σ))
                    end

                    dist = Normal(μ, σ)
                    df_clean[!, col] = coalesce.(col_data, rand(dist, nrows))
                    imputed_count = n_missing
                    performed_imputation = true
                end

            elseif method == "arima" && time_col !== nothing
                println("ARIMA imputing $col...")
                before = count(ismissing, df[!, col])
                df_clean[!, col] = arima_impute(df, Symbol(col), Symbol(time_col))
                after = count(ismissing, df_clean[!, col])
                imputed_count = before - after
                performed_imputation = true

            elseif method == "rnn" && time_col !== nothing
                println("RNN imputing $col...")
                before = count(ismissing, df[!, col])
                df_clean[!, col] = rnn_impute(df, Symbol(col), Symbol(time_col))
                after = count(ismissing, df_clean[!, col])
                imputed_count = before - after
                performed_imputation = true
            end

            if performed_imputation
                push!(report, (col, method, imputed_count))
            end
        end
    end

    println("📝 Imputation Report:")
    show(report)

    return df_clean, report
end

# ----------------------------
# Save report
# ----------------------------
function save_imputation_report(report::DataFrame, filename::String)
    dir = dirname(filename)
    if !isempty(dir)
        mkpath(dir)
    end

    if endswith(filename, ".csv")
        CSV.write(filename, report)
    elseif endswith(filename, ".json")
        json_data = [Dict("Column"=>row.Column, "Method"=>row.Method, "ImputedCount"=>row.ImputedCount) for row in eachrow(report)]
        open(filename, "w") do io
            JSON.print(io, json_data)
        end
    else
        error("Unsupported format (use .csv or .json)")
    end
    println("✅ Report saved to $filename")
end

end # module
