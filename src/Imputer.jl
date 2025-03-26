module Imputer

using DataFrames, Statistics, StatsBase, Distributions
using Impute, Dates, JSON, CSV
using ARIMA
using Flux

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
        if eltype(df[!, col]) <: Union{Missing, Number}
            col_data = df[!, col]
            if method == "mean"
                df_clean[!, col] = Impute.impute(col_data, Impute.mean)
            elseif method == "median"
                df_clean[!, col] = Impute.impute(col_data, Impute.median)
            elseif method == "mode"
                df_clean[!, col] = Impute.impute(col_data, Impute.mode)
            end
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
    df_sorted = sort(df, time_col)
    y = df_sorted[!, col]
    ts_vals = collect(skipmissing(y))

    if length(ts_vals) < 10
        println("⚠️ Too few values for ARIMA.")
        return y
    end

    model = fit(ARIMA, ts_vals, 1, 0, 0)
    y_filled = copy(y)

    for i in eachindex(y)
        if ismissing(y[i])
            forecast = predict(model, 1)
            y_filled[i] = forecast[1]
            push!(ts_vals, forecast[1])
            model = fit(ARIMA, ts_vals, 1, 0, 0)
        end
    end
    return y_filled
end

# ----------------------------
# RNN Imputation
# ----------------------------
function rnn_impute(df::DataFrame, col::Symbol, time_col::Symbol; window=5, epochs=50)
    df_sorted = sort(df, time_col)
    y = df_sorted[!, col]
    values = skipmissing(y) |> collect |> Float32.

    if length(values) < window + 5
        println("⚠️ Too few values for RNN.")
        return y
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

    y_filled = copy(y)
    input_seq = values[end-window+1:end]

    for i in eachindex(y)
        if ismissing(y[i])
            x = reshape(Float32.(input_seq), window, 1, 1)
            prediction = model(x)[1]
            y_filled[i] = prediction
            push!(input_seq, prediction)
            popfirst!(input_seq)
        end
    end
    return y_filled
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

            if method == "distribution"
                vals = skipmissing(col_data)
                dist = Normal(mean(vals), std(vals))
                df_clean[!, col] = coalesce.(col_data, rand(dist, nrows))
                imputed_count = n_missing

            elseif method == "arima" && time_col !== nothing
                println("ARIMA imputing $col...")
                before = count(ismissing, df[!, col])
                df_clean[!, col] = arima_impute(df, Symbol(col), Symbol(time_col))
                after = count(ismissing, df_clean[!, col])
                imputed_count = before - after

            elseif method == "rnn" && time_col !== nothing
                println("RNN imputing $col...")
                before = count(ismissing, df[!, col])
                df_clean[!, col] = rnn_impute(df, Symbol(col), Symbol(time_col))
                after = count(ismissing, df_clean[!, col])
                imputed_count = before - after
            end

            push!(report, (col, method, imputed_count))
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
