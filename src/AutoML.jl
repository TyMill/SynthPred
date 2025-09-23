module AutoML

using DataFrames
using Dates
using Random
using Statistics
using StatsBase
using Base.MathConstants: π

export run_automl, blend_top_models, predict_ensemble, EnsembleModel

"""A lightweight description of how to fit and predict with a model."""
struct ModelSpec{F,G}
    name::String
    fit::F
    predict::G
end

"""A fitted model paired with the specification that created it."""
struct FittedModel{S,M}
    spec::S
    model::M
end

const AnyModelSpec = ModelSpec{F,G} where {F,G}
const AnyFittedModel = FittedModel{S,M} where {S,M}

"""Container holding the fitted models used for prediction."""
struct EnsembleModel
    models::Vector{AnyFittedModel}
    labels::Vector{Any}
end

"""Encode a feature column into a numeric vector."""
function encode_column(col::AbstractVector)
    T = Base.nonmissingtype(eltype(col))

    if T <: Real
        data = Union{Missing, Float64}[ismissing(v) ? missing : Float64(v) for v in col]
        return fill_missing_numeric(data)
    elseif T <: Dates.TimeType
        data = Union{Missing, Float64}[ismissing(v) ? missing : Float64(Dates.value(v)) for v in col]
        return fill_missing_numeric(data)
    else
        str_data = String[ismissing(v) ? "__missing__" : string(v) for v in col]
        mapping = Dict{String, Int}()
        encoded = Vector{Float64}(undef, length(str_data))
        for (i, val) in enumerate(str_data)
            idx = get!(mapping, val, length(mapping) + 1)
            encoded[i] = Float64(idx)
        end
        return encoded
    end
end

"""Replace missing numeric values with the column mean."""
function fill_missing_numeric(vec::Vector{Union{Missing, Float64}})
    if isempty(vec)
        return Float64[]
    elseif all(ismissing, vec)
        return zeros(Float64, length(vec))
    end

    mean_val = mean(skipmissing(vec))
    return [Float64(coalesce(v, mean_val)) for v in vec]
end

"""Encode all feature columns of a ``DataFrame`` into a numeric matrix."""
function encode_features(df::DataFrame)
    if ncol(df) == 0
        return zeros(Float64, nrow(df), 0)
    end

    cols = [encode_column(df[!, name]) for name in names(df)]
    return hcat(cols...)
end

"""Encode the target vector into integer labels."""
function encode_target(y::AbstractVector)
    if any(ismissing, y)
        throw(ArgumentError("Target column contains missing values."))
    end

    values = collect(y)
    uniques = unique(values)
    mapping = Dict{Any, Int}()
    for (i, val) in enumerate(uniques)
        mapping[val] = i
    end

    encoded = [mapping[val] for val in values]
    return encoded, Vector{Any}(uniques)
end

"""Simple k-nearest neighbours classifier."""
struct KNNModel
    k::Int
    features::Matrix{Float64}
    labels::Vector{Int}
end

function fit_knn(X::Matrix{Float64}, y::Vector{Int}; k::Int=3)
    return KNNModel(k, X, y)
end

function predict_knn(model::KNNModel, Xnew::Matrix{Float64})
    n_samples = size(Xnew, 1)
    preds = Vector{Int}(undef, n_samples)

    for i in 1:n_samples
        distances = [sum((model.features[j, :] .- Xnew[i, :]).^2) for j in 1:size(model.features, 1)]
        k = min(model.k, length(distances))
        idx = partialsortperm(distances, 1:k)
        neighbor_labels = model.labels[idx]
        preds[i] = majority_vote(neighbor_labels)
    end

    return preds
end

"""Gaussian Naive Bayes classifier."""
struct GaussianNBModel
    classes::Vector{Int}
    priors::Vector{Float64}
    means::Vector{Vector{Float64}}
    vars::Vector{Vector{Float64}}
end

function fit_gaussian_nb(X::Matrix{Float64}, y::Vector{Int})
    classes = sort(collect(unique(y)))
    priors = Vector{Float64}(undef, length(classes))
    means = Vector{Vector{Float64}}(undef, length(classes))
    vars = Vector{Vector{Float64}}(undef, length(classes))

    for (idx, c) in enumerate(classes)
        mask = findall(==(c), y)
        subset = X[mask, :]
        priors[idx] = length(mask) / length(y)
        class_means = vec(mean(subset; dims=1))
        class_vars = vec(var(subset; dims=1, corrected=false))
        class_vars .= max.(class_vars, 1e-6)
        means[idx] = class_means
        vars[idx] = class_vars
    end

    return GaussianNBModel(classes, priors, means, vars)
end

function predict_gaussian_nb(model::GaussianNBModel, Xnew::Matrix{Float64})
    n_samples = size(Xnew, 1)
    preds = Vector{Int}(undef, n_samples)

    for i in 1:n_samples
        best_class = model.classes[1]
        best_logprob = -Inf

        for (idx, c) in enumerate(model.classes)
            logprob = log(model.priors[idx])
            for j in 1:length(model.means[idx])
                μ = model.means[idx][j]
                σ2 = model.vars[idx][j]
                x = Xnew[i, j]
                logprob += -0.5 * (log(2π * σ2) + (x - μ)^2 / σ2)
            end

            if logprob > best_logprob
                best_logprob = logprob
                best_class = c
            end
        end

        preds[i] = best_class
    end

    return preds
end

function majority_vote(labels::AbstractVector{Int})
    counts = countmap(labels)
    best_label = first(keys(counts))
    best_count = counts[best_label]

    for (label, count) in counts
        if count > best_count || (count == best_count && label < best_label)
            best_label = label
            best_count = count
        end
    end

    return best_label
end

function accuracy_score(pred::AbstractVector{Int}, truth::AbstractVector{Int})
    return isempty(truth) ? 0.0 : sum(pred .== truth) / length(truth)
end

function kfold_indices(n::Int, k::Int)
    k = clamp(k, 1, n)
    indices = collect(1:n)
    shuffle!(indices)

    fold_sizes = fill(div(n, k), k)
    remainder = rem(n, k)
    for i in 1:remainder
        fold_sizes[i] += 1
    end

    folds = Vector{Vector{Int}}(undef, k)
    start = 1
    for i in 1:k
        stop = start + fold_sizes[i] - 1
        folds[i] = indices[start:stop]
        start = stop + 1
    end

    return folds
end

function evaluate_model(spec::AnyModelSpec, X::Matrix{Float64}, y::Vector{Int}; nfolds::Int=5)
    n = size(X, 1)
    if n == 0
        return 0.0
    elseif n == 1
        model = spec.fit(X, y)
        preds = spec.predict(model, X)
        return accuracy_score(preds, y)
    end

    k = clamp(min(nfolds, n), 2, n)
    folds = kfold_indices(n, k)
    scores = Float64[]

    for i in 1:length(folds)
        test_idx = folds[i]
        train_idx = Int[]
        for j in 1:length(folds)
            j == i && continue
            append!(train_idx, folds[j])
        end

        model = spec.fit(X[train_idx, :], y[train_idx])
        preds = spec.predict(model, X[test_idx, :])
        push!(scores, accuracy_score(preds, y[test_idx]))
    end

    return mean(scores)
end

function default_model_specs()
    AnyModelSpec[
        ModelSpec("Gaussian Naive Bayes", fit_gaussian_nb, predict_gaussian_nb),
        ModelSpec("K-Nearest Neighbors (k=3)", (X, y) -> fit_knn(X, y; k=3), predict_knn)
    ]
end

"""
    run_automl(df::DataFrame, target_col::Symbol; max_models=5, nfolds=5)

Evaluate a collection of lightweight models and return the top performers
according to cross-validated accuracy.
"""
function run_automl(df::DataFrame, target_col::Symbol; max_models::Int=5, nfolds::Int=5)
    y_raw = df[!, target_col]
    X_df = select(df, Not(target_col))

    X = encode_features(X_df)
    y, _ = encode_target(y_raw)

    available = default_model_specs()
    limit = min(max_models, length(available))
    if limit <= 0
        @warn "max_models keyword resulted in no models being evaluated."
        return AnyModelSpec[], Float64[]
    end
    selected_specs = available[1:limit]

    results = Tuple{AnyModelSpec, Float64}[]
    for spec in selected_specs
        try
            score = evaluate_model(spec, X, y; nfolds=nfolds)
            push!(results, (spec, score))
        catch err
            @warn "Model $(spec.name) failed during evaluation" exception=err
        end
    end

    if isempty(results)
        @warn "No models were successfully evaluated."
        return AnyModelSpec[], Float64[]
    end

    sorted_results = sort(results, by = x -> -x[2])
    k = min(2, length(sorted_results))
    if k < 2
        @warn "Only $(k) model(s) successfully evaluated; returning available result(s)."
    end

    selected = sorted_results[1:k]
    println("🏆 Top $(k) Model$(k == 1 ? "" : "s"):")
    for (i, (spec, score)) in enumerate(selected)
        println("$(i). $(spec.name) | Accuracy: $(round(score, digits=4))")
    end

    top_models = first.(selected)
    performances = last.(selected)
    return top_models, performances
end

"""
    blend_top_models(models::AbstractVector{<:AnyModelSpec}, X::DataFrame, y)

Fit the selected models on the full dataset and build a simple majority-vote
ensemble. If only a single model is provided the fitted model is returned.
"""
function blend_top_models(models::AbstractVector{<:AnyModelSpec}, X::DataFrame, y)
    if isempty(models)
        throw(ArgumentError("blend_top_models requires at least one model, but none were provided."))
    end

    encoded_X = encode_features(X)
    encoded_y, labels = encode_target(y)

    selected = length(models) == 1 ? models : models[1:min(2, length(models))]
    fitted_models = AnyFittedModel[]

    for spec in selected
        fitted = spec.fit(encoded_X, encoded_y)
        push!(fitted_models, FittedModel(spec, fitted))
    end

    if length(selected) == 1
        @warn "Only one model provided; returning fitted model without ensembling."
    else
        println("✅ Ensemble model trained.")
    end

    return EnsembleModel(fitted_models, labels)
end

"""
    predict_ensemble(mach::EnsembleModel, Xnew::DataFrame)

Generate predictions from the fitted ensemble using majority voting.
"""
function predict_ensemble(mach::EnsembleModel, Xnew::DataFrame)
    if isempty(mach.models)
        return Any[]
    end

    encoded_X = encode_features(Xnew)
    votes = [model.spec.predict(model.model, encoded_X) for model in mach.models]

    n_samples = size(encoded_X, 1)
    aggregated = Vector{Int}(undef, n_samples)

    for i in 1:n_samples
        row = [votes[j][i] for j in 1:length(votes)]
        aggregated[i] = majority_vote(row)
    end

    return [mach.labels[idx] for idx in aggregated]
end

end # module
