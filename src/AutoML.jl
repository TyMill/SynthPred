module AutoML

using MLJ
using MLJTuning
using MLJEnsembles
using DataFrames
using Statistics

export run_automl, blend_top_models, predict_ensemble

"""
    run_automl(df::DataFrame, target_col::Symbol; max_models=5)

Trains multiple classifiers and selects up to the top 2 based on accuracy.
Returns the available models and their scores.
"""
function run_automl(df::DataFrame, target_col::Symbol; max_models=5)
    y = df[!, target_col]
    X = select(df, Not(target_col))

    schema = schema(X)
    X = coerce(X, schema.types)

    models = [
        (@load DecisionTreeClassifier verbosity=0)(),
        (@load RandomForestClassifier verbosity=0)(),
        (@load LogisticClassifier verbosity=0)(),
        (@load KNNClassifier verbosity=0)(),
        (@load GaussianNB verbosity=0)()
    ]

    results = []
    for model in models[1:min(max_models, length(models))]
        try
            mach = machine(model, X, y)
            eval = evaluate!(mach, resampling=CV(nfolds=5), measure=accuracy, verbosity=0)
            push!(results, (model, mean(eval.measurement)))
        catch err
            @warn "Model $(model) failed during evaluation" exception = err
        end
    end

    if isempty(results)
        @warn "No models were successfully evaluated."
        return Any[], Float64[]
    end

    sorted_results = sort(results, by=x -> -x[2])
    k = min(2, length(sorted_results))
    if k < 2
        @warn "Only $(k) model(s) successfully evaluated; returning available result(s)."
    end

    selected = sorted_results[1:k]
    top_models = first.(selected)
    performances = last.(selected)

    println("🏆 Top $(k) Model$(k == 1 ? "" : "s"):")
    for (i, m) in enumerate(top_models)
        println("$(i). $(m) | Accuracy: $(round(performances[i], digits=4))")
    end

    return top_models, performances
end

"""
    blend_top_models(models::Vector, X::DataFrame, y)::Machine

Creates a simple ensemble of the top models (average of probabilities) when
multiple are supplied, or fits a single model directly if only one is
available.
"""
function blend_top_models(models::Vector, X::DataFrame, y)::Machine
    if isempty(models)
        throw(ArgumentError("blend_top_models requires at least one model, but none were provided."))
    elseif length(models) == 1
        @warn "Only one model provided; returning fitted model without ensembling."
        mach = machine(only(models), X, y)
        fit!(mach)
        return mach
    end

    selected_models = models[1:2]
    weights = fill(1 / length(selected_models), length(selected_models))
    ensemble = EnsembleModel(atom=selected_models, weights=weights, strategy=:average_prob)
    mach = machine(ensemble, X, y)
    fit!(mach)
    println("✅ Ensemble model trained.")
    return mach
end

"""
    predict_ensemble(mach::Machine, Xnew::DataFrame)

Uses ensemble model to predict class probabilities and labels.
"""
function predict_ensemble(mach::Machine, Xnew::DataFrame)
    Xnew = coerce(Xnew, schema(Xnew).types)
    yhat = predict(mach, Xnew)

    # Attempt to convert probabilities to labels (for classifiers)
    try
        labels = mode.(yhat)
        return labels
    catch
        return yhat  # fallback: return probabilities
    end
end

end # module
