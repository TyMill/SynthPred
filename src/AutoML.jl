module AutoML

using MLJ
using MLJTuning
using MLJEnsembles
using DataFrames
using Statistics

export run_automl, blend_top_models, predict_ensemble

"""
    run_automl(df::DataFrame, target_col::Symbol; max_models=5)

Trains multiple classifiers and selects top 2 based on accuracy.
Returns top 2 models and their scores.
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
        mach = machine(model, X, y)
        eval = evaluate!(mach, resampling=CV(nfolds=5), measure=accuracy, verbosity=0)
        push!(results, (model, mean(eval.measurement)))
    end

    sorted_results = sort(results, by=x -> -x[2])
    top_models = [x[1] for x in sorted_results[1:2]]
    performances = [x[2] for x in sorted_results[1:2]]

    println("🏆 Top 2 Models:")
    for (i, m) in enumerate(top_models)
        println("$(i). $(m) | Accuracy: $(round(performances[i], digits=4))")
    end

    return top_models, performances
end

"""
    blend_top_models(models::Vector, X::DataFrame, y)::Machine

Creates a simple ensemble of the top 2 models (average of probabilities).
"""
function blend_top_models(models::Vector, X::DataFrame, y)::Machine
    model1, model2 = models
    ensemble = EnsembleModel(atom=[model1, model2], weights=[0.5, 0.5], strategy=:average_prob)
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
