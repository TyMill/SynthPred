using Test
using CSV, DataFrames
using SynthPred.Exploration
using SynthPred.Imputer
using SynthPred.AutoML

@testset "Exploration module" begin
    df = DataFrame(a=[1, 2, missing, 4], b=["x", "y", "z", missing])
    @test typeof(describe_data(df)) == Nothing
    report = missing_report(df)
    @test size(report, 1) == 2
end

@testset "Imputer module - simple mean" begin
    df = DataFrame(x=[1.0, 2.0, missing, 4.0])
    df_imp = impute_simple(df, "mean")
    @test all(!ismissing, df_imp.x)
end

@testset "AutoML module - top model selection" begin
    df = DataFrame(
        feature1 = rand(10),
        feature2 = rand(["red", "blue", "green"], 10),
        feature3 = [Date(2024, 1, i+1) for i in 1:10],
        target = rand(["A", "B"], 10)
    )
    top_models, scores = run_automl(df, :target; max_models=3)
    @test length(top_models) == 2
    @test scores[1] >= 0 && scores[1] <= 1
end
