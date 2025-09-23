using Test
using CSV, DataFrames
using Dates
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

@testset "Imputer module - simple mode handles strings" begin
    df = DataFrame(category=Union{Missing, String}["apple", "banana", missing, "apple", missing])
    expected = coalesce.(df.category, "apple")
    df_imp = impute_simple(df, "mode")

    @test all(!ismissing, df_imp.category)
    @test df_imp.category == expected
end

@testset "Imputer module - distribution handles constant column" begin
    df = DataFrame(x=[1.0, 1.0, 1.0, missing])
    df_imp, report = impute_advanced(df, "distribution")
    @test all(!ismissing, df_imp.x)
    @test size(report, 1) == 1
    @test report.Method[1] == "distribution"
end

@testset "Imputer module - advanced preserves row order" begin
    dates = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 12))
    values = Union{Missing, Float64}[1.0, 2.0, missing, 4.0, 5.0, 6.0, missing, 8.0, 9.0, 10.0, 11.0, 12.0]
    df_sorted = DataFrame(time=dates, value=values)
    perm = reverse(1:length(dates))
    df_shuffled = df_sorted[perm, :]

    imputed_sorted, _ = impute_advanced(df_sorted, "arima")
    imputed_shuffled, _ = impute_advanced(df_shuffled, "arima")

    @test all(!ismissing, imputed_sorted.value)
    @test all(!ismissing, imputed_shuffled.value)
    @test imputed_shuffled.value == imputed_sorted.value[perm]
    @test all(isfinite, imputed_sorted.value)
end

@testset "Imputer module - save report builds directories" begin
    report = DataFrame(Column=["x"], Method=["mean"], ImputedCount=[1])
    temp_dir = mktempdir()
    report_path = joinpath(temp_dir, "nested", "reports", "imputation_report.csv")

    save_imputation_report(report, report_path)

    @test isfile(report_path)
    loaded = CSV.read(report_path, DataFrame)
    @test loaded.Column == report.Column
    @test loaded.Method == report.Method
    @test loaded.ImputedCount == report.ImputedCount
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

@testset "AutoML module - handles limited models" begin
    df = DataFrame(
        feature1 = rand(10),
        feature2 = rand(["red", "blue"], 10),
        target = rand(["A", "B"], 10)
    )

    top_models, scores = run_automl(df, :target; max_models=1)
    @test length(top_models) == 1
    @test length(scores) == 1

    X = select(df, Not(:target))
    y = df[:, :target]
    mach = blend_top_models(top_models, X, y)
    @test !isnothing(mach)
end
