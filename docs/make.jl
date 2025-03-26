using Pkg
Pkg.develop(path = joinpath(@__DIR__, ".."))

using Documenter
using SynthPred

makedocs(
    sitename = "SynthPred.jl",
    modules = [SynthPred],
    format = Documenter.HTML(),
    pages = ["Home" => "index.md"]
)

deploydocs(
    repo = "github.com/TyMill/SynthPred.jl",
    target = "build"
)
