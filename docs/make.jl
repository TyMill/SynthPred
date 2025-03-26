using Documenter
using SynthPred

makedocs(
    sitename = "SynthPred.jl Documentation",
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md"
    ],
    authors = ["Tymoteusz Miller"],
    repo = "https://github.com/TyMill/SynthPred.jl",
    modules = [SynthPred],
    clean = true
)
