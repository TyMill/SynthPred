using Pkg
Pkg.activate("docs")
Pkg.instantiate()

using Documenter
using SynthPred

makedocs(
    sitename = "SynthPred.jl",
    modules = [SynthPred],
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(
    repo = "github.com/TyMill/SynthPred.jl",
    target = "build"
)
