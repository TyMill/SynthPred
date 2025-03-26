using Pkg

# Aktywuj środowisko dokumentacji, które ma tylko Documenter.jl
Pkg.activate("docs")
Pkg.instantiate()

using Documenter
using SynthPred  # Pakiet główny jest aktywny, bo CI startuje z głównego katalogu

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
