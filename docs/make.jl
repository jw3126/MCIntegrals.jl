using Documenter, MCIntegrals

makedocs(;
    modules=[MCIntegrals],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/jw3126/MCIntegrals.jl/blob/{commit}{path}#L{line}",
    sitename="MCIntegrals.jl",
    authors="Jan Weidner",
    assets=[],
)

deploydocs(;
    repo="github.com/jw3126/MCIntegrals.jl",
)
