using MultiMLDatasets
using Documenter

DocMeta.setdocmeta!(MultiMLDatasets, :DocTestSetup, :(using MultiMLDatasets); recursive=true)

makedocs(;
    modules=[MultiMLDatasets],
    authors="Manuel Berkemeier <manuelbb@mail.uni-paderborn.de> and contributors",
    repo="https://github.com/manuelbb-upb/MultiMLDatasets.jl/blob/{commit}{path}#{line}",
    sitename="MultiMLDatasets.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://manuelbb-upb.github.io/MultiMLDatasets.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/manuelbb-upb/MultiMLDatasets.jl",
    devbranch="main",
)
