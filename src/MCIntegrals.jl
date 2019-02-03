module MCIntegrals

using Random: AbstractRNG

function rand(rng::AbstractRNG, args...)
    # make sure this package always explicitly passes
    # rng to rand calls
    Random.rand(rng, args...)
end

include("core.jl")
include("plots.jl")

end # module
