import Random
using Random: AbstractRNG, MersenneTwister

# TODO use KissThreading, for parallelization 
# once it is on metadata
# https://github.com/mohamed82008/KissThreading.jl

function rand(rng::AbstractRNG, args...)
    # make sure this package always explicitly passes
    # rng to rand calls
    Random.rand(rng, args...)
end

struct ParallelRNG{R <: AbstractRNG}
    rngs::Vector{R}
end

function rand(p::ParallelRNG, args...)
    rng = first(p.rngs)
    Random.rand(rng, args...)
end

const GLOBAL_PAR_RNG = let
    rngs = map(MersenneTwister, 1:Sys.CPU_THREADS)
    ParallelRNG(rngs)
end
