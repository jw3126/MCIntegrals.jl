
export MCVanilla, Vegas, Domain
export integral, ∫

using QuickTypes: @qstruct
using ArgCheck
using LinearAlgebra, Statistics
using StaticArrays
using Base.Threads: @threads
using Setfield: @settable

import Random

abstract type MCAlgorithm end

@settable @qstruct MCVanilla{R}(
        neval::Int64=10^6,
        rng::R=GLOBAL_PAR_RNG,
    ) do
        @argcheck neval > 2
end <: MCAlgorithm

"""
    Domain

Encodes a domain of integration, i.e. a product of finte intervals.
"""
struct Domain{N,T}
    lower::SVector{N,T}
    upper::SVector{N,T}
end

# function cartesian_product(d1::Domain, d2::Domain)
#     Domain(vcat(d1.lower, d2.lower), vcat(d1.upper, d2.upper))
# end

function pointtype(::Type{Domain{N,T}}) where {N,T}
    SVector{N,T}
end

function volume(dom::Domain)
    prod(dom.upper - dom.lower)
end

function Base.ndims(dom::Domain{N,T}) where {N,T}
    N
end

function Domain(interval::NTuple{2,Number})
    (a,b) = float.(promote(interval...))
    Domain(@SVector[a], @SVector[b])
end

function Domain(dom::Domain)
    dom
end

function Domain(lims)
    lower = SVector(map(float ∘ first, lims))
    upper = SVector(map(float ∘ last, lims))
    Domain(lower, upper)
end

function uniform(rng, dom::Domain)
    V = pointtype(typeof(dom))
    Δ = (dom.upper - dom.lower)
    dom.lower .+ rand(rng, V) .* Δ
end

"""

    integral(f, dom [,alg])

Compute the integral of the function `f` over a domain `dom`, via the algorithm `alg`.
"""
function integral(f, dom, alg=Vegas())
    f2, dom2, alg2 = canonicalize(f, dom, alg)
    integral_kernel(f2, dom2, alg2)
end

function canonicalize(f, dom, alg)
    f, Domain(dom), alg
end

function canonicalize(f, I::NTuple{2,Number}, alg)
    f_v = f ∘ first
    dom = Domain(I)
    f_v, dom, alg
end

const ∫ = integral

function integral_kernel(f, dom::Domain, alg::MCVanilla)
    mc_kernel(f, alg.rng, dom, neval=alg.neval)
end
function draw(rng, dom::Domain)
    vol = volume(dom)
    x = uniform(rng, dom)
    (position=x, weight = volume(dom))
end

"""
    mc_kernel(f, rng::AbstractRNG, dom; neval)

Monte Carlo integration of function `f` over `dom`.
`dom` must support the following methods:
* volume(dom): Return the volume of dom
* draw(rng, dom): Return an object with properties `position::SVector`, `weight::Real`.
"""
function mc_kernel(f, rng::AbstractRNG, dom; neval)
    N = neval
    x = uniform(rng, dom)
    y = float(f(x)) * volume(dom)
    sum = y
    sum2 = y.^2
    for _ in 1:(N-1)
        s = draw(rng, dom)
        y = s.weight * f(s.position)
        sum += y
        sum2 += y.^2
    end
    
    mean = sum/N
    var_f = (sum2/N) - mean .^2
    var_f = max(zero(var_f), var_f)
    var_fmean = var_f / N
    std = sqrt.(var_fmean)
    (value = mean, std=std, neval=N)
end

function mc_kernel(f, p::ParallelRNG, dom; neval)
    rngs = p.rngs
    res1 = mc_kernel(f, rngs[1], dom, neval=2)
    T = typeof(res1)
    nthreads = length(rngs)
    results = Vector{T}(undef, nthreads)
    neval_i = ceil(Int, neval / nthreads)
    @threads for i in 1:nthreads
        res = mc_kernel(f, rngs[i], dom, neval=neval_i)
        results[i] = res
    end
    fuseall(results)
end

function fuseall(results)
    N = sum(res->res.neval, results)
    value = sum(res->res.value * res.neval/N   , results)
    var   = sum(res->(res.std * res.neval/N).^2, results)
    (value=value, std=sqrt.(var), neval=N)
end

# function fuse(res1, res2)
#     var1 = res1.std .^ 2
#     var2 = res2.std .^ 2
#     w1, w2 = fusion_weights(var1, var2)
#     (
#         value = @.(w1 * res1.value + w2*res2.value),
#         std = @.(sqrt(w1^2*var1 + w2^2*var2))
#     )
# end
# 
# function fusion_weights(var1::AbstractVector, var2::AbstractVector)
#     pairs = fusion_weights_scalar.(var1, var2)
#     first.(pairs), last.(pairs)
# end
# 
# function fusion_weights(var1, var2)
#     fusion_weights_scalar(var1, var2)
# end
# 
# function fusion_weights_scalar(var1, var2)
#     z = zero(var1)
#     o = one(var1)
#     if iszero(var1)
#         (o, z)
#     elseif iszero(var2)
#         (z,o)
#     else
#         p1 = o/var1
#         p2 = o/var2
#         p  = p1 + p2
#         (p1/p, p2/p)
#     end
# end

