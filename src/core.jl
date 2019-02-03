
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

abstract type VegasDamping end
struct NoDamping <: VegasDamping end

"""
Damping for quantile estimation as proposed in original Vegas Paper by Lepage.
"""
@qstruct LepageDamping{T}(;
    alpha::T=1) <: VegasDamping

# TODO renames, this is not really Vegas algorithm
@settable @qstruct Vegas{RNG,REG}(
        neval::Int=10^4;
        rng::RNG=GLOBAL_PAR_RNG,
        regularization::REG=LepageDamping(),
    ) do
        @argcheck neval > 2
end <: MCAlgorithm

@qstruct VegasGrid{N,T}(
    boundaries::NTuple{N,Vector{T}}
   ) do
    for i in 1:N
        @argcheck length(boundaries[i]) >= 2
        @argcheck issorted(boundaries[i])
    end
end

function domain(iq::VegasGrid)
    lo = map(first, iq.boundaries)
    hi = map(last , iq.boundaries)
    Domain(SVector(lo), SVector(hi))
end

function volume(iq::VegasGrid)
    volume(domain(iq))
end

function uniform(rng, iq::VegasGrid)
    uniform(rng, domain(iq))
end

@qstruct CDF{P,V}(
    positions::Vector{P}, 
    values::Vector{V}, 
    ) do
    @argcheck first(values) == 0.
    @argcheck issorted(values)
    @argcheck issorted(positions)
    @argcheck length(positions) == length(values)
    @argcheck length(positions) >= 2
end

function linterpol(x, (x_min, x_max), (y_min, y_max))
    @argcheck x_min <= x <= x_max
    if x == x_min
        y_min
    elseif x == x_max
        y_max
    else
        w1 = x_max - x
        w2 = x - x_min
        w = w1 + w2
        @assert w1 + w2 ≈ w
        y_min * w1/w + y_max * w2/w
    end
end

function quantiles(cdf::CDF, nwalls)
    ret = [first(cdf.positions)]
    qvals = range(first(cdf.values), stop=last(cdf.values), length=nwalls)
    i = 1
    for qval in qvals[2:end]
        while cdf.values[i] < qval
            i += 1
        end
        xs = cdf.values[i-1], cdf.values[i]
        ys = cdf.positions[i-1], cdf.positions[i]
        q = linterpol(qval, xs, ys)
        push!(ret, q)
    end
    @assert length(ret) == nwalls
    ret
end

function Base.size(s::VegasGrid)
    map(s.boundaries) do b
        length(b) - 1
    end
end
Base.length(s::VegasGrid) = prod(size(s))
Base.eachindex(s::VegasGrid) = CartesianIndices(s)
Base.axes(s::VegasGrid) = map(Base.OneTo, size(s))
function Base.CartesianIndices(s::VegasGrid)
    CartesianIndices(axes(s))
end
function Base.LinearIndices(s::VegasGrid)
    LinearIndices(axes(s))
end

function rand_index(rng, s::VegasGrid)
    li = rand(rng, LinearIndices(s))
    CartesianIndices(s)[li]
end

function Base.getindex(s::VegasGrid, index::CartesianIndex)
    lower = map(s.boundaries, Tuple(index)) do walls, i
        walls[i]
    end
    upper = map(s.boundaries, Tuple(index)) do walls, i
        walls[i+1]
    end
    Domain(SVector(lower), SVector(upper))
end

function draw_x_index_cell(rng, s::VegasGrid)
    i = rand_index(rng,s)
    cell = s[i]
    x = uniform(rng, cell)
    (position=x, index=i, cell=cell)
end

function draw(rng, dom::Domain)
    vol = volume(dom)
    x = uniform(rng, dom)
    (position=x, weight = volume(dom))
end

function draw(rng, iq::VegasGrid)
    s = draw_x_index_cell(rng, iq)
    wt = volume(s.cell) * length(iq)
    (position=s.position, weight = wt)
end

function estimate_pdf(histogram, ::NoDamping)
    pdf = map(/, histogram.sums, histogram.counts)
    pdf ./= sum(pdf)
    pdf
end

function estimate_pdf(histogram, r::LepageDamping)
    pdf = estimate_pdf(histogram, NoDamping())
    updf = map(pdf) do p
        ((p - 1) / log(p))^r.alpha
    end
    normalize!(updf, 1)
end

function estimate_cdf(histogram, alg::Vegas)
    pdf = estimate_pdf(histogram, alg.regularization)
    cdf_vals = cumsum(pdf)
    z = zero(eltype(cdf_vals))
    pushfirst!(cdf_vals, z)
    CDF(histogram.walls, cdf_vals)
end

function tuneonce(f, iq::VegasGrid,
              alg::Vegas=Vegas();
              neval=1000, 
              outsize=size(iq),
            )
    s = draw_x_index_cell(alg.rng, iq)
    y = float(norm(f(s.position)))

    hists = map(iq.boundaries) do xs
        nbins = length(xs) - 1
        (
            walls = xs,
            counts = fill(0, nbins),
            sums = fill(zero(typeof(y)), nbins),
        )
    end

    for _ in 1:neval
        s = draw_x_index_cell(alg.rng, iq)
        y = norm(f(s.position))
        cell = s.cell
        for i in eachindex(iq.boundaries)
            h = hists[i]
            wt = cell.upper[i] - cell.lower[i]
            h.counts[s.index[i]] += 1
            h.sums[s.index[i]] += y*wt
        end
    end

    bdries_new = map(hists, outsize) do h, ncells
        nwalls = ncells + 1
        cdf = estimate_cdf(h, alg)
        quantiles(cdf, nwalls)
    end

    VegasGrid(bdries_new)
end

function integral_kernel(f, dom::VegasGrid, alg::Vegas)
    mc_kernel(f, alg.rng, dom; neval=alg.neval)
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

function default_grid_size(dom::Domain{N}) where {N}
    ntuple(_ -> 100, Val(N))
end

function equidistant_grid(
        dom::Domain, 
        size=default_grid_size(dom)
    )

    bdries = map(Tuple(dom.lower), Tuple(dom.upper), size) do lo, hi, nbins
        nwalls = nbins + 1
        r = range(lo, stop=hi, length=nwalls)
        collect(r)
    end
    VegasGrid(bdries)
end

function initvr(f, dom::Domain, alg::Vegas)
    equidistant_grid(dom)
end

function tune(f, iq::VegasGrid, alg::Vegas)
    iq = tuneonce(f, iq, alg, neval=2000)
    iq = tuneonce(f, iq, alg, neval=2000)
    iq = tuneonce(f, iq, alg, neval=2000)
    iq = tuneonce(f, iq, alg, neval=2000)
    iq = tuneonce(f, iq, alg, neval=2000)
end

function tune(f, dom::Domain, alg)
    iq = initvr(f, dom, alg)
    tune(f, iq, alg)
end

function integral_kernel(f, dom::Domain, alg::Vegas)
    iq = tune(f, dom, alg)
    integral_kernel(f, iq, alg)
end

function canonicalize(f, dom::VegasGrid, alg::Vegas)
    f, dom, alg
end
