
export MCVanilla, Vegas, Domain
export integral, ∫

using QuickTypes
using ArgCheck
using LinearAlgebra, Statistics
using StaticArrays

abstract type MCAlgorithm end

@qstruct MCVanilla(neval::Int64=10^6) do
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

function uniform(dom::Domain)
    V = pointtype(typeof(dom))
    Δ = (dom.upper - dom.lower)
    dom.lower .+ rand(V) .* Δ
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
    N = alg.neval
    x = uniform(dom)
    y = float(f(x))
    sum = y
    sum2 = y.^2
    for _ in 1:(N-1)
        x = uniform(dom)
        y = f(x)
        sum += y
        sum2 += y.^2
    end
    
    vol = volume(dom)
    mean = sum/N
    var_f = (sum2/N) - mean .^2
    var_f = max(zero(var_f), var_f)
    var_fmean = var_f / N
    std = sqrt.(var_fmean)
    (value = mean*vol, std=std*vol)
end

abstract type VegasDamping end
struct NoDamping <: VegasDamping end

"""
Damping for quantile estimation as proposed in original Vegas Paper by Lepage.
"""
@qstruct LepageDamping{T}(;
    alpha::T=1) <: VegasDamping

# TODO renames, this is not really Vegas algorithm
@qstruct Vegas{R}(
    neval::Int=10^4,
    regularization::R=LepageDamping(),
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

function uniform(iq::VegasGrid)
    uniform(domain(iq))
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

function rand_index(s::VegasGrid)
    li = rand(LinearIndices(s))
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

function create_sample(s::VegasGrid)
    i = rand_index(s)
    cell = s[i]
    x = uniform(cell)
    # wt = volume(cell)
    (x=x, index=i, cell=cell)
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

function estimate_cdf(histogram, r::VegasDamping)
    pdf = estimate_pdf(histogram, r)
    cdf_vals = cumsum(pdf)
    z = zero(eltype(cdf_vals))
    pushfirst!(cdf_vals, z)
    CDF(histogram.walls, cdf_vals)
end

function tuneonce(f, iq::VegasGrid; 
              neval=1000, 
              outsize=size(iq),
              regularization::VegasDamping=LepageDamping(),
            )
    s = create_sample(iq)
    y = float(norm(f(s.x)))

    hists = map(iq.boundaries) do xs
        nbins = length(xs) - 1
        (
            walls = xs,
            counts = fill(0, nbins),
            sums = fill(zero(typeof(y)), nbins),
        )
    end

    for _ in 1:neval
        s = create_sample(iq)
        y = norm(f(s.x))
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
        cdf = estimate_cdf(h, regularization)
        quantiles(cdf, nwalls)
    end

    VegasGrid(bdries_new)
end

function integral_kernel(f, dom::VegasGrid, alg::Vegas)
    N = alg.neval
    x = uniform(dom)
    y = float(f(x))
    sum = y
    sum2 = y.^2
    for _ in 1:(N-1)
        s = create_sample(dom)
        wt = volume(s.cell) * length(dom)
        y = wt * f(s.x)
        sum += y
        sum2 += y.^2
    end
    
    mean = sum/N
    var_f = (sum2/N) - mean .^2
    var_f = max(zero(var_f), var_f)
    var_fmean = var_f / N
    std = sqrt.(var_fmean)
    (value = mean, std=std)
end

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
    iq = tuneonce(f, iq, neval=2000)
    iq = tuneonce(f, iq, neval=2000)
    iq = tuneonce(f, iq, neval=2000)
    iq = tuneonce(f, iq, neval=2000)
    iq = tuneonce(f, iq, neval=2000)
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
