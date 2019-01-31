module MCIntegrals

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

function cartesian_product(d1::Domain, d2::Domain)
    Domain(vcat(d1.lower, d2.lower), vcat(d1.upper, d2.upper))
end

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

function integral(f, domain, alg=Vegas())
    integral_dom(f, domain, alg)
end

function integral_dom(f, dom, alg)
    integral_alg(f, Domain(dom), alg)
end

function integral_dom(f, I::NTuple{2,Number}, alg)
    f_v = f ∘ first
    dom = Domain(I)
    integral_alg(f_v, dom, alg)
end

const ∫ = integral

function integral_alg(f, dom, alg::MCVanilla)
    N = alg.neval
    x = uniform(dom)
    y = f(x)
    sum = y
    sum2 = y^2
    for _ in 1:(N-1)
        x = uniform(dom)
        y = f(x)
        sum += y
        sum2 += y^2
    end
    
    vol = volume(dom)
    mean = sum/N
    var_f = (sum2/N) - mean^2
    var_f = max(zero(var_f), var_f)
    var_fmean = var_f / N
    std = sqrt(var_fmean)
    (value = mean*vol, std=std*vol)
end

# TODO renames, this is not really Vegas algorithm
@qstruct Vegas(
    neval::Int=10^4,
    ) do
    @argcheck neval > 2
end <: MCAlgorithm


@qstruct Stratification{N,T}(
    boundaries::NTuple{N,Vector{T}}
   ) do
    for i in 1:N
        @argcheck length(boundaries[i]) >= 2
        @argcheck issorted(boundaries[i])
    end
end

@qstruct CDF(
    positions::Vector{Float64}, 
    values::Vector{Float64}, 
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

function importance_quantiles(cdf::CDF, nwalls)
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
    # if i != length(cdf.values)
    #     @show cdf.values
    # end
    # @assert i == length(cdf.values)
    @assert length(ret) == nwalls
    ret
end

function Base.size(s::Stratification)
    map(s.boundaries) do b
        length(b) - 1
    end
end
Base.length(s::Stratification) = prod(size(s))
Base.eachindex(s::Stratification) = CartesianIndices(s)
Base.axes(s::Stratification) = map(Base.OneTo, size(s))
function Base.CartesianIndices(s::Stratification)
    CartesianIndices(axes(s))
end
function Base.LinearIndices(s::Stratification)
    LinearIndices(axes(s))
end

function rand_index(s::Stratification)
    li = rand(LinearIndices(s))
    CartesianIndices(s)[li]
end

function Base.getindex(s::Stratification, index::CartesianIndex)
    lower = map(s.boundaries, Tuple(index)) do walls, i
        walls[i]
    end
    upper = map(s.boundaries, Tuple(index)) do walls, i
        walls[i+1]
    end
    Domain(SVector(lower), SVector(upper))
end

function create_sample(s::Stratification)
    i = rand_index(s)
    cell = s[i]
    x = uniform(cell)
    # wt = volume(cell)
    (x=x, index=i, cell=cell)
end

function estimate_cdf(histogram)
    h = histogram
    xs = h.walls
    cdf_val = 0.
    cdf_values = [cdf_val]
    for i in eachindex(h.counts)
        reg_sum   = 0. #1e-1*total_sum/n
        reg_count = 0. #1
        vol = xs[i+1] - xs[i]
        cdf_val += h.sums[i] / h.counts[i]
        push!(cdf_values, cdf_val)
    end
    CDF(xs, cdf_values)
end

function tune(f, iq::Stratification; neval=1000, outsize=size(iq))
    hists = map(iq.boundaries) do xs
        n = length(xs)
        (
            walls = xs,
            counts = fill(0, n-1),
            sums = fill(0., n-1),
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
        cdf = estimate_cdf(h)
        importance_quantiles(cdf, nwalls)
    end

    Stratification(bdries_new)
end

function integral_alg(f, iq::Stratification, alg::Vegas)
    N = alg.neval
    sum = 0.
    sum2 = 0.
    for _ in 1:N
        s = create_sample(iq)
        wt = volume(s.cell) * length(iq)
        y = f(s.x) * wt
        sum += y
        sum2 += y^2
    end
    mean = sum/N
    var_f = (sum2/N) - mean^2
    var_f = max(0., var_f)
    var_fmean = var_f / N
    std = sqrt(var_fmean)
    (value = mean, std=std)
end

function default_strat_size(dom::Domain{N}) where {N}
    ntuple(_ -> 100, Val(N))
end

function equidistant_stratification(
        dom::Domain, 
        size=default_strat_size(dom)
    )

    bdries = map(Tuple(dom.lower), Tuple(dom.upper), size) do lo, hi, nbins
        nwalls = nbins + 1
        r = range(lo, stop=hi, length=nwalls)
        collect(r)
    end
    Stratification(bdries)
end

function integral_alg(f, dom::Domain, alg::Vegas)
    iq = equidistant_stratification(dom)
    iq = tune(f, iq, neval=2000)
    iq = tune(f, iq, neval=2000)
    iq = tune(f, iq, neval=2000)
    integral_alg(f, iq, alg)
end

end # module
