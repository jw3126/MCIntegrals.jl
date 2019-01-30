module MCIntegrals

export MCVanilla, Vegas
export integral

using QuickTypes
using ArgCheck
using LinearAlgebra, Statistics

abstract type MCAlgorithm end

@qstruct MCVanilla(neval::Int64=10^6) do
    @argcheck neval > 2
end <: MCAlgorithm

function uniform(min, max)
    min + rand() * (max - min)
end

function integral(f, domain, alg=Vegas)
    integral_alg(f, domain, alg)
end

function integral_alg(f, (a,b), alg::MCVanilla)
    N = alg.neval
    x = uniform(a,b)
    y = f(x)
    sum = y
    sum2 = y^2
    for _ in 1:(N-1)
        x = uniform(a,b)
        y = f(x)
        sum += y
        sum2 += y^2
    end
    
    vol = b - a
    mean = sum/N
    var_f = (sum2/N) - mean^2
    var_f = max(0., var_f)
    var_fmean = var_f / N
    std = sqrt(var_fmean)
    (value = mean*vol, std=std*vol)
end

@qstruct Vegas(
    neval::Int=10^4,
    ) do
    @argcheck neval > 2
end <: MCAlgorithm

@qstruct ImportanceQuantiles(positions::Vector{Float64}) do
    @argcheck issorted(positions)
    @argcheck length(positions) >= 2
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

function importance_quantiles(cdf::CDF, nquantiles)
    ret = [first(cdf.positions)]
    qvals = range(first(cdf.values), stop=last(cdf.values), length=nquantiles)
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
    @assert i == length(cdf.values)
    @assert length(ret) == nquantiles
    ImportanceQuantiles(ret)
end

function create_sample(iq::ImportanceQuantiles)
    xs = iq.positions
    inds = 1:length(xs) - 1
    i = rand(inds)
    x_lo = xs[i]
    x_hi = xs[i+1]
    Δ = x_hi - x_lo
    x = x_lo + rand() * Δ
    @assert x_lo <= x <= x_hi
    # vol = last(xs) - first(xs)
    # Δ_avg = vol 
    wt = Δ #* (length(iq.positions) - 1)
    (x=x, index=i, weight=wt)
end

function tune(f, iq::ImportanceQuantiles; neval=1000, nquantiles=length(iq.positions))
    xs = iq.positions
    n = length(xs)
    counts = fill(0, n-1)
    sums = fill(0., n-1)
    for _ in 1:neval
        s = create_sample(iq)
        y = norm(f(s.x))
        counts[s.index] += 1
        sums[s.index] += y*s.weight
    end
    
    cdf_val = 0.
    cdf_values = [cdf_val]
    for i in eachindex(counts)
        reg_sum   = 0. #1e-1*total_sum/n
        reg_count = 0. #1
        vol = iq.positions[i+1] - iq.positions[i]
        cdf_val += sums[i] / counts[i]
        push!(cdf_values, cdf_val)
    end
    cdf = CDF(iq.positions, cdf_values)
    importance_quantiles(cdf, nquantiles)
end

function integral_alg(f, iq::ImportanceQuantiles, alg::Vegas)
    N = alg.neval
    sum = 0.
    sum2 = 0.
    xs = iq.positions
    inds = 1:length(xs)-1
    vol = last(xs) - first(xs)
    for _ in 1:N
        s = create_sample(iq)
        y = f(s.x) * s.weight * (length(iq.positions) - 1)
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

function integral_alg(f, (a,b), alg::Vegas)
    iq = ImportanceQuantiles([a,b])
    iq = tune(f, iq, neval=200, nquantiles=10)
    iq = tune(f, iq, neval=2000, nquantiles=100)
    iq = tune(f, iq, neval=2000, nquantiles=100)
    integral_alg(f, iq, alg)
end

end # module
