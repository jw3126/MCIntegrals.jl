using Setfield

export Vegas

abstract type VegasDamping end
struct NoDamping <: VegasDamping end

"""
Damping for quantile estimation as proposed in original Vegas Paper by Lepage.
"""
@qstruct LepageDamping{T}(;
    alpha::T=1) <: VegasDamping

# TODO renames, this is not really Vegas algorithm
@settable @qstruct Vegas{RNG,REG}(
        neval::Int=10^5;
        niter::Int=10,
        nbins::Int=max(2, neval ÷ (niter*100)), # number of bins along each dimension
        ndrop::Int=niter÷2,
        rng::RNG=GLOBAL_PAR_RNG,
        regularization::REG=LepageDamping(),
    ) do
        @argcheck neval > 2
        @argcheck niter >= 1
end <: MCAlgorithm

@qstruct VegasGrid{N,T}(
    boundaries::NTuple{N,Vector{T}}
   ) do
    for i in 1:N
        @argcheck length(boundaries[i]) >= 2
        @argcheck issorted(boundaries[i])
    end
end

function Base.:(==)(grid1::VegasGrid, grid2::VegasGrid)
    grid1.boundaries == grid2.boundaries
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

function draw(rng, iq::VegasGrid)
    s = draw_x_index_cell(rng, iq)
    wt = volume(s.cell) * length(iq)
    (position=s.position, weight = wt)
end

struct VegasHist{N,D,NT,T}
    grid    ::VegasGrid{N,D}
    normsums::NTuple{N,Vector{NT   }}
    counts  ::NTuple{N,Vector{Int64}}
    sum     ::Base.RefValue{T}
    sum2    ::Base.RefValue{T}
end

struct VegasVR{N,D,T}
    grid::VegasGrid{N,D}
    values::Vector{T}
    vars::Vector{T}
    neval::Int64
end

@qstruct CDF{P,V}(
    positions::Vector{P}, 
    values::Vector{V}, 
    isfinite::Bool,
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
    cdf.isfinite || return cdf.positions
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

function midpoint(dom::Domain)
    middle.(dom.lower, dom.upper)
end

function midpoint(grid::VegasGrid)
    midpoint(domain(grid))
end

function init_vegas_hist(f, vr)
    x = midpoint(vr.grid)
    y = f(x)
    T = typeof(float.(y))
    NT = typeof(float(norm(y)))
    bincounts = map(vr.grid.boundaries) do walls
        length(walls) - 1
    end
    normsums = map(bincounts) do nbins
        fill(zero(NT), nbins)
    end
    counts = map(bincounts) do nbins
        fill(zero(Int64), nbins)
    end
    sum  = Ref(zero(T))
    sum2 = Ref(zero(T))
    VegasHist(vr.grid, normsums, counts, sum, sum2)
end

function create_vegas_hist(f, vr, alg)
    create_vegas_hist(f, alg.rng, vr, alg)
end

function create_vegas_hist(f, rng::AbstractRNG, vr, alg)
    h = init_vegas_hist(f, vr)
    fill_vegas_hist!(f,h,rng; neval=ceil(Int,alg.neval/alg.niter))
end

function create_vegas_hist(f, p::ParallelRNG, vr, alg)
    rngs = p.rngs
    nthreads = length(rngs)
    h1 = init_vegas_hist(f, vr)
    hists = [deepcopy(h1) for _ in 1:nthreads]
    neval_i = ceil(Int, alg.neval / alg.niter / nthreads)
    @threads for i in 1:nthreads
        fill_vegas_hist!(f, hists[i], rngs[i], neval=neval_i)
    end
    fuseall(hists)
end

function fuse(h1::VegasHist, h2::VegasHist)
    @assert h1.grid == h2.grid
    VegasHist(
        h1.grid,
        h1.normsums .+ h2.normsums,
        h1.counts .+ h2.counts,
        Ref(h1.sum[] + h2.sum[]),
        Ref(h1.sum2[] + h2.sum2[]),
    )
end

function fuseall(hists::AbstractVector{<:VegasHist})::VegasHist
    reduce(fuse, hists)
end

function fill_vegas_hist!(f, h::VegasHist, rng::AbstractRNG; neval)
    iq = h.grid
    sum = h.sum[]
    sum2 = h.sum2[]
    for _ in 1:neval
        s = draw_x_index_cell(rng, iq)
        cell = s.cell
        fx = f(s.position)
        y  = fx * volume(cell) * length(iq)
        sum  += y
        sum2 += y.^2
        for axis in eachindex(iq.boundaries)
            wt = (cell.upper[axis] - cell.lower[axis]) * size(iq)[axis]
            ibin = s.index[axis]
            h.counts[axis][ibin]   += 1
            h.normsums[axis][ibin] += norm(fx)*wt
        end
    end
    h.sum[] = sum
    h.sum2[] = sum2
    h
end

function axisinds(grid::VegasGrid{N}) where {N}
    ntuple(identity, Val{N}())
end

function estimate_integral(h::VegasHist)
    count = sum(h.counts[1])::Int64
    mean, var = mean_var(sum=h.sum[],
                         sum2=h.sum2[],
                         count=count)
    (value = mean, var=var, neval=count)
end

function update_vegasvr(vr::VegasVR, h::VegasHist, alg::Vegas)
    axes = axisinds(h.grid)
    boundaries = map(axes) do axis
        cdf = estimate_cdf(h, axis, alg)
        nq = length(cdf.positions)
        quantiles(cdf, nq)
    end
    grid = VegasGrid(boundaries)
    val, var, neval = estimate_integral(h)
    values = push!(copy(vr.values), val)
    vars   = push!(copy(vr.vars  ), var)
    VegasVR(grid, values, vars, vr.neval + neval)
end

function estimate_pdf(h::VegasHist, axis::Int, ::NoDamping)
    if any(iszero, h.counts[axis])
        # TODO: we could guarantee, that there are evals in each bin
        msg = """Empty bin in vegas histogram.
        gridsize = $(size(h.grid))
        axis = $axis
        counts[$axis] = $(h.counts[axis])
        neval = $(sum(h.counts[1]))
        Consider increasing `neval` or decreasing `nbins`.
        """
        throw(ArgumentError(msg))
    end
    pdf = map(/, h.normsums[axis], h.counts[axis])
    s = sum(pdf)
    if iszero(s)
        isfinite = false
    else
        pdf ./= s
        isfinite = true
    end
    pdf, isfinite
end

function estimate_pdf(h::VegasHist, axis::Int, r::LepageDamping)
    pdf, isfinite = estimate_pdf(h, axis, NoDamping())
    isfinite || return (pdf, false)
    updf = map(pdf) do p
        ((p - 1) / log(p))^r.alpha
    end
    (normalize!(updf, 1), true)
end

function estimate_cdf(h, axis::Int, alg::Vegas)
    pdf, isfinite = estimate_pdf(h, axis, alg.regularization)
    cdf_vals = cumsum(pdf)
    z = zero(eltype(cdf_vals))
    pushfirst!(cdf_vals, z)
    walls = h.grid.boundaries[axis]
    CDF(walls, cdf_vals, isfinite)
end

function tuneonce(f, vr::VegasVR, alg::Vegas=Vegas())
    h = create_vegas_hist(f, vr, alg)
    update_vegasvr(vr, h, alg)
end

function default_grid_size(dom::Domain{N}, nbins) where {N}
    ntuple(_ -> nbins, Val(N))
end

function equidistant_grid(dom::Domain, size)

    bdries = map(Tuple(dom.lower), Tuple(dom.upper), size) do lo, hi, nbins
        nwalls = nbins + 1
        r = range(lo, stop=hi, length=nwalls)
        collect(r)
    end
    VegasGrid(bdries)
end

function initvr(f, dom::Domain, alg::Vegas)
    size = default_grid_size(dom, alg.nbins)
    iq = equidistant_grid(dom, size)
    s = draw_x_index_cell(alg.rng, iq)
    y = (f(s.position).^2) ./ 2
    T = typeof(y)
    neval = 0
    VegasVR(iq, T[], T[], neval)
end

function tune(f, vr::VegasVR, alg::Vegas)
    for i in 1:alg.niter
        vr = tuneonce(f, vr, alg)
    end
    vr
end

function tune(f, dom::Domain, alg)
    vr = initvr(f, dom, alg)
    tune(f, vr, alg)
end

# function div_pos(x, y)
#     ret = x / y
#     if iszero(y) & (!iszero(x))
#         typeof(ret)(Inf)
#     else
#         ret
#     end
# end
function combine_scalar(val1, var1, val2, var2)
    p1 = 1/var1
    p2 = 1/var2
    wt1  = p1 / (p1 + p2)
    wt2 = p2 / (p1 + p2)
    val = wt1*val1 + wt2*val2
    var = (wt1^2)*var1 + (wt2^2)*var2
    T = typeof((val, var))
    ret = if iszero(var1) & iszero(var2)
        @assert val1 ≈ val2
        (val1, var1)
    elseif iszero(var1)
        (val1, var1)
    elseif iszero(var2)
        (val2, var2)
    else
        (val, var)
    end
    T(ret)
end

function combine_vec(val1, var1, val2, var2)
    pairs = combine_scalar.(val1, var1, val2, var2)
    first.(pairs), last.(pairs)
end

function combine(nt1, nt2) # named tuple val var
    val1, var1 = nt1
    val2, var2 = nt2
    
    value, var = if nt1.value isa AbstractVector
        combine_vec(val1,var1,val2,var2)
    else
        combine_scalar(val1,var1,val2,var2)
    end
    (value=value, var=var)
end

function finish_integral(vr::VegasVR, alg::Vegas)
    index = (alg.ndrop+1):length(vr.values)
    values = vr.values[index]
    vars = vr.vars[index]
    precs = map(vars) do var
        1 ./ var
    end
    var = 1 ./ sum(precs)
    std = sqrt.(var)
    weighted_values = map(precs, values) do p, val
        var .* p .* val
    end
    value = sum(weighted_values)
    pairs = map(values, vars) do val, var
        (value=val, var=var)
    end
    value, var = reduce(combine, pairs)
    (value=value, std=sqrt.(var), neval=vr.neval)
end

function integral_kernel(f, dom::Domain, alg::Vegas)
    vr = tune(f, dom, alg)
    finish_integral(vr, alg)
end

function integral_kernel(f, dom::VegasGrid, alg::MCVanilla)
    mc_kernel(f, alg.rng, dom, neval=alg.neval)
end

function canonicalize(f, dom::VegasGrid, alg::MCVanilla)
    f, dom, alg
end
