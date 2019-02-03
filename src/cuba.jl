import Random

export CubaAlg

struct CubaAlg{A,K}
    algorithm::A
    kw::K
end

function CubaAlg(a; kw...)
    CubaAlg(a, kw)
end

struct CubaFun{T,F,D<:Domain}
    inner::F
    domain::D
    ndim::Int
    ncomp::Int
end

function CubaFun(f, dom::Domain)
    inner = f
    ndim   = ndims(dom)
    rng = Random.GLOBAL_RNG
    x = draw(rng, dom).position
    y = f(x)
    ncomp  = length(y)
    T = typeof(float.(y))
    F = typeof(f)
    D = typeof(dom)
    CubaFun{T,F,D}(inner, dom, ndim, ncomp)
end

function (f::CubaFun)(pt0, out)
    dom = f.domain
    vol = volume(dom)
    lo = dom.lower
    up = dom.upper
    pt = lo .+ pt0 .* (up .- lo)
    y = vol * f.inner(pt) 
    copyto!(out, y)
end

function from_vector(::Type{T}, v) where {T <: Number}
    @assert length(v) == 1
    T(first(v))
end

function from_vector(::Type{T}, v) where {T}
    T(v)
end

canonicalize(f, dom::NTuple{2,Number}, alg::CubaAlg) = canonicalize_cuba(f∘first, dom, alg)
canonicalize(f, dom, alg::CubaAlg) = canonicalize_cuba(f, dom, alg)

function canonicalize_cuba(f, dom, alg::CubaAlg)
    dom2 = Domain(dom)
    CubaFun(f, dom2), dom2, alg
end

@noinline function integral_kernel(f::CubaFun{T}, dom::Domain, cuba::CubaAlg) where {T}
    raw = cuba.algorithm(f, f.ndim, f.ncomp; cuba.kw...)
    
    (value=from_vector(T, raw.integral), 
        std=from_vector(T, raw.error), 
        neval=raw.neval, raw=raw)
end
