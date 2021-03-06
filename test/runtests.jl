using MCIntegrals
const P = MCIntegrals
using Test
using StaticArrays
using LinearAlgebra
using Random
using Setfield
using Cuba: vegas
using HCubature

function isconsistent(truth, est; nstd=6, kw_approx...)
    val = est.value
    Δ = nstd * est.std
    if ≈(val, truth; kw_approx...)
        true
    else
        truth - Δ <= val <= truth + Δ
    end
end


ALGS = [Vegas(), MCVanilla()]

@testset "RNG Reproducibility" begin
    for alg in ALGS
        @set! alg.rng = MersenneTwister(1)
        res1 = ∫(cos, (0,1), alg)
        @set! alg.rng = MersenneTwister(1)
        res2 = ∫(cos, (0,1), alg)
        @test res1 === res2
        res3 = ∫(cos, (0,1), alg)
        @test !(res3 === res2)
    end
end

@testset "Domain" begin
    dom = Domain((1f0, 2f0))
    @test P.pointtype(typeof(dom)) === SVector{1, Float32}
    @test P.volume(dom) === 1f0
end

@testset "exotic types $(typeof(alg))" for alg in [
        MCVanilla(10),
        Vegas(10), 
        CubaAlg(vegas),
       ]
    est = ∫(identity, (0f0, 1f0), alg)
    @test typeof(est.value) === Float32
    @test typeof(est.std) === Float32

    est = ∫(_ -> true, (0f0, 1f0), alg)
    @test typeof(est.value) === Float64
    @test typeof(est.std) === Float64

    est = ∫(identity, (0, 1), alg)
    @test typeof(est.value) === Float64
    @test typeof(est.std) === Float64

    est = ∫(identity, (big"0", big"1"), alg)
    @test typeof(est.value) === BigFloat
    @test typeof(est.std) === BigFloat

    v = @SVector[1.,2.]
    est = ∫(_ -> v, (0,1), alg)
    @test typeof(est.value) === typeof(v)
    @test typeof(est.std)   === typeof(est.value)
end

@testset "std bernoulli" begin
    p = 0.1 + 0.8*rand()
    f = x -> 0 <= x <= p
    
    neval = 1000
    true_value = p
    true_std = sqrt(p*(1-p)/neval)
    
    est = ∫(f, (0,1), MCVanilla(neval))
    @test est.std ≈ true_std rtol=0.2
    @test isconsistent(true_value, est)
    
    est = ∫(f, (0,1), Vegas(neval))
    @test est.std < 0.2*true_std
    @test isconsistent(true_value, est)
end

@testset "constant $(typeof(alg))" for alg in [
    Vegas(10^1),
    MCVanilla(10^1),
    CubaAlg(vegas),
    HCubatureAlg(),
    ]
    for dim in 1:4
        for _ in 1:10
            lower = @SVector randn(dim)
            upper = lower + @SVector rand(dim)
            val = randn()
            f = _ -> val
            dom = Domain(lower, upper)
            est = ∫(f, dom, alg)
            vol = prod(upper .- lower)
            truth = val * vol
            @test isconsistent(truth, est)
            @test est.std < 1e-3
        end
    end
end

@testset "∫x^2" begin
    est_vegas = ∫(x->x^2, (0,1), Vegas(10000))
    @test isconsistent(1/3, est_vegas)
    @test est_vegas.std < 1e-3

    est_vanilla = ∫(x->x^2, (0,1), MCVanilla(1000))
    @test isconsistent(1/3, est_vanilla)
    @test est_vanilla.std < 1e-2
end

@testset "∫(f, (a,b), $(typeof(alg)))" for alg in [
        MCVanilla(10^4),
        Vegas(10^4),
       ]

    funs = [
        (f = x -> x^2, F = x -> x^3/3),
        (f = cos, F = sin),
        let
            a = randn()
            (f = _ -> a, F = x -> a*x)
        end
    ]

    for (f,F) in funs
        for _ in 1:10
            a = 10randn()
            b = a + 10rand()
            est = ∫(f, (a,b), alg)
            truth = F(b) - F(a)
            @test isconsistent(truth, est)
        end
    end
end

@testset "Ball $(typeof(alg))" for alg in [
    MCVanilla(10^4),
    Vegas(10^4, rng=MersenneTwister(1))]

    f(x) = norm(x) < 1 ? 1. : 0.
    est = ∫(f, (-2,2), alg)
    @test isconsistent(2,est)

    est = ∫(f, ((-1,1),(-1,1)), alg)
    @test isconsistent(π, est)

    est = ∫(f, ((-1,1),(-1,1),(-1,1)), alg)
    @test isconsistent(4π/3, est)
end


@testset "Fubini $(typeof(alg))" for alg in [
    MCVanilla(10^4),
    Vegas(10^4)]
    for _ in 1:10
        a1,a2 = randn(2)
        b1 = a1 + 10rand()
        b2 = a2 + 10rand()
        dom1 = Domain((a1, b1))
        dom2 = Domain((a2, b2))
        dom = Domain(((a1, b1), (a2, b2)))
        f1 = x -> cos(x[1])
        F1 = x -> sin(x[1])
        f2 = x -> x[1]^2
        F2 = x -> x[1]^3/3
        f  = v -> f1(v[1]) * f2(v[2])
        est1 = ∫(f1, dom1, alg)
        est2 = ∫(f2, dom2, alg)
        est  = ∫(f , dom , alg)
        @test isconsistent(est1.value*est2.value, est)
    end
end

@testset "Simple Vegas Example" begin
    a = -0.237267860990952
    b = -0.19099487884978464
    val = 0.665843778891144
    truth = (b - a) * val
    f = _ -> val
    Ω = Domain((a,b))
    
    alg = Vegas(5)
    est = ∫(f, Ω, alg)
    @test est.value ≈ truth
    @test 0 <= est.std <= sqrt(eps(Float64))
    
    iq = P.initvr(f, Ω, alg)
    est = ∫(f, iq, alg)
    @test est.value ≈ truth
    @test 0 <= est.std <= sqrt(eps(Float64))
    
    iq2 = P.tune(f, Ω, alg)
    est = ∫(f, iq2, alg)
    @test est.value ≈ truth
    @test 0 <= est.std <= sqrt(eps(Float64))
    walls = iq2.boundaries[1]
    @test walls ≈ range(a,stop=b, length=length(walls))
end

