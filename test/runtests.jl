using MCIntegrals
using Test
using StaticArrays
using LinearAlgebra

function isconsistent(truth, est; nstd=6, kw_approx...)
    val = est.value
    Δ = nstd * est.std
    if ≈(val, truth; kw_approx...)
        true
    else
        truth - Δ <= val <= truth + Δ
    end
end

@testset "constant $(alg)" for alg in [
    Vegas(10^2),
    MCVanilla(10^2),
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
            @test est.std < 1e-2
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

@testset "∫(f, (a,b), $alg)" for alg in [
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

@testset "Ball $alg" for alg in [
    MCVanilla(10^4),
    Vegas(10^4)]

    f(x) = norm(x) < 1 ? 1. : 0.
    est = ∫(f, (-2,2), alg)
    @test isconsistent(2,est)

    est = ∫(f, ((-1,1),(-1,1)), alg)
    @test isconsistent(π, est)

    est = ∫(f, ((-1,1),(-1,1),(-1,1)), alg)
    @test isconsistent(4π/3, est)
end


@testset "Fubini $alg" for alg in [
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
