using MCIntegrals
using Test

function isconsistent(truth, est; nstd=6, kw_approx...)
    val = est.value
    Δ = nstd * est.std
    if ≈(val, truth; kw_approx...)
        true
    else
        truth - Δ <= val <= truth + Δ
    end
end

@testset "integral $alg" for alg in [
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
            est = integral(f, (a,b), alg)
            truth = F(b) - F(a)
            @test isconsistent(truth, est)
        end
    end
end
