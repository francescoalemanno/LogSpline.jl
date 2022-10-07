using LogSpline, Test

@testset "Vector Equation" begin
    x = randn(1000)
    knots = LinRange(extrema(x)..., 7)
    fn = fit_logspline(x,knots)
    lx = LinRange(extrema(x)..., 100)
    w = fn.(lx)
    w ./= sum(w)
    m1 = sum(w.*lx)
    m2 = sum(w.*lx.^2)
    @show m1, m2
end
