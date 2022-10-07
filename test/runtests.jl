using LogSpline, Test, Random

@testset "basic test LogSpline64" begin
    x = randn(Xoshiro(1337), 5000)
    range = extrema(x)
    knots = LinRange(range..., 12)
    fn = fit_logspline(x, knots, order = 3)
    @test fn.converged
    lx = LinRange(range..., 100)
    w = fn.(lx)
    w ./= sum(w)
    m1 = sum(w .* lx)
    m2 = sqrt(sum(w .* lx .^ 2))
    @test abs(m1) < 0.05
    @test abs(m2 - 1) < 0.05
end

@testset "basic test LogSpline32" begin
    x = randn(Xoshiro(1337), Float32, 5000)
    range = extrema(x)
    knots = LinRange(range..., 12)
    fn = fit_logspline(x, knots, order = 3)
    @test !fn.converged
    lx = LinRange(range..., 100)
    w = fn.(lx)
    w ./= sum(w)
    m1 = sum(w .* lx)
    m2 = sqrt(sum(w .* lx .^ 2))
    @test abs(m1) < 0.05
    @test abs(m2 - 1) < 0.05
end

@testset "basic test Spline64" begin
    x = LinRange(-2, 2, 100)
    y = @. sin(x)
    knots = LinRange(-2, 2, 30)
    fn = fit_spline(x, y, knots, order = 3)
    lx = LinRange(-2, 2, 500)
    ly = @. abs(sin(lx) - fn(lx))
    @test maximum(ly) < 1e-6
end
