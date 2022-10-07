using LogSpline, Test, Random

@testset "basic test LogSpline64" begin
    x = randn(Xoshiro(1337), 5000)
    range = extrema(x)
    kn = LinRange(range..., 12)
    fn = fit_logspline(x, kn, order = 3)
    @test fn.converged
    lx = LinRange(range..., 100)
    w = fn.(lx)
    w ./= sum(w)
    m1 = sum(w .* lx)
    m2 = sqrt(sum(w .* lx .^ 2))
    @test abs(m1) < 0.05
    @test abs(m2 - 1) < 0.05
    #@show abs(m1), abs(m2 - 1)
end

@testset "basic test LogSpline32" begin
    x = randn(Xoshiro(1337), Float32, 5000)
    range = extrema(x)
    kn = LinRange(range..., 12)
    fn = fit_logspline(x, kn, order = 3)
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
    kn = LinRange(-2, 2, 30)
    fn = fit_spline(x, y, kn, order = 3)
    lx = LinRange(-2, 2, 500)
    ly = @. abs(sin(lx) - fn(lx))
    @test maximum(ly) < 1e-6
end

@testset "basic test Spline32" begin
    x = LinRange(-2.0f0, 2.0f0, 100)
    y = @. sin(x)
    kn = LinRange(-2.0f0, 2.0f0, 30)
    fn = fit_spline(x, y, kn, order = 3)
    lx = LinRange(-2.0f0, 2.0f0, 500)
    ly = @. abs(sin(lx) - fn(lx))
    @test maximum(ly) < 1.0f-6
end

@testset "basic test Spline64 auto-knots" begin
    g(x) = sin(x * x)
    x = LinRange(-2, 2, 100)
    y = @. g(x)
    lx = LinRange(-2, 2, 1000)
    for o = 1:3, n = 10:20
        kn = LinRange(-2, 2, n)
        kn2 = knots_spline(x, y, n, order = o)
        fn = fit_spline(x, y, kn, order = o)
        fn2 = fit_spline(x, y, kn2, order = o)
        ly = @. abs(g(lx) - fn(lx))
        ly2 = @. abs(g(lx) - fn2(lx))
        @test maximum(ly2) < maximum(ly) < 0.1
    end
end

@testset "basic test LogSpline64 - auto-knots" begin
    x = randn(Xoshiro(1337), 15000)
    range = extrema(x)
    kn = knots_logspline(x, 15, order = 3)
    fn = fit_logspline(x, kn, order = 3)
    @test fn.converged
    lx = LinRange(range..., 100)
    w = fn.(lx)
    w ./= sum(w)
    m1 = sum(w .* lx)
    m2 = sqrt(sum(w .* lx .^ 2))
    @test abs(m1) < 0.05
    @test abs(m2 - 1) < 0.05
    #@show abs(m1), abs(m2 - 1)
end
