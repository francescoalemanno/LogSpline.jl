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
    @test abs(m1) < 0.02
    @test abs(m2 - 1) < 0.02
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
    @test abs(m1) < 0.02
    @test abs(m2 - 1) < 0.02
    #@show abs(m1), abs(m2 - 1)
end

@testset "basic test Spline64" begin
    x = LinRange(-2, 2, 100)
    y = @. sin(x)
    kn = LinRange(-2, 2, 30)
    fn = fit_spline(x, y, kn, order = 3)
    lx = LinRange(-2, 2, 500)
    ly = @. abs(sin(lx) - fn(lx))
    @test maximum(ly) < 6e-7
    #@show maximum(ly)
end

@testset "basic test Spline32" begin
    x = LinRange(-2.0f0, 2.0f0, 100)
    y = @. sin(x)
    kn = LinRange(-2.0f0, 2.0f0, 30)
    fn = fit_spline(x, y, kn, order = 3)
    lx = LinRange(-2.0f0, 2.0f0, 500)
    ly = @. abs(sin(lx) - fn(lx))
    @test maximum(ly) < 1.0f-6
    #@show maximum(ly)
end

@testset "basic test Spline64 auto-knots" begin
    g(x) = sin(x * x)
    x = LinRange(-2, 2, 100)
    y = @. g(x)
    lx = LinRange(-2, 2, 1000)
    m, q = (-0.06903451725862919, -2.218609304652326)
    cnt = 0
    for o = 1:3, n = 10:20
        Eb = exp(m * cnt .+ q)
        kn = LinRange(-2, 2, n)
        kn2 = knots_spline(x, y, n, order = o)
        fn = fit_spline(x, y, kn, order = o)
        fn2 = fit_spline(x, y, kn2, order = o)
        ly = @. abs(g(lx) - fn(lx))
        ly2 = @. abs(g(lx) - fn2(lx))
        mm = maximum(ly)
        mm2 = maximum(ly2)
        @test mm2 < mm < Eb
        cnt += 1
        #println("A[$cnt] = $(mm)")
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
    @test abs(m1) < 0.005
    @test abs(m2 - 1) < 0.005
    #@show abs(m1), abs(m2 - 1)
end

#=
function optimize()
    A=zeros(33)
    A[1] = 0.09739501658632077
    A[2] = 0.08380260552378493
    A[3] = 0.07873921196896749
    A[4] = 0.07447509118923823
    A[5] = 0.06693746950746982
    A[6] = 0.05891708723389533
    A[7] = 0.0522294300851911
    A[8] = 0.046800841347920175
    A[9] = 0.03922313602542682
    A[10] = 0.032363836591737694
    A[11] = 0.030692691842828745
    A[12] = 0.05075782807498297
    A[13] = 0.03546560889020334
    A[14] = 0.025197933276088547
    A[15] = 0.018250500661489477
    A[16] = 0.013442883390454297
    A[17] = 0.010086961451125975
    A[18] = 0.0076831486003153815
    A[19] = 0.0063348220712095005
    A[20] = 0.00532251269532108
    A[21] = 0.004535042159006852
    A[22] = 0.003902544868103819
    A[23] = 0.006579386398392284
    A[24] = 0.005312138061195015
    A[25] = 0.0041785904749035985
    A[26] = 0.0032218111375995617
    A[27] = 0.0025002596044230163
    A[28] = 0.0019289696071560503
    A[29] = 0.0014909703020589138
    A[30] = 0.0011581544219226303
    A[31] = 0.0009057901168887939
    A[32] = 0.0007134949225064413
    A[33] = 0.0005659585869829398


    cost = Inf
    bm,bq=(Inf,Inf)

    for m in LinRange(-2.0,2.0,2000)
        for q in LinRange(-5,5,2000)
            S = exp.(m .* (eachindex(A).-1) .+ q)
            quant = .-(A .- S)
            qual, score = (mean(quant.>0), maximum(quant))
            if qual==1 && score<cost
                cost = score
                bm=m
                bq = q
            end
        end
    end
    println("m,q = (",bm, ", ", bq, ") ")
end

optimize()
=#
