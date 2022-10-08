"""LogSpline.jl

    main procedures: fit_logspline, knots_logspline, fit_spline, knots_spline

    extras: cox_deboor

    internal: find_inside, diff_fn
"""
module LogSpline
using LinearAlgebra: Diagonal, qr, dot, ColumnNorm
using Statistics: quantile

function find_inside(t, x)
    N = length(t)
    #perform quadratic search
    q = 1
    while (q + 1)^2 < length(t) && t[(q+1)^2] < x
        q += 1
    end
    #refine with final linear search
    for i = q^2:N-1
        if t[i] <= x < t[i+1]
            return i
        end
    end
    return -1
end

function mpr_points(xi, order)
    K = length(xi) + order - 1
    Ps = collect(xi)
    while true
        append!(Ps, (Ps[1:end-1] .+ Ps[2:end]) ./ 2)
        sort!(Ps)
        if length(Ps) >= K * (2 * order + 1)
            break
        end
    end
    Ps[1] = 2 * Ps[1] - Ps[2]
    Ps[end] = 2 * Ps[end] - Ps[end-1]
    w = diff(Ps)
    m = (Ps[1:end-1] .+ Ps[2:end]) ./ 2
    m, w
end

function cox_deboor(x, t, order)
    K = order + 1
    Bs = zeros(length(t) + K - 2)
    dr = zeros(K - 1)
    dl = zeros(K - 1)
    i = find_inside(t, x)
    if i < 0
        if x >= t[end]
            Bs[end] = exp(t[end] - x)
        end
        if x <= t[1]
            Bs[1] = exp(x - t[1])
        end
        return Bs
    end
    Bs[i] = one(x)
    for j = 1:K-1
        cr = clamp(i + j, 1, length(t))
        cl = clamp(i + 1 - j, 1, length(t))
        dr[j] = t[cr] - x
        dl[j] = x - t[cl]
        saved = zero(x)
        for r = 1:j
            term = Bs[r+i-1] / (dr[r] + dl[j+1-r])
            Bs[r+i-1] = saved + dr[r] * term
            saved = dl[j+1-r] * term
        end
        Bs[j+i] = saved
    end
    Bs
end

struct LogSplineFn{T<:AbstractFloat}
    logZ::T
    C::Vector{T}
    xi::Vector{T}
    order::Int
    converged::Bool
    err::T
end

function (ls::LogSplineFn)(v)
    exp(-ls.logZ + dot(cox_deboor(v, ls.xi, ls.order), ls.C))
end

function fit_logspline(
    s::AbstractVector{T},
    xi::AbstractVector{T};
    maxiter::Int = 200,
    abstol::T = eps(T),
    order::Int = 3,
    preconditioning::Bool = true,
    pivoting::Bool = true,
    verbose::Bool = false,
    trust_region::T = zero(T),
) where {T<:AbstractFloat}
    issorted(xi) || error("Knots \"xi\" must be sorted.")
    n_z = zero(T)
    N = length(s)
    K = length(xi) + order - 1
    C = fill(n_z, K)
    back_C = identity.(C)
    J = fill(n_z, K, K)
    D = fill(n_z, K)
    aBk = fill(n_z, K)
    int_x, w_x = mpr_points(xi, order)
    Ni = length(int_x)
    BKi = fill(n_z, Ni, K)
    for i = 1:N
        aBk .+= cox_deboor(s[i], xi, order) ./ N
    end
    for i = 1:Ni
        BKi[i, :] .= cox_deboor(int_x[i], xi, order)
    end
    low_trust_region = T(sqrt(K) / 200)
    if trust_region <= 0
        trust_region = T(2 * sqrt(K))
    end
    perr = T(Inf)
    oerr = T(Inf)

    for iters = 1:maxiter
        err = n_z
        J .= n_z
        D .= n_z
        for p = 1:K
            tbw = n_z
            tw = n_z
            for i = 1:Ni
                lw = n_z
                for k = 1:K
                    lw += C[k] * BKi[i, k]
                end
                w = exp(lw) * w_x[i]
                bw = w * BKi[i, p]
                tbw += bw
                tw += w
            end
            D[p] = tbw / tw
        end

        for p = 1:K, q = p:K
            tbw = n_z
            tw = n_z
            for i = 1:Ni
                lw = n_z
                for k = 1:K
                    lw += C[k] * BKi[i, k]
                end
                w = exp(lw) * w_x[i]
                bw = w * (BKi[i, p] - D[p]) * (BKi[i, q] - D[q])
                tbw += bw
                tw += w
            end
            J[p, q] = tbw / tw
            p != q && (J[q, p] = J[p, q])
        end

        for p = 1:K
            D[p] = aBk[p] - D[p]
        end

        err = T(sqrt(dot(D, D) / K))
        if preconditioning
            jacobi = Diagonal(J)
            J .= jacobi * J
            D .= jacobi * D
        end

        deltas = (
            if pivoting
                qr(J, ColumnNorm()) \ D
            else
                J \ D
            end
        )

        norm_deltas = T(sqrt(sum(abs2, deltas)))
        reduced = false
        if norm_deltas > trust_region
            reduced = true
            deltas .*= trust_region / norm_deltas
        end

        back_C .= C
        @. C += deltas

        if (err < oerr < perr) && (reduced)
            trust_region *= T(1.05)
        end

        if err < oerr && isfinite(err) && (!isnan(err))
            perr = oerr
            oerr = err
            verbose && println("$iters, $err, $norm_deltas / $trust_region")
        else
            trust_region /= 2
            verbose && println("reducing trust_region size -> $trust_region")
        end

        if err < abstol || trust_region < low_trust_region
            break
        end
    end

    Z = n_z
    for (i, w) in enumerate(w_x)
        res = dot(BKi[i, :], C)
        Z += w * exp(res)
    end

    LogSplineFn(T(log(Z)), collect(C), collect(xi), order, oerr < abstol, oerr)
end

struct SplineFn{T<:AbstractFloat}
    C::Vector{T}
    xi::Vector{T}
    order::Int
end

function (sp::SplineFn)(x)
    dot(cox_deboor(x, sp.xi, sp.order), sp.C)
end

function fit_spline(
    x::AbstractVector{T},
    y::AbstractVector{T},
    xi::AbstractVector{T};
    order::Int = 3,
) where {T<:AbstractFloat}
    issorted(xi) || error("Knots \"xi\" must be sorted.")
    N = length(x)
    K = length(xi) + order - 1
    M = zeros(T, N, K)
    scal = T(1 / sqrt(N))
    for i = 1:N
        M[i, :] .= cox_deboor(x[i], xi, order) / scal
    end
    C = qr(M, ColumnNorm()) \ (y ./ scal)
    SplineFn(collect(C), collect(xi), order)
end

function diff_fn(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    xm = (x[1:end-1] .+ x[2:end]) ./ 2
    w = diff(x)
    yd = diff(y) ./ w
    xm, yd, w
end

function knots_spline(
    x::AbstractVector{T},
    y::AbstractVector{T},
    N::Int;
    order::Int,
) where {T<:AbstractFloat}
    order >= 0 || error("Order must be non-negative.")
    N > 0 || error("N must be positive.")
    w = x
    s = sortperm(x)
    mx, dy = (x[s], y[s])
    for _ = 0:order
        mx, dy, w = diff_fn(mx, dy)
    end
    Dkyik = abs.(dy) .^ (1 / (1 + order))
    I_dkyik = cumsum(Dkyik .* w)
    spacing = LinRange(extrema(I_dkyik)..., N)
    kn = zeros(N+2)
    for (j, p) in enumerate(spacing)
        i = find_inside(I_dkyik, p)
        if i == -1
            # remember to include endpoint
            i = length(I_dkyik) - 1
        end
        kn[j+1] = (p - I_dkyik[i]) / (I_dkyik[i+1] - I_dkyik[i]) * (mx[i+1] - mx[i]) + mx[i]
    end
    kn[1] = minimum(x)
    kn[end] = maximum(x)
    sort(unique(kn))
end

function knots_logspline(
    sample::AbstractVector{T},
    N::Int;
    order::Int,
) where {T<:AbstractFloat}
    sN = 0
    while (2^order * sN)^2 < length(sample) || sN < N + order + 2
        sN += 1
    end
    q = LinRange(0, 1, sN)
    x = quantile(sample, q)
    xm = (x[1:end-1] .+ x[2:end]) ./ 2
    y = log.(diff(q) ./ diff(x))
    kn = knots_spline(xm, y, N; order = order)
    kn[1] = minimum(sample)
    kn[end] = maximum(sample)
    return kn
end

export cox_deboor, fit_logspline, fit_spline, knots_spline, knots_logspline
end
