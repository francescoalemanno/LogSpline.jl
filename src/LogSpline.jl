"""
    LogSpline.jl
"""
module LogSpline

using LinearAlgebra: Diagonal, qr, dot, ColumnNorm

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
    verbose = false,
) where {T<:AbstractFloat}
    n_z = zero(T)
    N = length(s)
    K = length(xi) + order - 1
    C = fill(n_z, K)
    back_C = identity.(C)
    J = fill(n_z, K, K)
    D = fill(n_z, K)
    aBk = fill(n_z, K)
    max_range = extrema((extrema(s)..., extrema(xi)...))
    int_x = LinRange(max_range..., 5 * K + 1)
    Ni = length(int_x)
    BKi = fill(n_z, Ni, K)
    for i = 1:N
        aBk .+= cox_deboor(s[i], xi, order) ./ N
    end
    for i = 1:Ni
        BKi[i, :] .= cox_deboor(int_x[i], xi, order)
    end

    oerr = T(Inf)
    trust_region = T(sqrt(K) / 2)
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
                w = exp(lw)
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
                w = exp(lw)
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
        if norm_deltas > trust_region
            deltas .*= trust_region / norm_deltas
        end

        back_C .= C
        @. C += deltas

        if err < oerr && isfinite(err) && (!isnan(err))
            oerr = err
            trust_region = min(trust_region * 1.05, 2 * sqrt(K))
            verbose && println("$iters, $err, $norm_deltas / $trust_region")
        else
            trust_region /= 2
            verbose && println("reducing trust_region size -> $trust_region")
        end

        if err < abstol || trust_region < T(0.001)
            break
        end
    end

    Z = n_z
    for v in int_x
        res = dot(cox_deboor(v, xi, order), C)
        Z += exp(res) * (int_x[2] - int_x[1])
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
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
    xi::AbstractVector{Float64};
    order::Int = 3,
)
    issorted(xi) || error("Knots \"xi\" must be sorted.")
    N = length(x)
    K = length(xi) + order - 1
    M = zeros(N, K)
    scal = 1 / sqrt(N)
    for i = 1:N
        M[i, :] .= cox_deboor(x[i], xi, order) / scal
    end
    C = qr(M, ColumnNorm()) \ (y ./ scal)
    SplineFn(collect(C), collect(xi), order)
end

export cox_deboor, fit_logspline, fit_spline
end
