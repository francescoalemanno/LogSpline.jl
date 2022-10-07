"""
    LogSpline.jl
"""
module LogSpline

function find_inside(t,x)
    N=length(t)
    #perform quadratic search
    q = 1
    while (q+1)^2<length(t) && t[(q+1)^2] < x
        q+=1
    end
    #refine with final linear search
    for i in q^2:N-1
        if t[i]<=x<t[i+1]
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

function fit_logspline(
    s::AbstractVector{Float64},
    xi::AbstractVector{Float64};
    maxiter::Int64 = 200,
    abstol::Float64 = 1e-16,
    order::Int64 = 3,
    preconditioning::Bool = true,
    pivoting::Bool = true,
)
    N = length(s)
    K = length(xi) + order - 1
    C = zeros(K)
    back_C = identity.(C)
    J = zeros(K, K)
    D = zeros(K)
    aBk = zeros(K)
    max_range = extrema((extrema(s)..., extrema(xi)...))
    int_x = LinRange(max_range..., 5 * K + 1)
    Ni = length(int_x)
    BKi = zeros(Ni, K)
    for i = 1:N
        aBk .+= cox_deboor(s[i], xi, order) ./ N
    end
    for i = 1:Ni
        BKi[i, :] .= cox_deboor(int_x[i], xi, order)
    end

    oerr = Inf
    trust_region = 0.5 * sqrt(K)
    for iters = 1:maxiter
        err = 0.0
        J .= 0.0
        D .= 0.0
        for p = 1:K
            tbw = 0.0
            tw = 0.0
            for i = 1:Ni
                lw = 0.0
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
            tbw = 0.0
            tw = 0.0
            for i = 1:Ni
                lw = 0.0
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

        err = sqrt(D'D / K)
        if preconditioning
            jacobi = Diagonal(J)
            J .= jacobi * J
            D .= jacobi * D
        end

        deltas = (
            if pivoting
                qr(J, Val(true)) \ D
            else
                J \ D
            end
        )

        norm_deltas = sqrt(sum(abs2, deltas))
        if norm_deltas > trust_region
            deltas .*= trust_region / norm_deltas
        end

        back_C .= C
        @. C += deltas

        if err < oerr && isfinite(err) && (!isnan(err))
            oerr = err
            trust_region = min(trust_region * 1.05, 2 * sqrt(K))
            println("$iters, $err, $norm_deltas / $trust_region")
        else
            trust_region /= 2
            println("reducing trust_region size -> $trust_region")
        end

        if err < abstol || trust_region < 0.001
            break
        end
    end

    Z = 0.0
    for v in int_x
        res = dot(cox_deboor(v, xi, order), C)
        Z += exp(res) * (int_x[2] - int_x[1])
    end

    logZ = log(Z)
    function pdf(v)
        exp(-logZ + dot(cox_deboor(v, xi, order), C))
    end, C
end



function fit_spline(
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
    xi::AbstractVector{Float64};
    order::Int64 = 3,
)
    issorted(xi) || error("Knots \"xi\" must be sorted.")
    N = length(x)
    K = length(xi) + order - 1
    M = zeros(N, K)
    scal = 1 / sqrt(N)
    for i = 1:N
        M[i, :] .= cox_deboor(x[i], xi, order) / scal
    end
    C = qr(M, Val(true)) \ (y ./ scal)
    function fn(x)
        return cox_deboor(x, xi, order)'C
    end
end

export cox_deboor, fit_logspline, fit_spline
end
