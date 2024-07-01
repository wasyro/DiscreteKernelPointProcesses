using Statistics
using Distributions

include("submodular_minimizer.jl")

function sigmoid(x)
    return 1 / (1 + exp(-x))
end

"""
    Restrict f:2ᵛ→R to fˢ:2ˢ→R
"""
function restriction(f :: Function, S :: Vector{Int}, V :: Vector{Int})
    return A -> f(intersect(V, S)[A])
end

"""
    Contract f:2ᵛ→R to fₛ:2ᵛ⁻ˢ→R
"""
function contraction(f :: Function, S :: Vector{Int}, V :: Vector{Int})
    return A -> f(union(setdiff(V, S)[A], S)) - f(S)
end


"""
    Get grow supergradient ŝ of a submodular function f at A ⊂ V
"""
function grow_supergrad(f :: Function, A :: Vector{Int}, V :: Vector{Int})
    N = length(V)
    fᵥ = f(V)
    fₐ = f(A)
    s = zeros(N)
    for i in 1:N
        if i in A
            s[i] = fᵥ - f(setdiff(V, [i]))
        else
            s[i] = f(union(A, [i])) - fₐ
        end
    end
    return s
end

"""
    Get shrink supergradient s̆ of a submodular function f at A ⊂ V
"""
function shrink_supergrad(f :: Function, A :: Vector{Int}, V :: Vector{Int})
    N = length(V)
    fₐ = f(A)
    s = zeros(N)
    for i in 1:N
        if i in A
            s[i] = fₐ - f(setdiff(A, [i]))
        else
            s[i] = f([i])
        end
    end
    return s
end

"""
    Get bar supergradient s̄ of a submodular function f at A ⊂ V
"""
function bar_supergrad(f :: Function, A :: Vector{Int}, V :: Vector{Int})
    N = length(V)
    fᵥ = f(A)
    s = zeros(N)
    for i in 1:N
        if i in A
            s[i] = fᵥ - f(setdiff(V, [i]))
        else
            s[i] = f([i])
        end
    end
    return s
end


"""
    Get m₁ in Djolonga & Krause (2014)
"""
function get_m₁(f :: Function, N :: Int)
    m = zeros(N)
    f_all = f(collect(1:N))
    for i in 1:N
        f₊ = f([i])
        f₋ = f_all - f(deleteat!(collect(1:N), i))
        m[i] = log(1 + exp(-f₋)) - log(1 + exp(f₊))
    end
    return m
end

"""
    Get m₂ in Djolonga & Krause (2014)
"""
function get_m₂(f :: Function, N :: Int)
    m = zeros(N)
    f_all = f(collect(1:N))
    for i in 1:N
        f₊ = f([i])
        f₋ = f_all - f(deleteat!(collect(1:N), i))
        m[i] = log(1 + exp(f₋)) - log(1 + exp(-f₊))
    end
    return m
end

"""
    Divide-and-conquer algorithm for evaluating an optimal bound of a log-partition function;
"""
function divide_and_conquer(
        f :: Function, N :: Int;
        tol :: Float64 = 1e-3)
    V = collect(1:N)

    s = ones(N) * f(V) / N
    A = minimum_norm_point(f, N, tol = tol)
    if abs(f(A) - sum(s[A])) <= tol
        return s
    else
        sₐ = divide_and_conquer(restriction(f, A, V), length(A))
        sᵥ₋ₐ = divide_and_conquer(contraction(f, A, V), N - length(A))
        for i in 1:N
            if i in A
                s[i] = sₐ[1]
                deleteat!(sₐ, 1)
            else
                s[i] = sᵥ₋ₐ[1]
                deleteat!(sᵥ₋ₐ, 1)
            end
        end
        return s
    end
end

"""
    Yielding lower-bound of a log-partition function if f is submodular.
"""
function lower_logZ_submod(
        f :: Function, N :: Int;
        tol :: Float64 = 1e-3)
    x = divide_and_conquer(f, N; tol = tol)
    val = sum(x -> log(1 + exp(x)), x)
    return x, val
end

"""
    Yielding upper-bound of a log-partition function if f is supermodular.
"""
function upper_logZ_supermod(
        f :: Function, N :: Int;
        tol :: Float64 = 1e-3)
    x = divide_and_conquer(A -> -f(A), N; tol = tol)
    val = sum(x -> log(1 + exp(-x)), x)
    return x, val
end


"""
    Evaluating an optimal bound of a log-partition function;
    Yielding lower-bound if f is supermodular.
"""
function lower_logZ_supermod(
        f :: Function, N :: Int;
        tol :: Float64 = 1e-3, choose_gradient :: Bool = true)
    V = collect(1:N)
    f₋ = A -> -f(A)
    m₂ = get_m₂(f₋, N)

    A = minimum_norm_point(A -> f₋(A) - sum(m₂[A]), N, tol = tol)
    if choose_gradient
        supergrads = [grow_supergrad(f₋, A, V), shrink_supergrad(f₋, A, V), bar_supergrad(f₋, A, V)]
        vals = [sum(x -> log(1 + exp(-x)), s) for s in supergrads]
        val_max, ind_max = findmax(vals)
        c = f₋(A) - sum(supergrads[ind_max][A])
        return supergrads[ind_max], val_max - c
    else
        s = bar_supergrad(f₋, A, V)
        c = f₋(A) - sum(s[A])
        return s, sum(x -> log(1 + exp(-x)), s) - c
    end
end

"""
    Evaluating an optimal bound of a log-partition function;
    Yielding upper-bound if f is submodular.
"""
function upper_logZ_submod(
        f :: Function, N :: Int;
        tol :: Float64 = 1e-3, choose_gradient :: Bool = true)
    V = collect(1:N)
    m₁ = get_m₁(f, N)
    A = minimum_norm_point(A -> f(A) + sum(m₁[A]), N, tol = tol)
    if choose_gradient
        supergrads = [grow_supergrad(f, A, V), shrink_supergrad(f, A, V), bar_supergrad(f, A, V)]
        vals = [sum(x -> log(1 + exp(x)), s) for s in supergrads]
        val_min, ind_min = findmin(vals)
        c = f(A) - sum(supergrads[ind_min][A])
        return supergrads[ind_min], val_min + c
    else
        s = bar_supergrad(f, A, V)
        c = f(A) - sum(s[A])
        return s, sum(x -> log(1 + exp(x)), s) + c
    end
end


"""
    Frank-Wolfe Iteration for evaluating an optimal bound of a log-partition function;
    Yielding upper-bound if f is supermodular.
"""
function upper_logZ_supermod_fw(
        f :: Function, x :: Vector{Float64};
        Tmin :: Int64 = 10, tol :: Float64 = 1e-3)

    ∇f(x) = -1 ./ (1 .+ exp.(x))

    conv_trace = zeros(0)
    t = 0
    conv_indicator = tol * 10
    while !(t >= Tmin && abs(conv_indicator) < tol)
        s = get_extremepoint(-∇f(x), A -> -f(A))

        conv_indicator = (x - s)' * ∇f(x)
        push!(conv_trace, conv_indicator)
        γₜ = 2 / (t + 2)
        x = (1 - γₜ) * x + γₜ * s
        t += 1
    end
    val = sum(x -> log(1 + exp(-x)), x)
    return x, val
end

"""
    Frank-Wolfe Iteration for evaluating an optimal bound of a log-partition function;
    Yielding lower-bound if f is submodular.
"""
function lower_logZ_submod_fw(
        f :: Function, x :: Vector{Float64};
        Tmin :: Int64 = 10, tol :: Float64 = 1e-3)
    x, val = upper_logZ_supermod_fw(A -> -f(A), -x; Tmin = Tmin, tol = tol)
    val = sum(x -> log(1 + exp(x)), x)
    return x, val
end

function upper_logZ_supermod_fw(
        f :: Function, N :: Int;
        Tmin :: Int64 = 10, tol :: Float64 = 1e-3)
    x, val = upper_logZ_supermod_fw(f, zeros(N); Tmin = Tmin, tol = tol)
    return x, val
end

function lower_logZ_submod_fw(
        f :: Function, N :: Int;
        Tmin :: Int64 = 10, tol :: Float64 = 1e-3)
    x, val = upper_logZ_supermod_fw(A -> -f(A), zeros(N); Tmin = Tmin, tol = tol)
    val = sum(x -> log(1 + exp(x)), x)
    return x, val
end


"""
    mean-field approximation of a subset probability P(A) ∝ exp(f(A))
"""
function mean_field_approx(f :: Function, N :: Int; M :: Int = 10, N_iter = 15)
    q = zeros(N, N_iter + 1)
    q[:, 1] .= 0.5
    for itr in 2:(N_iter + 1)
        _q = copy(q[:, itr - 1])
        for i in 1:N
            q₋ᵢ = copy(_q)
            q₋ᵢ[i] = 0.0
            X = hcat(rand.(Bernoulli.(q₋ᵢ), M)...)'
            A_ary = [findall(Bool.(X[:, m])) for m in 1:M]

            f_diff_mean = mean(A -> f(union(A, i)) - f(A), A_ary)
            _q[i] = sigmoid(f_diff_mean)
        end
        q[:, itr] .= _q
    end
    return q
end

"""
    obtain an ELBO with the mean-field approximation
"""
function mean_field_ELBO(f :: Function, N :: Int; M :: Int = 50, M_mf :: Int = 10, N_iter = 15)
    q_mat = mean_field_approx(f, N; M = M_mf, N_iter = N_iter)
    q = q_mat[:, end]
    entropy = -sum(q .* log.(q) .+ (1 .- q) .* log.(1 .- q))
    X = hcat(rand.(Bernoulli.(q), M)...)'
    A_ary = [findall(X[:, i]) for i in 1:M]
    multilinear = mean(f, A_ary)

    return multilinear + entropy
end
