using Random
include("submodular_minimizer.jl")

"""
    Deterministic Unconstrained Submoudular Maximization algorithm for non-monotone and non-negative submodular functions.
    This yields 1/3-approximation of the optimal value of f.
    Buchbinder, et al., (2012)
"""
function deterministic_usm(f :: Function, N :: Int;
        order :: Union{Nothing, Vector{Int}} = nothing,
        ensure_non_negative :: Bool = false)

    if !ensure_non_negative
        _f = f
    else
        # ensure non-negativity
        fmin = f(minimum_norm_point(f, N))
        _f = A -> f(A) + fmin
    end

    A = Int[]
    B = collect(1:N)

    if !isnothing(order)
        if Set(order) != Set(1:N)
            throw(ArgumentError("Got an invalid the argument for order. Set(order) == Set(1:N) must be satisfied."))
        end
    else
        order = collect(1:N)
    end

    a = zeros(N)
    b = zeros(N)
    for i in 1:N
        a[i] = _f(union(A, order[i])) - _f(A)
        b[i] = _f(setdiff(B, order[i])) - _f(B)
        if a[i] >= b[i]
            A = union(A, order[i])
        else
            B = setdiff(B, order[i])
        end
    end
    return A
end


"""
    Randomized Unconstrained Submoudular Maximization algorithm for non-monotone and non-negative submodular functions.
    This yields 1/2-approximation of the optimal value of f in expectation.
    Buchbinder, et al., (2012)
"""
function randomized_usm(f :: Function, N :: Int;
        order :: Union{Nothing, Vector{Int}} = nothing,
        ensure_non_negative :: Bool = false,
        seed :: Union{Nothing, Int} = nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    if !ensure_non_negative
        _f = f
    else
        # ensure non-negativity
        fmin = f(minimum_norm_point(f, N))
        _f = A -> f(A) + fmin
    end

    A = Int[]
    B = collect(1:N)

    if !isnothing(order)
        if Set(order) != Set(1:N)
            throw(ArgumentError("Got an invalid the argument for order. Set(order) == Set(1:N) must be satisfied."))
        end
    else
        order = collect(1:N)
    end

    a = zeros(N)
    b = zeros(N)
    for i in 1:N
        a[i] = _f(union(A, order[i])) - _f(A)
        b[i] = _f(setdiff(B, order[i])) - _f(B)
        if rand() <= max(a[i], 0) / (max(a[i], 0) + max(b[i], 0))
            A = union(A, order[i])
        else
            B = setdiff(B, order[i])
        end
    end
    return A
end


"""
    Randomized greedy algorithm for cardinality constrained submoudular maximization.
    This yields ((1 - k/(eN)) / e - ε)-approximation of the optimal value of f in expectation, when k ≤ N/2.
    Buchbinder, et al., (2014)
"""
function randomized_csm(f :: Function, k :: Int, N :: Int;
        order :: Union{Nothing, Vector{Int}} = nothing,
        ensure_non_negative :: Bool = false,
        seed :: Union{Nothing, Int} = nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    if !ensure_non_negative
        _f = f
    else
        # ensure non-negativity
        fmin = f(minimum_norm_point(f, N))
        _f = A -> f(A) + fmin
    end

    A = Int[]
    B = collect(1:N)

    if !isnothing(order)
        if Set(order) != Set(1:N)
            throw(ArgumentError("Got an invalid the argument for order. Set(order) == Set(1:N) must be satisfied."))
        end
    else
        order = collect(1:N)
    end

    for i in 1:k
        a = ones(N) * -Inf
        for j in B
            a[j] = _f(union(A, j)) - _f(A)
        end
        M = sortperm(a, rev = true)[1:k]
        u = rand(M)
        A = union(A, u)
        B = setdiff(B, u)
    end
    return A
end
