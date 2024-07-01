using Random
using LinearAlgebra
using StaticArrays
using KernelFunctions
using Distributions
using Zygote
using ProgressMeter


struct KPP
    ϕ :: Function
    dϕ :: Function
    L :: AbstractMatrix
    N :: Int64

    function KPP(ϕ, L)
        N = size(L, 1)
        new(ϕ, ϕ', L, N)
    end

    function KPP(ϕ, dϕ, L)
        N = size(L, 1)
        new(ϕ, dϕ, L, N)
    end
end

function trϕ(A :: Vector{Int}, kpp :: KPP; tol = 1e-10)
    n = length(A)
    if iszero(n)
        return 0.0
    else
        λ_ary = eigvals(@views kpp.L[A, A])
        for i in 1:length(λ_ary)
            if -tol < λ_ary[i] < 0.0
                λ_ary[i] = 0.0
            end
        end
        return sum(kpp.ϕ, λ_ary)
    end
end

function gibbs(
        kpp :: KPP,
        n_samples :: Int;
        n_thinning :: Int = 0,
        show_progress :: Bool = true)
    p = Progress(n_samples; enabled = show_progress, desc = "Sampling...")
    N = kpp.N
    S = zeros(Int, 0)
    S_trace = Vector{Vector{Int}}(undef, 0)

    for iter in 1:n_samples
        for i in 1:(N * (n_thinning + 1))
            cand_item = rand(1:N)
            S = setdiff(S, cand_item)
            S_cand = union(S, cand_item)
            Δf = trϕ(S_cand, kpp) - trϕ(S, kpp)

            is_accept = rand(Bernoulli(1/(1 + exp(-Δf))))
            S = is_accept ? S_cand : S
            sort!(S)
        end
        append!(S_trace, [S])
        next!(p)
    end
    return S_trace
end

function rwm(
        kpp :: KPP,
        n_samples :: Int;
        n_thinning :: Int = 0,
        show_progress :: Bool = true)
    p = Progress(n_samples; enabled = show_progress, desc = "Sampling...")

    N = kpp.N
    S_present = zeros(Int, 0)
    S_trace = Vector{Vector{Int}}(undef, 0)

    for iter in 1:n_samples
        for i in 1:(N * (n_thinning + 1))
            cand_item = rand(1:N)
            S_new = copy(S_present)
            if cand_item in S_new
                deleteat!(S_new, findall(x -> x == cand_item, S_new))
            else
                push!(S_new, cand_item)
            end
            #cand_items = randperm(N)[1:10]
            #S_new = copy(S_present)
            #for cand_item in cand_items
            #    if cand_item in S_new
            #        deleteat!(S_new, findall(x -> x == cand_item, S_new))
            #    else
            #        push!(S_new, cand_item)
            #    end
            #end

            Δf = trϕ(S_new, kpp) - trϕ(S_present, kpp)

            if Δf >= 0
                S_present = copy(S_new)
            else
                if log(rand()) <= Δf
                    S_present = copy(S_new)
                end
            end
        end
        push!(S_trace, S_present)
        next!(p)
    end
    return S_trace
end

"""
    Metropolis sampler under cardinality constraint
"""
function rwm_cardinality(
        kpp :: KPP,
        k :: Int,
        n_samples :: Int;
        n_thinning :: Int = 0,
        show_progress :: Bool = true)
    p = Progress(n_samples; enabled = show_progress, desc = "Sampling...")

    if k >= kpp.N
        throw(ArgumentError("The cardinality k must be less than the number of elements in the ground set."))
    elseif k <= 0
        throw(ArgumentError("The cardinality k must be positive."))
    end

    N = kpp.N
    V = collect(1:N)
    S_present = randperm(N)[1:k]
    S_trace = Vector{Vector{Int}}(undef, 0)

    for iter in 1:n_samples
        for i in 1:(N * (n_thinning + 1))
            cand_item_in = rand(S_present)
            cand_item_out = rand(setdiff(V, S_present))

            S_new = copy(S_present)
            deleteat!(S_new, findall(x -> x == cand_item_in, S_new))
            push!(S_new, cand_item_out)

            Δf = trϕ(S_new, kpp) - trϕ(S_present, kpp)

            if Δf >= 0
                S_present = copy(S_new)
            else
                if log(rand()) <= Δf
                    S_present = copy(S_new)
                end
            end
        end
        push!(S_trace, S_present)
        next!(p)
    end
    return S_trace
end



function box_cox(x, λ)
    if iszero(λ)
        return log(x)
    else
        return (x ^ λ - 1.0) / λ
    end
end

function box_cox_derivative(x, λ)
    if iszero(λ)
        return 1.0 / x
    else
        return x ^ (λ - 1)
    end
end

"""
    return the derivative dϕ(L[A])/dL
"""
function dϕ(A :: Vector{Int}, kpp :: KPP)
    n = length(A)
    if iszero(n)
        return 0.0
    else
        λ, U = eigen(@views kpp.L[A, A])
        ϕΛ = @views Diagonal(kpp.dϕ.(λ))
        return U * ϕΛ * U'
    end
end
