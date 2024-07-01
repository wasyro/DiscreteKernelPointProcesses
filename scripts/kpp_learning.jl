using Random
using LinearAlgebra
using SparseArrays
using StatsBase
using Combinatorics
using ProgressMeter
using ProgressBars
using Plots
using JLD2

include("KPP.jl");
include("marginal_evaluater.jl");
include("kpp_expectations.jl");


### For Ratio Matching
##

function g(u :: Real)
    return 1 / (1 + u)
end

function compute_mean_loglik(samples :: Vector{Vector{Int}}, kpp :: KPP)
    N = kpp.N
    Z = 0.0
    for i in 0:(2^N - 1)
        z = reverse(digits(i, base = 2, pad = N))
        A = findall(Vector{Bool}(z))
        Z += exp(trϕ(A, kpp))
    end
    return mean(A -> trϕ(A, kpp), samples) - log(Z)
end

function J_RM_partial(A :: Vector{Int}, n :: Int, kpp :: KPP)
    Aⁿ = n in A ? setdiff(A, n) : sort!(union(A, n))
    return g(exp(trϕ(A, kpp) - trϕ(Aⁿ, kpp))) ^ 2
end

function J_RM(samples :: Vector{Vector{Int}}, kpp :: KPP)
    N = kpp.N
    return sum([J_RM_partial(sample, n, kpp) for sample in samples, n in 1:N]) / length(samples)
end

function grad_J_RM_partial(A :: Vector{Int}, n :: Int, kpp :: KPP)
    Aⁿ = n in A ? setdiff(A, n) : sort!(union(A, n))
    u_m_n = exp(trϕ(A, kpp) - trϕ(Aⁿ, kpp))

    U_Aⁿ = sparse(I(N)[Aⁿ, :])
    U_A = sparse(I(N)[A, :])
    dg²_du = -2 * g(u_m_n) / (1 + u_m_n) ^ 2
    du_dL = u_m_n * (U_A' * dϕ(A, kpp) * U_A - U_Aⁿ' * dϕ(Aⁿ, kpp) * U_Aⁿ)

    return dg²_du * du_dL
end

function grad_J_RM_minibatch(samples :: Vector{Vector{Int}}, n_ary :: Vector{Int}, kpp :: KPP)
    N = kpp.N
    n_minibatch_items = length(n_ary)

    return mean([grad_J_RM_partial(A, n, kpp) for A in samples, n in n_ary]) * N / n_minibatch_items
end

function grad_loglik(samples :: Vector{Vector{Int}}, kpp :: KPP)
    N = kpp.N

    Z = 0.0
    numer = zeros(N, N)
    for i in 0:(2^N - 1)
        z = reverse(digits(i, base = 2, pad = N))
        A = findall(Vector{Bool}(z))
        U_A = sparse(I(N)[A, :])

        val = exp(trϕ(A, kpp))
        numer .+= val * U_A' * dϕ(A, kpp) * U_A
        Z += val
    end

    U_samples = [sparse(I(N)[sample, :]) for sample in samples]
    term1 = mean(m -> U_samples[m]' * dϕ(samples[m], kpp) * U_samples[m], 1:length(samples))
    term2 = -numer / Z
    return term1 + term2
end


"""
    Learning a DKPP by ratio matching and SGD

    η: learning rate
    ϵ: fudge factor for AdaGrad
    β₁: mixing parameter of Momentum
    β₂: mixing parameter of RMSProp
    ρ: regularization parameter
    regularizer: type of regularizer (only 'l2' is supported)
    n_minibatch_samples: number of samples in a minibatch
    n_minibatch_items: number of items in a minibatch
"""
function sgd_rm(kpp :: KPP, samples, n_iter;
        show_progress = true, optimizer = "sgd",
        η = 1e-3, ϵ = 1e-7, β₁ = 0.9, β₂ = 0.999,
        ρ = 1e-3, regularizer = "l2",
        n_minibatch_samples = min(10, length(samples)),
        n_minibatch_items = min(10, kpp.N))
    if !(optimizer in ["adam", "sgd"])
        throw(ArgumentError("Invalid optimizer: $optimizer. Choose from 'adam' or 'sgd'"))
    end
    if !(regularizer in ["l2"])
        throw(ArgumentError("Invalid regularizer: $regularizer. Choose from 'l2'"))
    end

    prog = Progress(n_iter - 1, enabled = show_progress)

    kpp_trace = Vector{KPP}(undef, n_iter)
    kpp_trace[1] = kpp
    cputime_trace = zeros(n_iter)

    V_present = cholesky(kpp.L).L
    historical_grad = zeros(kpp.N, kpp.N)
    historical_velocity = zeros(kpp.N, kpp.N)

    for i in 2:n_iter
        cputime_trace[i] = @elapsed begin
            m_ary = randperm(length(samples))[1:n_minibatch_samples]
            n_ary = randperm(N)[1:n_minibatch_items]

            gradL = grad_J_RM_minibatch(samples[m_ary], n_ary, kpp_trace[i - 1])
            gradV =  2 * gradL * V_present
            if regularizer == "l2"
                gradV += 2 * ρ * V_present
            end

            if optimizer == "adam"
                historical_grad = β₂ * historical_grad + (1 - β₂) * gradV .^ 2
                historical_velocity = β₁ * historical_velocity + (1 - β₁) * gradV
                adj_hgrad = historical_grad ./ (1 - β₂ ^ (i - 1))
                adj_hvelocity = historical_velocity ./ (1 - β₁ ^ (i - 1))

                adj_grad = adj_hvelocity ./ (.√(adj_hgrad) .+ ϵ)
                V_present -= η * adj_grad
            elseif optimizer == "sgd"
                V_present -= η * gradV
            end

            kpp_trace[i] = KPP(kpp.ϕ, kpp.dϕ, V_present * V_present')
        end
        next!(prog)
    end
    return kpp_trace, cumsum(cputime_trace)
end

##

### For Localized Pseudo-sphere Divergence
##

"""
    gradient w.r.t. L of log ∑ w(A) exp(trϕ(L[A])) ^ (1 - c), where w(A) = (n_duplicated[A] / M) ^ c.
"""
function grad_logsum_psd(samples :: Vector{Vector{Int}}, kpp :: KPP, c :: Real)
    M = length(samples)
    samples_unique = unique(samples)
    n_duplicated = [count(A -> sample_unique == A, samples) for sample_unique in samples_unique]
    M_unique = length(samples_unique)

    U_samples_unique = [sparse(I(N)[A, :]) for A in samples_unique]
    denom = sum(m -> (n_duplicated[m] / M) ^ c * exp((1 - c) * trϕ(samples_unique[m], kpp)), 1:M_unique)
    numer = (1 - c) * sum(
        m -> (n_duplicated[m] / M) ^ c * exp((1 - c) * trϕ(samples_unique[m], kpp)) * U_samples_unique[m]' * dϕ(samples_unique[m], kpp) * U_samples_unique[m],
        1:M_unique)

    return numer / denom
end

function localized_ps(kpp1 :: KPP, kpp2 :: KPP;
        α :: Real = 2, α′ :: Real = -1, γ :: Real = -(1 - α) / (1 - α′)) :: Float64
    β = (α + γ * α′) / (1 + γ)
    all_subsets = collect(combinations(1:kpp1.N))

    Z = sum(A -> exp(trϕ(A, kpp1)), all_subsets)
    term1 = log(sum(A -> (exp(trϕ(A, kpp1)) / Z) ^ α * exp((1 - α) * trϕ(A, kpp2)), all_subsets)) / (1 + γ)
    term2 = log(sum(A -> (exp(trϕ(A, kpp1)) / Z) ^ α′ * exp((1 - α′) * trϕ(A, kpp2)), all_subsets)) * γ / (1 + γ)
    term3 = isone(β) ? 0.0 : -log(sum(A -> (exp(trϕ(A, kpp1)) / Z) ^ β * exp((1 - β) * trϕ(A, kpp2)), all_subsets))

    return term1 + term2 + term3
end


function localized_ps(samples :: Vector{Vector{Int}}, kpp :: KPP;
        α :: Real = 2, α′ :: Real = -1, γ :: Real = -(1 - α) / (1 - α′)) :: Float64
    β = (α + γ * α′) / (1 + γ)
    M = length(samples)
    samples_unique = unique(samples)
    n_duplicated = [count(A -> sample_unique == A, samples) for sample_unique in samples_unique]
    M_unique = length(samples_unique)

    term1 = log(sum(m -> (n_duplicated[m] / M) ^ α * exp((1 - α) * trϕ(samples_unique[m], kpp)), 1:M_unique)) / (1 + γ)
    term2 = log(sum(m -> (n_duplicated[m] / M) ^ α′ * exp((1 - α′) * trϕ(samples_unique[m], kpp)), 1:M_unique)) * γ / (1 + γ)
    term3 = isone(β) ? 0.0 : -log(sum(m -> (n_duplicated[m] / M) ^ β * exp((1 - β) * trϕ(samples_unique[m], kpp)), 1:M_unique))

    return term1 + term2 + term3
end

"""
    gradient of the localized pseudo-sphere divergence w.r.t. L.
"""
function grad_localized_ps(samples :: Vector{Vector{Int}}, kpp :: KPP;
        α :: Real = 2, α′ :: Real = -1, γ :: Real = -(1 - α) / (1 - α′)) :: Matrix{Float64}
    β = (α + γ * α′) / (1 + γ)

    M = length(samples)
    samples_unique = unique(samples)
    n_duplicated = [count(A -> sample_unique == A, samples) for sample_unique in samples_unique]
    M_unique = length(samples_unique)

    U_samples_unique = [sparse(I(N)[A, :]) for A in samples_unique]
    trϕ_unique_ary = [trϕ(A, kpp) for A in samples_unique]
    UtdϕU_unique_ary = [U_samples_unique[m]' * dϕ(samples_unique[m], kpp) * U_samples_unique[m] for m in 1:M_unique]

    term1_numer = (1 - α) * sum(
        m -> (n_duplicated[m] / M) ^ α * exp((1 - α) * trϕ_unique_ary[m]) * UtdϕU_unique_ary[m],
        1:M_unique)
    term1_denom = sum(m -> (n_duplicated[m] / M) ^ α * exp((1 - α) * trϕ_unique_ary[m]), 1:M_unique)
    term1 = term1_numer / term1_denom / (1 + γ)

    term2_numer = (1 - α′) * sum(
        m -> (n_duplicated[m] / M) ^ α′ * exp((1 - α′) * trϕ_unique_ary[m]) * UtdϕU_unique_ary[m],
        1:M_unique)
    term2_denom = sum(m -> (n_duplicated[m] / M) ^ α′ * exp((1 - α′) * trϕ_unique_ary[m]), 1:M_unique)
    term2 = term2_numer / term2_denom * γ / (1 + γ)

    if isone(β)
        term3 = zeros(kpp.N, kpp.N)
    else
        term3_numer = -(1 - β) * sum(
            m -> (n_duplicated[m] / M) ^ β * exp((1 - β) * trϕ_unique_ary[m]) * UtdϕU_unique_ary[m],
            1:M_unique)
        term3_denom = sum(m -> (n_duplicated[m] / M) ^ β * exp((1 - β) * trϕ_unique_ary[m]), 1:M_unique)
        term3 = term3_numer / term3_denom
    end
    return term1 + term2 + term3
end

"""
    Learning a DKPP by the localized pseudo-sphere divergence and SGD

    α, α′, γ: hyperparameters of the localized pseudo-sphere divergence
    η: learning rate
    ϵ: fudge factor for AdaGrad
    β₁: mixing parameter of Momentum
    β₂: mixing parameter of RMSProp
    ρ: regularization parameter
    regularizer: type of regularizer (only 'l2' is supported)
"""
function sgd_lpsd(kpp :: KPP, samples, n_iter;
        α :: Real = 2, α′ :: Real = -1, γ :: Real = -(1 - α) / (1 - α′),
        show_progress = true, optimizer = "sgd",
        η = 1e-3, ϵ = 1e-7, β₁ = 0.9, β₂ = 0.999,
        ρ = 1e-3, regularizer = "l2", logarithm = false)
    if !(optimizer in ["adam", "sgd"])
        throw(ArgumentError("Invalid optimizer: $optimizer. Choose from 'adam' or 'sgd'"))
    end
    if !(regularizer in ["l2"])
        throw(ArgumentError("Invalid regularizer: $regularizer. Choose from 'l2'"))
    end

    prog = Progress(n_iter - 1, enabled = show_progress)

    kpp_trace = Vector{KPP}(undef, n_iter)
    kpp_trace[1] = kpp
    cputime_trace = zeros(n_iter)

    V_present = cholesky(kpp.L).L
    historical_grad = zeros(kpp.N, kpp.N)
    historical_velocity = zeros(kpp.N, kpp.N)

    for i in 2:n_iter
        cputime_trace[i] = @elapsed begin
            gradL = grad_localized_ps(samples, kpp_trace[i - 1], α = α, α′ = α′, γ = γ)
            if logarithm
                gradL /= localized_ps(samples, kpp_trace[i - 1], α = α, α′ = α′, γ = γ)
            end
            gradV =  2 * gradL * V_present
            if regularizer == "l2"
                gradV += 2 * ρ * V_present
            end

            if optimizer == "adam"
                historical_grad = β₂ * historical_grad + (1 - β₂) * gradV .^ 2
                historical_velocity = β₁ * historical_velocity + (1 - β₁) * gradV
                adj_hgrad = historical_grad ./ (1 - β₂ ^ (i - 1))
                adj_hvelocity = historical_velocity ./ (1 - β₁ ^ (i - 1))

                adj_grad = adj_hvelocity ./ (.√(adj_hgrad) .+ ϵ)
                V_present -= η * adj_grad
            elseif optimizer == "sgd"
                V_present -= η * gradV
            end

            kpp_trace[i] = KPP(kpp.ϕ, kpp.dϕ, V_present * V_present')
        end
        next!(prog)
    end
    return kpp_trace, cumsum(cputime_trace)
end

function approx_mean_loglik(samples :: Vector{Vector{Int}}, kpp :: KPP; kwargs...) :: Float64
    return mean(A -> trϕ(A, kpp), samples) - log(importance_sampling_Z(kpp, N; kwargs...))
end
