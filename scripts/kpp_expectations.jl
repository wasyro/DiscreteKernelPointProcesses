using Distributions
using Combinatorics
using ProgressMeter

include("KPP.jl")
include("marginal_evaluater.jl")
include("submodular_minimizer.jl")
include("submodular_maximizer.jl")


"""
    Evaluating the partition function Z by importance sampling with mean-field approximation
"""
function importance_sampling_Z(kpp :: KPP, N :: Int; M = 100, M_mf = 10, N_iter_mf = 15)
    q_mat = mean_field_approx(A -> trϕ(A, kpp), N; M = M_mf, N_iter = N_iter_mf)
    q = q_mat[:, end]
    dists_mf = Bernoulli.(q)
    mf_samples = hcat(rand.(dists_mf, M)...)'
    mf_samples_set = [findall(Bool.(mf_samples[:, m])) for m in 1:M]

    weights = map(i -> exp(trϕ(mf_samples_set[i], kpp)) / prod(pdf.(dists_mf, mf_samples[:, i])), 1:M)
    Z_approx = mean(i -> weights[i], 1:M)

    return Z_approx
end

"""
    Evaluating the expectation E[g(A)] by importance sampling with mean-field approximation
"""
function importance_sampling_expectation(g :: Function, kpp :: KPP, N :: Int; M = 100, M_mf = 10, N_iter_mf = 15)
    q_mat = mean_field_approx(A -> trϕ(A, kpp), N; M = M_mf, N_iter = N_iter_mf)
    q = q_mat[:, end]
    dists_mf = Bernoulli.(q)
    mf_samples = hcat(rand.(dists_mf, M)...)'
    mf_samples_set = [findall(Bool.(mf_samples[:, m])) for m in 1:M]

    weights = map(i -> exp(trϕ(mf_samples_set[i], kpp)) / prod(pdf.(dists_mf, mf_samples[:, i])), 1:M)
    Z_approx = mean(i -> weights[i], 1:M)
    Eg_approx = mean(i -> g(mf_samples_set[i]) * weights[i] / Z_approx , 1:M)

    return Eg_approx
end

"""
    Evaluating the partition function Z by importance sampling with given proposal Bernoulli probabilities
"""
function importance_sampling_Z(kpp :: KPP, q :: Vector{Float64}; M = 100)
    dists_mf = Bernoulli.(q)
    mf_samples = hcat(rand.(dists_mf, M)...)'
    mf_samples_set = [findall(Bool.(mf_samples[:, m])) for m in 1:M]

    weights = map(i -> exp(trϕ(mf_samples_set[i], kpp)) / prod(pdf.(dists_mf, mf_samples[:, i])), 1:M)
    Z_approx = mean(i -> weights[i], 1:M)

    return Z_approx
end

"""
    Evaluating the expectation E[g(A)] by importance sampling with given proposal Bernoulli probabilities
"""
function importance_sampling_expectation(g :: Function, kpp :: KPP, q :: Vector{Float64}; M = 100)
    dists_mf = Bernoulli.(q)
    mf_samples = hcat(rand.(dists_mf, M)...)'
    mf_samples_set = [findall(Bool.(mf_samples[:, m])) for m in 1:M]

    weights = map(i -> exp(trϕ(mf_samples_set[i], kpp)) / prod(pdf.(dists_mf, mf_samples[:, i])), 1:M)
    Z_approx = mean(i -> weights[i], 1:M)
    Eg_approx = mean(i -> g(mf_samples_set[i]) * weights[i] / Z_approx , 1:M)

    return Eg_approx
end


"""
    Evaluating the marginal P(A_in ⊆ A ⊆ A_out) by Rao--Blackwellized importance sampling with mean-field approximation
"""
function importance_sampling_marginal(
        A_in :: Vector{Int}, A_out :: Vector{Int}, kpp :: KPP, N :: Int;
        M = 100, M_mf = 10, N_iter_mf = 15)
    if !issubset(A_in, A_out)
        error("A_in must be a subset of A_out")
    end

    A_diff = setdiff(A_out, A_in)
    q_mat = mean_field_approx(A -> trϕ(A, kpp), N; M = M_mf, N_iter = N_iter_mf)
    q = q_mat[:, end]
    dists_mf = Bernoulli.(q)
    dists_mf_diff = @views dists_mf[A_diff]
    mf_samples = hcat(rand.(dists_mf, M)...)'
    mf_samples_diff = @views mf_samples[A_diff, :]
    mf_samples_set = [findall(Bool.(mf_samples[:, m])) for m in 1:M]
    mf_samples_set_diff = [findall(Bool.(mf_samples_diff[:, m])) for m in 1:M]

    Z_approx = mean(i -> exp(trϕ(mf_samples_set[i], kpp)) / prod(pdf.(dists_mf, mf_samples[:, i])), 1:M)
    if A_in == A_out
        return exp(trϕ(A_in, kpp)) / Z_approx
    end
    numer = mean(i -> exp(trϕ(union(A_in, A_diff[mf_samples_set_diff[i]]), kpp)) / prod(pdf.(dists_mf_diff, mf_samples_diff[:, i])), 1:M)

    return numer / Z_approx
end


"""
    Evaluating the marginal P(A_in ⊆ A ⊆ A_out) by Rao--Blackwellized importance sampling with given proposal Bernoulli probabilities
"""
function importance_sampling_marginal(A_in :: Vector{Int}, A_out :: Vector{Int}, kpp :: KPP, q :: Vector{Float64}; M = 100)
    if !issubset(A_in, A_out)
        error("A_in must be a subset of A_out")
    end

    A_diff = setdiff(A_out, A_in)
    dists_mf = Bernoulli.(q)
    dists_mf_diff = @views dists_mf[A_diff]
    mf_samples = hcat(rand.(dists_mf, M)...)'
    mf_samples_diff = @views mf_samples[A_diff, :]
    mf_samples_set = [findall(Bool.(mf_samples[:, m])) for m in 1:M]
    mf_samples_set_diff = [findall(Bool.(mf_samples_diff[:, m])) for m in 1:M]

    Z_approx = mean(i -> exp(trϕ(mf_samples_set[i], kpp)) / prod(pdf.(dists_mf, mf_samples[:, i])), 1:M)
    if A_in == A_out
        return exp(trϕ(A_in, kpp)) / Z_approx
    end
    numer = mean(i -> exp(trϕ(union(A_in, A_diff[mf_samples_set_diff[i]]), kpp)) / prod(pdf.(dists_mf_diff, mf_samples_diff[:, i])), 1:M)

    return numer / Z_approx
end


"""
    Evaluating the marginal P(|A| = k) by Rao--Blackwellized importance sampling
"""
function importance_sampling_marginal(
        k :: Int, kpp :: KPP;
        Z :: Union{Float64, Nothing} = nothing,
        normalize :: Bool = true,
        mean_field :: Bool = true,
        take_log :: Bool = false,
        q :: Vector{Float64} = ones(kpp.N) * 0.5,
        M = 100, M_mf = 10, N_iter_mf = 15)
    if normalize
        if isnothing(Z)
            if mean_field
                q_mat = mean_field_approx(A -> trϕ(A, kpp), N; M = M_mf, N_iter = N_iter_mf)
                q = q_mat[:, end]
            end
        else
            Z = importance_sampling_Z(kpp, q; M = M)
        end
    end

    mc_samples = [randperm(N)[1:k] for _ in 1:M]
    logEw = log(normalize ? mean(A -> exp(trϕ(A, kpp)) / Z, mc_samples) : mean(A -> exp(trϕ(A, kpp)), mc_samples))
    logbinom = log(binomial(BigInt(kpp.N), BigInt(k)))

    if take_log
        return Float64(logEw + logbinom)
    else
        return Float64(exp(logEw + logbinom))
    end
end
