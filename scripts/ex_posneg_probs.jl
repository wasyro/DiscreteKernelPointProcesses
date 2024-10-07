using Random
using KernelFunctions
using ProgressBars
using ColorSchemes
using Plots

include("KPP.jl");
include("kpp_expectations.jl");

palette = :tol_bright
default(palette = palette,
        markerstrokecolor = :auto,
        titlefontsize = 16, tickfontsize = 10, legendfontsize = 12, labelfontsize = 14,
        fontfamily = "Helvetica")

# Make gridpoints
xs = collect(1:20)
ys = collect(1:20)
gridpoints = reshape([[x, y] for x in xs, y in ys], :)

N = length(gridpoints)

# Make kernel matrix
L = kernelmatrix(SqExponentialKernel(), gridpoints)

# Indices of GATHERED and SCATTERED (25 points)
inds_gathered = reshape([findfirst(p -> p == [x, y], gridpoints) for x in 8:12, y in 8:12], :)
inds_scattered = reshape(sort(randomized_csm(A -> logdet(L[A, A]), 25, N, seed = 123)), :)

# Compute probabilities
N_trials = 30
λ_ary = collect(range(0.0, 2.0, length = 50))
logprobs_gathered = zeros(N_trials, length(λ_ary))
logprobs_scattered = zeros(N_trials, length(λ_ary))

Random.seed!(123)
@showprogress for i in 1:length(λ_ary)
    kpp = KPP(x -> box_cox(x, λ_ary[i]), L)
    trϕ_scattered = trϕ(inds_scattered, kpp)
    trϕ_gathered = trϕ(inds_gathered, kpp)
    for j in 1:N_trials
        log_marginal = importance_sampling_marginal(25, kpp, normalize = false, mean_field = false, take_log = true, M = 1000)
        logprobs_gathered[j, i] = trϕ_gathered - log_marginal
        logprobs_scattered[j, i] = trϕ_scattered - log_marginal
    end
end

# Plot
x_gp = [p[1] for p in gridpoints]
y_gp = [p[2] for p in gridpoints]


## Check gridpoints
p_scattered = scatter(x_gp, y_gp, color = :black, legend = :none, markersize = 1)
scatter!(p_scattered, x_gp[inds_scattered], y_gp[inds_scattered], color = colorschemes[palette][1], markersize = 6)
p_gathered = scatter(x_gp, y_gp, color = :black, legend = :none, markersize = 1)
scatter!(p_gathered, x_gp[inds_gathered], y_gp[inds_gathered], color = colorschemes[palette][2], markersize = 6)

# Transforming the base of the logarithm to 10
logprobs_scattered = logprobs_scattered / log(10)
logprobs_gathered = logprobs_gathered / log(10)

mean_logprobs_scattered = reshape(mean(logprobs_scattered, dims = 1), :)
std_logprobs_scattered = reshape(std(logprobs_scattered, dims = 1), :)
mean_logprobs_gathered = reshape(mean(logprobs_gathered, dims = 1), :)
std_logprobs_gathered = reshape(std(logprobs_gathered, dims = 1), :)

## Plot log-probabilities
p_probs = scatter(λ_ary, mean_logprobs_scattered, yerror = std_logprobs_scattered, label = "Scattered", xlabel = "λ")
scatter!(p_probs, λ_ary, mean_logprobs_gathered, yerror = std_logprobs_gathered, label = "Gathered")
p_probs
