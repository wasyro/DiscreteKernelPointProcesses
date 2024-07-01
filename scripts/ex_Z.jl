using Base.Threads
using Random
using StatsBase
using ProgressBars
using Plots
using ColorSchemes

include("KPP.jl");
include("marginal_evaluater.jl");
include("kpp_expectations.jl");

palette = :tol_bright
default(palette = palette,
        markerstrokecolor = :auto,
        titlefontsize = 16, tickfontsize = 10, legendfontsize = 12, labelfontsize = 14,
        fontfamily = "Helvetica")


N = 16
N_trials = 30
λ_ary = vcat(range(0.0, 2.0, length = 50))
vals_lower = zeros(N_trials, length(λ_ary))
vals_upper = zeros(N_trials, length(λ_ary))
vals_elbo = zeros(N_trials, length(λ_ary))
vals_is = zeros(N_trials, length(λ_ary))
vals_truth = zeros(N_trials, length(λ_ary))
Random.seed!(123)
Threads.@threads for i in ProgressBar(1:length(λ_ary))
    ϕ = x -> box_cox(x, λ_ary[i])
    for j in 1:N_trials
        L = rand(Wishart(N, diagm(ones(N)))) / N
        kpp = KPP(ϕ, L)
        if λ_ary[i] <= 1.0
            # submodular
            x_lower, val_lower = lower_logZ_submod(A -> trϕ(A, kpp), N)
            x_upper, val_upper = upper_logZ_submod(A -> trϕ(A, kpp), N)
        else
            # supermodular
            x_lower, val_lower = lower_logZ_supermod(A -> trϕ(A, kpp), N)
            x_upper, val_upper = upper_logZ_supermod(A -> trϕ(A, kpp), N)
        end

        val_elbo = mean_field_ELBO(A -> trϕ(A, kpp), N; M_mf = 15)

        expval_is = importance_sampling_Z(kpp, N; M_mf = 25)

        Z = 0.0
        for i in 0:(2^N - 1)
            z = reverse(digits(i, base = 2, pad = N))
            A = findall(Vector{Bool}(z))
            Z += exp(trϕ(A, kpp))
        end
        logZ = log(Z)

        vals_lower[j, i] = val_lower
        vals_upper[j, i] = val_upper
        vals_elbo[j, i] = val_elbo
        vals_is[j, i] = log(expval_is)
        vals_truth[j, i] = logZ
    end
end


mean_diffvals_is = reshape(mean(vals_is - vals_truth, dims = 1), :)
std_diffvals_is = reshape(std(vals_is - vals_truth, dims = 1), :)
p = scatter(λ_ary, [mean_diffvals_lower mean_diffvals_upper mean_diffvals_elbo mean_diffvals_is],
            yerror = [std_diffvals_lower std_diffvals_upper std_diffvals_elbo std_diffvals_is],
            label = ["L-field (lower)" "L-field (upper)" "ELBO" "importance sampling"],
            title = "approximation and bounds of logZ - Gaps",
            xlabel = "lambda", ylabel = "Gap from Truth",
            xlims = (minimum(λ_ary) - 0.05, maximum(λ_ary) + 0.05),
            legendfontsize = 12)
hline!(p, [0.0], lw = 2, ls = :dot, lc = :black, label = "")
savefig(p, "../outputs/DKPP_logZ_gaps_is.png")
savefig(p, "../outputs/DKPP_logZ_gaps_is.svg")
savefig(p, "../outputs/DKPP_logZ_gaps_is.pdf")


mean_ratiovals_elbo = reshape(mean(exp.(vals_elbo .- vals_truth), dims = 1), :)
std_ratiovals_elbo = reshape(std(exp.(vals_elbo .- vals_truth), dims = 1), :)
mean_ratiovals_is = reshape(mean(exp.(vals_is .- vals_truth), dims = 1), :)
std_ratiovals_is = reshape(std(exp.(vals_is .- vals_truth), dims = 1), :)
p = scatter(λ_ary, [mean_ratiovals_elbo mean_ratiovals_is],
            yerror = [std_ratiovals_elbo std_ratiovals_is],
            label = ["ELBO" "importance sampling"],
            title = "approximation qualities of Z - Ratio",
            xlabel = "lambda", ylabel = "Ratio from Truth",
            palette = colorschemes[palette][3:end],
            xlims = (minimum(λ_ary) - 0.05, maximum(λ_ary) + 0.05),
            legendposition = :bottomright, legendfontsize = 12)
hline!(p, [1.0], lw = 2, ls = :dot, lc = :black, label = "")
savefig(p, "../outputs/DKPP_logZ_ratio_is.png")
savefig(p, "../outputs/DKPP_logZ_ratio_is.svg")
savefig(p, "../outputs/DKPP_logZ_ratio_is.pdf")



## with and without mean-field approximation
N = 64
N_trials = 30
λ_ary = vcat(range(0.0, 2.0, length = 50))
vals_is = zeros(N_trials, length(λ_ary), N_trials)
vals_is_mf = zeros(N_trials, length(λ_ary), N_trials)
Random.seed!(123)
for i in ProgressBar(1:length(λ_ary))
    ϕ = x -> box_cox(x, λ_ary[i])

    for j in 1:N_trials
        L = rand(Wishart(N, diagm(ones(N)))) / N
        kpp = KPP(ϕ, L)
        for k in 1:N_trials
            val_is_mf = importance_sampling_Z(kpp, N; M_mf = 15)
            val_is = importance_sampling_Z(kpp, ones(N) .* 0.5)

            vals_is[j, i, k] = log(val_is)
            vals_is_mf[j, i, k] = log(val_is_mf)
        end
    end
end
