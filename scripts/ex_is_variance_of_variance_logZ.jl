using Base.Threads
using Random
using StatsBase
using ProgressBars
using Plots
using ColorSchemes
using JLD2

include("KPP.jl");
include("marginal_evaluater.jl");
include("kpp_expectations.jl");

palette = :tol_bright
default(palette = palette,
        markerstrokecolor = :auto,
        titlefontsize = 16, tickfontsize = 10, legendfontsize = 12, labelfontsize = 14,
        fontfamily = "Helvetica")

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
        Threads.@threads for k in 1:N_trials
            val_is_mf = importance_sampling_Z(kpp, N; M_mf = 15)
            val_is = importance_sampling_Z(kpp, ones(N) .* 0.5)

            vals_is[j, i, k] = log(val_is)
            vals_is_mf[j, i, k] = log(val_is_mf)
        end
    end
end

jldsave("../outputs/DKPP_logZ_is_n64.jld2"; vals_is)
jldsave("../outputs/DKPP_logZ_is_mf_n64.jld2"; vals_is_mf)
#vals_is = load("../outputs/DKPP_logZ_is_n64.jld2")["vals_is"]
#vals_is_mf = load("../outputs/DKPP_logZ_is_mf_n64.jld2")["vals_is_mf"]

mean_vals_is = reshape(mean(vals_is, dims = 3), (N_trials, :))
var_vals_is = reshape(var(vals_is, dims = 3), (N_trials, :))
mean_vals_is_mf = reshape(mean(vals_is_mf, dims = 3), (N_trials, :))
var_vals_is_mf = reshape(var(vals_is_mf, dims = 3), (N_trials, :))

mean_of_var_vals_is = reshape(mean(var_vals_is, dims = 1), :)
mean_of_var_vals_is_mf = reshape(mean(var_vals_is_mf, dims = 1), :)
std_of_var_vals_is = reshape(std(var_vals_is, dims = 1), :)
std_of_var_vals_is_mf = reshape(std(var_vals_is_mf, dims = 1), :)

p = scatter(λ_ary, [mean_of_var_vals_is_mf mean_of_var_vals_is],
            yerror = [std_of_var_vals_is_mf std_of_var_vals_is],
            label = ["importance sampling (w/ MF)" "importance sampling (wo/ MF)"],
            title = "Variance of approximated logZ",
            xlabel = "lambda", ylabel = "Var(logZ)",
            palette = colorschemes[palette][4:end])
savefig(p, "../outputs/DKPP_logZ_is_varlogZ_n64.png")
savefig(p, "../outputs/DKPP_logZ_is_varlogZ_n64.svg")
savefig(p, "../outputs/DKPP_logZ_is_varlogZ_n64.pdf")
