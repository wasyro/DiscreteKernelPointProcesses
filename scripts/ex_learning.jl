using Random
using LinearAlgebra
using SparseArrays
using StatsBase
using ProgressMeter
using ProgressBars
using Plots
using JLD2

include("KPP.jl");
include("marginal_evaluater.jl");
include("kpp_sampler.jl");
include("kpp_learning.jl");


palette = :tol_bright
default(palette = palette,
        markerstrokecolor = :auto,
        titlefontsize = 16, tickfontsize = 10, legendfontsize = 10, labelfontsize = 14,
        fontfamily = "Helvetica")


N = 32
M = 1000
M_test = 100
Random.seed!(123)
ϕ = x -> box_cox(x, 1.5)

L = rand(Wishart(N, diagm(ones(N)))) / N
kpp = KPP(ϕ, L)
samples = gibbs(kpp, M)
samples_test = gibbs(kpp, M_test)


## Adam vs. SGD
Random.seed!(1)
n_iter = 1000
kpp_init = KPP(ϕ, rand(Wishart(N, diagm(ones(N)))) / N)
res_adam = sgd_rm(kpp_init, samples, n_iter; η = 0.001, ρ = 0.0)
res_sgd = sgd_rm(kpp_init, samples, n_iter; η = 0.1, ρ = 0.0, optimizer = "sgd")

logliks_adam = @showprogress [approx_mean_loglik(samples, res_adam[1][i]; N_iter_mf = 5) for i in 1:n_iter]
logliks_sgd = @showprogress [approx_mean_loglik(samples, res_sgd[1][i]; N_iter_mf = 5) for i in 1:n_iter]
logliks_adam_test = @showprogress [approx_mean_loglik(samples_test, res_adam[1][i]; N_iter_mf = 5) for i in 1:n_iter]
logliks_sgd_test = @showprogress [approx_mean_loglik(samples_test, res_sgd[1][i]; N_iter_mf = 5) for i in 1:n_iter]
loglik_truth = approx_mean_loglik(samples, kpp; N_iter_mf = 5, M = 10000)
loglik_truth_test = approx_mean_loglik(samples_test, kpp; N_iter_mf = 5, M = 10000)

p = plot([res_adam[2], res_sgd[2]], [logliks_adam, logliks_sgd], label = ["Adam (train)" "SGD (train)"], xlabel = "CPU time [s]", ylabel = "Log-likelihood (approx.)")
plot!(p, [res_adam[2], res_sgd[2]], [logliks_adam_test, logliks_sgd_test], label = ["Adam (test)" "SGD (test)"])
hline!(p, [loglik_truth loglik_truth_test], label = ["Truth (train)" "Truth (test)"], lw = 2)
savefig(p, "../outputs/DKPP_sgd_rm_loglik_n32_phi15.png")
savefig(p, "../outputs/DKPP_sgd_rm_loglik_n32_phi15.pdf")



## With and withoug regularization
Random.seed!(1)
n_iter = 1000
res_sgd = sgd_rm(kpp_init, samples, n_iter; η = 0.1, ρ = 0.0, optimizer = "sgd")
res_sgd_l2_0002 = sgd_rm(kpp_init, samples, n_iter; η = 0.1, ρ = 0.002, optimizer = "sgd")

logliks_sgd = @showprogress [approx_mean_loglik(samples, res_sgd[1][i]; N_iter_mf = 5) for i in 1:n_iter]
logliks_sgd_l2_0002 = @showprogress [approx_mean_loglik(samples, res_sgd_l2_0002[1][i]; N_iter_mf = 5) for i in 1:n_iter]
logliks_sgd_test = @showprogress [approx_mean_loglik(samples_test, res_sgd[1][i]; N_iter_mf = 5) for i in 1:n_iter]
logliks_sgd_l2_0002_test = @showprogress [approx_mean_loglik(samples_test, res_sgd_l2_0002[1][i]; N_iter_mf = 5) for i in 1:n_iter]
loglik_truth = approx_mean_loglik(samples, kpp; N_iter_mf = 5, M = 10000)
loglik_truth_test = approx_mean_loglik(samples_test, kpp; N_iter_mf = 5, M = 10000)

p = plot([res_sgd[2], res_sgd_l2_0002[2]], [logliks_sgd, logliks_sgd_l2_0002], label = ["wo/ regularization (train)" "w/ L2 regularization (ρ = 0.002, train)"], xlabel = "CPU time [s]", ylabel = "Log-likelihood (approx.)")
plot!(p, [res_sgd[2], res_sgd_l2_0002[2]], [logliks_sgd_test, logliks_sgd_l2_0002_test], label = ["wo/ regularization (test)" "w/ L2 regularization (ρ = 0.002, test)"])
hline!(p, [loglik_truth loglik_truth_test], label = ["Truth (train)" "Truth (test)"], lw = 2)
savefig(p, "../outputs/DKPP_sgd_rm_loglik_n32_phi15_l2_0002.png")
savefig(p, "../outputs/DKPP_sgd_rm_loglik_n32_phi15_l2_0002.pdf")






Random.seed!(123)
ϕ = x -> box_cox(x, 0.5)

L = rand(Wishart(N, diagm(ones(N)))) / N
kpp = KPP(ϕ, L)
samples = gibbs(kpp, M)
samples_test = gibbs(kpp, M_test)

Random.seed!(1)
n_iter = 1000
kpp_init = KPP(ϕ, rand(Wishart(N, diagm(ones(N)))) / N)
res_adam = sgd_rm(kpp_init, samples, n_iter; η = 0.001)
res_sgd = sgd_rm(kpp_init, samples, n_iter; η = 0.1, optimizer = "sgd")

logliks_adam = @showprogress [approx_mean_loglik(samples, res_adam[1][i]; N_iter_mf = 5) for i in 1:n_iter]
logliks_sgd = @showprogress [approx_mean_loglik(samples, res_sgd[1][i]; N_iter_mf = 5) for i in 1:n_iter]
logliks_adam_test = @showprogress [approx_mean_loglik(samples_test, res_adam[1][i]; N_iter_mf = 5) for i in 1:n_iter]
logliks_sgd_test = @showprogress [approx_mean_loglik(samples_test, res_sgd[1][i]; N_iter_mf = 5) for i in 1:n_iter]
loglik_truth = approx_mean_loglik(samples, kpp; N_iter_mf = 5, M = 10000)
loglik_truth_test = approx_mean_loglik(samples_test, kpp; N_iter_mf = 5, M = 10000)
p = plot([res_adam[2], res_sgd[2]], [logliks_adam, logliks_sgd], label = ["Adam (train)" "SGD (train)"], xlabel = "CPU time [s]", ylabel = "Log-likelihood (approx.)")
plot!(p, [res_adam[2], res_sgd[2]], [logliks_adam_test, logliks_sgd_test], label = ["Adam (test)" "SGD (test)"])
hline!(p, [loglik_truth loglik_truth_test], label = ["Truth (train)" "Truth (test)"], lw = 2)
savefig(p, "../outputs/DKPP_sgd_rm_loglik_n32_phi05.png")
savefig(p, "../outputs/DKPP_sgd_rm_loglik_n32_phi05.pdf")




N = 32
M = 1000
M_test = 100
Random.seed!(123)
ϕ = x -> box_cox(x, 0.5)
_dϕ = x -> box_cox_derivative(x, 0.5)

L = rand(Wishart(N, diagm(ones(N)))) / N
kpp = KPP(ϕ, L)
kpp_test = KPP(ϕ_test, L)
samples = gibbs(kpp, M)
samples_test = gibbs(kpp, M_test)


## RM vs. LPSD
Random.seed!(1)
n_iter = 1000
kpp_init = KPP(ϕ, _dϕ, rand(Wishart(N, diagm(ones(N)))) / N)
res_rm = sgd_rm(kpp_init, samples, n_iter; η = 0.1, ρ = 0.0)
res_lpsd = sgd_lpsd(kpp_init, samples, n_iter; η = 0.1, ρ = 0.0, α = 2, α′ = -1)

logliks_rm = @showprogress [approx_mean_loglik(samples, res_rm[1][i]; N_iter_mf = 5) for i in 1:n_iter]
logliks_lpsd = @showprogress [approx_mean_loglik(samples, res_lpsd[1][i]; N_iter_mf = 5) for i in 1:n_iter]
logliks_rm_test = @showprogress [approx_mean_loglik(samples_test, res_rm[1][i]; N_iter_mf = 5) for i in 1:n_iter]
logliks_lpsd_test = @showprogress [approx_mean_loglik(samples_test, res_lpsd[1][i]; N_iter_mf = 5) for i in 1:n_iter]
loglik_truth = approx_mean_loglik(samples, kpp; N_iter_mf = 5, M = 10000)
loglik_truth_test = approx_mean_loglik(samples_test, kpp; N_iter_mf = 5, M = 10000)

#p = plot([res_rm[2], res_lpsd[2]], [logliks_rm, logliks_lpsd], label = ["Ratio Matching (train)" "Localized PS (train)"], xlabel = "CPU time [s]", ylabel = "Log-likelihood (approx.)")
#plot!(p, [res_rm[2], res_lpsd[2]], [logliks_rm_test, logliks_lpsd_test], label = ["Ratio Matching (test)" "Localized PS (test)"])
p = plot([logliks_rm, logliks_lpsd], label = ["Ratio Matching (train)" "Localized PS (train)"], xlabel = "Iteration", ylabel = "Log-likelihood (approx.)")
plot!(p, [logliks_rm_test, logliks_lpsd_test], label = ["Ratio Matching (test)" "Localized PS (test)"])
hline!(p, [loglik_truth loglik_truth_test], label = ["Truth (train)" "Truth (test)"], lw = 2)
savefig(p, "../outputs/DKPP_rm_lpsd_loglik_n32_phi05.png")
savefig(p, "../outputs/DKPP_rm_lpsd_loglik_n32_phi05.pdf")



N = 8
M = 10000
Random.seed!(123)
ϕ = x -> box_cox(x, 0.5)
_dϕ = x -> box_cox_derivative(x, 0.5)

L = rand(Wishart(N, diagm(ones(N)))) / N
kpp = KPP(ϕ, L)
samples = gibbs(kpp, M)

## RM vs. LPSD
Random.seed!(1)
n_iter = 1000
kpp_init = KPP(ϕ, _dϕ, rand(Wishart(N, diagm(ones(N)))) / N)
res_rm = sgd_rm(kpp_init, samples, n_iter; η = 0.1, ρ = 0.0)
res_lpsd = sgd_lpsd(kpp_init, samples, n_iter; η = 0.1, ρ = 0.0, α = 2, α′ = -1)

logliks_rm = @showprogress [compute_mean_loglik(samples, res_rm[1][i]) for i in 1:n_iter]
logliks_lpsd = @showprogress [compute_mean_loglik(samples, res_lpsd[1][i]) for i in 1:n_iter]
loglik_truth = compute_mean_loglik(samples, kpp)

#p = plot([res_rm[2], res_lpsd[2]], [logliks_rm, logliks_lpsd], label = ["Ratio Matching (train)" "Localized PS (train)"], xlabel = "CPU time [s]", ylabel = "Log-likelihood")
p = plot([logliks_rm, logliks_lpsd], label = ["Ratio Matching (train)" "Localized PS (train)"], xlabel = "iteration", ylabel = "Log-likelihood")
hline!(p, [loglik_truth], label = ["Truth (train)"], lw = 2)
savefig(p, "../outputs/DKPP_rm_lpsd_loglik_n8_phi05.png")
savefig(p, "../outputs/DKPP_rm_lpsd_loglik_n8_phi05.pdf")

