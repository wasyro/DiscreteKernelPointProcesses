using Random
using LinearAlgebra
using SparseArrays
using StatsBase
using Distributed
using ProgressMeter
using ProgressBars
using CairoMakie
using ColorSchemes
using JLD2
using CSV
using Query
using DataFrames
using Printf

include("KPP.jl");
include("marginal_evaluater.jl");
include("kpp_expectations.jl");
include("kpp_learning.jl");

#using Plots
#palette = :tol_bright
#default(palette = palette,
#        markerstrokecolor = :auto,
#        titlefontsize = 16, tickfontsize = 10, legendfontsize = 12, labelfontsize = 14,
#        fontfamily = "Helvetica")

thm = Theme(palette = (color = colorschemes[:tol_bright], ), font = "Helvetica", fontsize = 18, linewidth = 3)
set_theme!(thm)


LinearAlgebra.BLAS.set_num_threads(Sys.CPU_THREADS - 1)
# load Amazon Baby Registry data
amazon_dir = joinpath("$(@__DIR__)", "..", "data", "AmazonBabyRegistry")
categories = ["apparel", "bath", "bedding", "carseats", "decor", "diaper",
              "feeding", "furniture", "gear", "gifts", "health", "media",
              "moms", "pottytrain", "safety", "strollers", "toys"]

for category in categories[[1, 12]]
    reg_name = "1_100_100_100_$(category)_regs.csv"
    txt_name = "1_100_100_100_$(category)_item_names.txt"

    samples = CSV.read(joinpath(amazon_dir, reg_name), DataFrame, header = 0) |>
        eachrow .|>
        Vector .|>
        skipmissing .|>
        collect

    N = length(readlines(joinpath(amazon_dir, txt_name)))
    M = length(samples)

    ϕ05 = x -> box_cox(x, 0.5)
    _dϕ05 = x -> box_cox_derivative(x, 0.5)
    ϕ15 = x -> box_cox(x, 1.5)
    _dϕ15 = x -> box_cox_derivative(x, 1.5)
    Random.seed!(123)
    L_init = rand(Wishart(N, diagm(ones(N)))) / N
    kpp_05 = KPP(ϕ05, _dϕ05, L_init)
    kpp_15 = KPP(ϕ15, _dϕ15, L_init)

    ## RM vs. LPSD
    #
    n_iter = 5000
    Random.seed!(123)
    res_rm_05 = sgd_rm(kpp_05, samples, n_iter; η = 0.1, ρ = 0.0)
    res_lpsd_05 = sgd_lpsd(kpp_05, samples, n_iter; η = 0.1, ρ = 0.0, α = 2, α′ = -1)
    res_rm_15 = sgd_rm(kpp_15, samples, n_iter; η = 0.1, ρ = 0.0)
    res_lpsd_15 = sgd_lpsd(kpp_15, samples, n_iter; η = 0.1, ρ = 0.0, α = 2, α′ = -1)

    Random.seed!(123)
    logliks_rm_05 = @showprogress [approx_mean_loglik(samples, res_rm_05[1][i]; N_iter_mf = 3) for i in 1:n_iter]
    logliks_lpsd_05 = @showprogress [approx_mean_loglik(samples, res_lpsd_05[1][i]; N_iter_mf = 3) for i in 1:n_iter]
    logliks_rm_15 = @showprogress [approx_mean_loglik(samples, res_rm_15[1][i]; N_iter_mf = 3) for i in 1:n_iter]
    logliks_lpsd_15 = @showprogress [approx_mean_loglik(samples, res_lpsd_15[1][i]; N_iter_mf = 3) for i in 1:n_iter]

    res_dict = Dict(:rm_05 => Dict(:res => res_rm_05, :loglik => logliks_rm_05),
                    :lpsd_05 => Dict(:res => res_lpsd_05, :loglik => logliks_lpsd_05),
                    :rm_15 => Dict(:res => res_rm_15, :loglik => logliks_rm_15),
                    :lpsd_15 => Dict(:res => res_lpsd_15, :loglik => logliks_lpsd_15))

    jldsave("../outputs/DKPP_res_amazon_$category.jld2"; res_dict)

    fig = Figure();
    ax = Axis(fig[1, 1]; xlabel = "Iteration", ylabel = "Mean log-likelihood (approx.)");
    lines!(ax, logliks_lpsd_05,
        color = colorschemes[:tol_bright][1], label = "Localized PS (λ = 0.5)");
    lines!(ax, logliks_lpsd_15,
        color = colorschemes[:tol_bright][1], linestyle = :dash, label = "Localized PS (λ = 1.5)");
    lines!(ax, logliks_rm_05,
        color = colorschemes[:tol_bright][2], label = "Ratio Matching (λ = 0.5)");
    lines!(ax, logliks_rm_15,
        color = colorschemes[:tol_bright][2], linestyle = :dash, label = "Ratio Matching (λ = 1.5)");
    axislegend(ax, position = :rb);
    save("../outputs/DKPP_loglik_amazon_$category.pdf", fig)
end

#meantime_lpsd = (res_lpsd_05[2][end] + res_lpsd_15[2][end]) / (2 * n_iter)
#meantime_rm = (res_rm_05[2][end] + res_rm_15[2][end]) / (2 * n_iter)
#txt_lpsd = @sprintf("Localized PS: %1.3e s/itr", meantime_lpsd)
#txt_rm = @sprintf("Ratio Matching: %1.3e s/itr", meantime_rm)
