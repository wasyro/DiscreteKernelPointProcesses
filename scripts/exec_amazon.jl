using LinearAlgebra
using Distributions
using Random
using DataFrames
using CSV
using Query

include("$(@__DIR__)/dpp_utils.jl")
include("$(@__DIR__)/dpp_experimenter.jl")
LinearAlgebra.BLAS.set_num_threads(Sys.CPU_THREADS - 1) 
# load Amazon Baby Registry data
amazon_dir = joinpath("$(@__DIR__)", "..", "data", "AmazonBabyRegistry")
categories = ["apparel", "bath", "bedding", "carseats", "decor", "diaper",
              "feeding", "furniture", "gear", "gifts", "health", "media",
              "moms", "pottytrain", "safety", "strollers", "toys"]

df_result = DataFrame()
n_exp = 30 # number of experiments
for category in categories
    reg_name = "1_100_100_100_$(category)_regs.csv"
    txt_name = "1_100_100_100_$(category)_item_names.txt"

    samples = CSV.read(joinpath(amazon_dir, reg_name), DataFrame, header = 0) |>
        eachrow .|>
        Vector .|>
        skipmissing .|>
        collect

    N = length(readlines(joinpath(amazon_dir, txt_name)))
    M = length(samples)

    ### Default hyperparameters
    outdir = joinpath("$(@__DIR__)", "..", "output", "amazon", "default", category)
    mkpath(outdir)

    Random.seed!(1234)
    results_amazon_wishart_default = map(1:n_exp) do i
        Linit = initializer(N, init = :wishart)
        outdir_i = joinpath(outdir, string(i))
        return experimenter(Linit, samples, outdir = outdir_i, max_iter = Int(1e5))
    end
    global df_result = vcat(
        df_result,
        summarize_to_df(results_amazon_wishart_default,
                        dict_cols = Dict(:category => Symbol(category), :params => "default"))
       )

    ### Accelerated hyperparameters
    outdir = joinpath("$(@__DIR__)", "..", "output", "amazon", "accelerated", category)
    mkpath(outdir)

    Random.seed!(1234)
    results_amazon_wishart_accelerated = map(1:n_exp) do i
        Linit = initializer(N, init = :wishart)
        outdir_i = joinpath(outdir, string(i))
        return experimenter(Linit, samples, outdir = outdir_i, max_iter = Int(1e5),
                            η = 0.1, ρ = 1.3, accelerate_steps = 5)
    end
    global df_result = vcat(
        df_result,
        summarize_to_df(results_amazon_wishart_accelerated,
                        dict_cols = Dict(:category => Symbol(category), :params => "accelerated"))
        )
end

outdir = joinpath("$(@__DIR__)", "..", "output")
CSV.write(joinpath(outdir, "amazon_results.csv"), df_result)


# load results from .csv

df_result = DataFrame()
for c in categories
    results_def = [load(joinpath(outdir, "amazon/default/$(c)/$(i)/results.jld2")) for i in 1:n_exp]
    df_def = summarize_to_df(results_def,
                             dict_cols = Dict(:category => c, :params => "default"),
                             str_keys = true)
    results_acc = [load(joinpath(outdir, "amazon/accelerated/$(c)/$(i)/results.jld2")) for i in 1:n_exp]
    df_acc = summarize_to_df(results_acc,
                             dict_cols = Dict(:category => c, :params => "accelerated"),
                             str_keys = true)
    df_result = vcat(df_result, df_def, df_acc)
end
