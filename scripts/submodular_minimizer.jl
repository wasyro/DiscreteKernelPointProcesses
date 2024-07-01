"""
    Get a maximizer <c, x> s.t. x ∈ B(f) (base polytope of a submodular function f) given c.
"""
function get_extremepoint(c :: Vector{Float64}, f :: Function)
    N = length(c)
    sort_ids = sortperm(c, rev = true)
    x = zeros(N)
    ids = zeros(Int64, 0)
    f₋ = f(ids)
    for i in 1:N
        push!(ids, sort_ids[i])
        f₊ = f(ids)
        x[sort_ids[i]] = f₊ - f₋
        f₋ = f₊
    end
    return x
end

"""
    Get a minimum-norm-point in an affine hull
"""
function affine_minimizer(B :: Array{Float64})
    n = size(B, 2)
    α = (B' * B \ ones(n)) / sum(B' * B \ ones(n))
    return B * α, α
end


"""
    Execute minimum-norm-point algorithm
"""
function minimum_norm_point(f :: Function, N :: Int; tol :: Float64 = 1e-5, debug :: Bool = false)::Vector{Int}
    # [step 0] starting with an initial extreme point
    x₀ = get_extremepoint(Float64.(N:-1:1), f)
    B = zeros(N, 1)
    B .= x₀
    x̂ = x₀

    λ = Float64[1.0]

    if debug println("===Start Iterations===") end
    while true
        # [step 1] find a point that minimizes <x̂, x> s.t. x ∈ B(f)
        if debug println("x̂ = $(x̂)\n") end

        x = get_extremepoint(-x̂, f)

        if debug
            println("===Got Extreme Point===")
            println("x = $(x)\n")
            println("===Check Convergence===")
            println("val = $(abs(x̂' * (x - x̂))), tol = $(tol)\n")
        end

        ## check convergence
        if abs(x̂' * (x - x̂)) <= tol
            if debug println("===Converged===") end
            return S = findall(x -> x <= 0, x̂)
        else
            if debug println("===Not Converged===\n") end
            B = hcat(B, x)
            push!(λ, 1.0)
        end

        j = 1
        y = zeros(N)
        while true
            if j > size(B, 1) + 1
                throw(ErrorException("Algorithm Did Not Converge"))
            end

            if debug
                println("===Start Minor Iterations===")
                println("x̂ = $(x̂), λ = $λ")

                println("===Affine Minimizer===")
                println("B = $B")
            end
            # [step 2] find the minimum-norm-point in the affine hull spaned by the points in B(f)
            y, α = affine_minimizer(B)
            if debug println("y = $y, α = $α\n") end

            # check the minimum-norm-point is an interior of the convex hull
            if all(α .>= 0)
                if debug println("===Minimum-norm-point Is in the Convex Hull===\n") end
                break
            end

            # [step 3]
            inds_α₋ = findall(x -> x < 0, α)
            θ = minimum(i -> λ[i] / (λ[i] - α[i]), inds_α₋)
            x̂ = θ * y + (1 - θ) * x̂
            λ = θ * α .+ (1 - θ) * λ
            #λ = [θ * α[i] + (1 - θ) * λ[i] for i in length(α)]
            if debug println("λ = $λ\n") end

            #inds_λ₊ = findall(x -> x > 0, λ)
            inds_λ₊ = findall(x -> x > 0 + tol, λ)
            B = hcat([B[:, i] for i in inds_λ₊]...)
            λ = λ[inds_λ₊]

            j += 1
        end
        x̂ = y
    end
end
