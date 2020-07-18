using Gadfly
using DataFrames
using StatsBase
using Distributions
using Colors
using ColorSchemes

function bc_upd(ϵ::Float64, α::Float64, τ::Float64, n_agents::Int, n_steps::Int)
    bc_ar = Array{Float64,2}(undef, n_agents, n_steps + 1)
    bc_ar[:, 1] = sort(rand(n_agents), rev=true)
    @views for j in 1:n_steps, i in 1:n_agents
        @inbounds bc_ar[i, j + 1] = (1 - α)*mean(bc_ar[:, j][findall(abs.(bc_ar[:, j] .- bc_ar[:, j][i]) .< ϵ)]) + α*τ
    end
    return bc_ar
end

res = bc_upd(.1, .2, .7, 50, 30)

df₀ = res |> rotr90 |> DataFrame
rename!(df₀, [Symbol("$i") for i in 1:size(df₀, 2)])
df = df₀ |> stack
df[!, :steps] = repeat(0:size(df₀, 1) - 1, outer=size(df₀, 2))

palette = [ get(ColorSchemes.viridis, i) for i in range(0, length=size(df₀, 2), stop=1) ]

plot(df, x=:steps, y=:value, color=:variable, Geom.point, Geom.line,
    Coord.cartesian(xmax=size(df₀, 1)),
    Guide.xlabel("Time"),
    Scale.color_discrete_manual(palette...),
    Guide.ylabel("Opinion"),
    Guide.title("ϵ = .1 / α = .2 / τ = .7"),
    Theme(key_position=:none, point_size=1.5pt))

# different approach

abstract type Agent end

mutable struct TruthSeeker <: Agent
    ϵ::Float64
    α::Float64
    op::Float64
end

opinions(pop::Array{TruthSeeker,1}) = Float64[ pop[i].op for i in eachindex(pop) ]

peers(pop::Array{TruthSeeker,1}) = Bool[ abs(pop[j].op - pop[i].op) ≤ pop[j].ϵ for i in eachindex(pop), j in eachindex(pop) ] # the peers for agent j are in the j-th column

function update!(pop::Array{TruthSeeker,1}, τ::Float64)
    prs = peers(pop)
    ops = opinions(pop)
    soc = sum(ops .* prs, dims=1) ./ sum(prs, dims=1)
    @inbounds for i in eachindex(pop)
        pop[i].op = pop[i].α*τ + (1 - pop[i].α)*soc[i]
    end
end

function run_model(ϵ::Float64, α::Float64, τ::Float64, numb_agents::Int, numb_updates::Int)
    pp = [ TruthSeeker(ϵ, α, rand()) for _ in 1:numb_agents ]
    res = zeros(numb_agents, numb_updates + 1)
    res[:, 1] = opinions(pp)
    @inbounds for i in 1:numb_updates
        update!(pp, τ)
        res[:, i + 1] = opinions(pp)
    end
    return res
end

function run_model(ϵ::Float64, α::Float64, τ::Float64, numb_agents::Int)
    pp = [ TruthSeeker(ϵ, α, rand()) for _ in 1:numb_agents ]
    res = Array{Float64,1}[]
    push!(res, opinions(pp))
    while true
        update!(pp, τ)
        push!(res, opinions(pp))
        if all(abs.(res[end] .- res[end - 1]) .< 10^-5)
            break
        end
    end
    return reduce(hcat, res)
end

res = run_model(.1, .2, .7, 50, 30)
res = run_model(.1, .2, .7, 50)

# NB `run_model(.1, .2, .7, 50, 30)` is the equivalent of `bc_upd(.1, .2, .7, 50, 30)`; the first has fewer allocations and is much faster (relatively speaking, given that both are very fast):
using BenchmarkTools
@btime run_model(.1, .2, .7, 50, 30);
@btime bc_upd(.1, .2, .7, 50, 30);

df₀ = res |> rotr90 |> DataFrame
srt = sortperm(Vector(df₀[1, :]))
permutecols!(df₀, srt)
rename!(df₀, [Symbol("$i") for i in 1:size(df₀, 2)])
df = df₀ |> stack
df[!, :steps] = repeat(0:size(df₀, 1) - 1, outer=size(df₀, 2))

palette = [ get(ColorSchemes.viridis, i) for i in range(0, length=size(df₀, 2), stop=1) ]

plot(df, x=:steps, y=:value, color=:variable, Geom.point, Geom.line,
    Coord.cartesian(xmax=size(df₀, 1)),
    Guide.xlabel("Time"),
    Scale.color_discrete_manual(palette...),
    Guide.ylabel("Opinion"),
    Guide.title("ϵ = .1 / α = .2 / τ = .7"),
    Theme(key_position=:none, point_size=1.5pt))

# first extension: random-per-agent ϵ and α (but we can still specify upper and lower bounds for both)

function run_model(ϵ_inf::Float64, ϵ_sup::Float64, α_inf::Float64, α_sup::Float64, τ::Float64, numb_agents::Int, numb_updates::Int)
    pp = [ TruthSeeker(ϵ_inf < ϵ_sup ? rand(Uniform(ϵ_inf, ϵ_sup)) : ϵ_inf, α_inf < α_sup ? rand(Uniform(α_inf, α_sup)) : α_inf, rand()) for _ in 1:numb_agents ]
    res = zeros(numb_agents, numb_updates + 1)
    res[:, 1] = opinions(pp)
    @inbounds for i in 1:numb_updates
        update!(pp, τ)
        res[:, i + 1] = opinions(pp)
    end
    return res
end

function run_model(ϵ_inf::Float64, ϵ_sup::Float64, α_inf::Float64, α_sup::Float64, τ::Float64, numb_agents::Int)
    pp = [ TruthSeeker(ϵ_inf < ϵ_sup ? rand(Uniform(ϵ_inf, ϵ_sup)) : ϵ_inf, α_inf < α_sup ? rand(Uniform(α_inf, α_sup)) : α_inf, rand()) for _ in 1:numb_agents ]
    res = Array{Float64,1}[]
    push!(res, opinions(pp))
    while true
        update!(pp, τ)
        push!(res, opinions(pp))
        if all(abs.(res[end] .- res[end - 1]) .< 10^-5)
            break
        end
    end
    return reduce(hcat, res)
end

res = run_model(.0, .25, .0, .5, .7, 50, 75)
res = run_model(.0, .25, .0, .5, .7, 50)

df₀ = res |> rotr90 |> DataFrame
srt = sortperm(Vector(df₀[1, :]))
permutecols!(df₀, srt)
rename!(df₀, [Symbol("$i") for i in 1:size(df₀, 2)])
df = df₀ |> stack
df[!, :steps] = repeat(0:size(df₀, 1) - 1, outer=size(df₀, 2))

plot(df, x=:steps, y=:value, color=:variable, Geom.point, Geom.line,
    Coord.cartesian(xmax=size(df₀, 1)),
    Guide.xlabel("Time"),
    Scale.color_discrete_manual(palette...),
    Guide.ylabel("Opinion"),
    Guide.title("ϵᵢ = U(0, .25)/ αᵢ = U(0, .5) / τ = .7"),
    Theme(key_position=:none, point_size=1.5pt))

