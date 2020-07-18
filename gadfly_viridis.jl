using Gadfly
using DataFrames
using StatsBase
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

const ϵ = 0.1
const α = 0.2
const τ = 0.7

res = bc_upd(ϵ, α, τ, 50, 30)

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
    Guide.title("ϵ = $ϵ / α = $α / τ = $τ"),
    Theme(key_position=:none, point_size=1.5pt))

# different approach

abstract type Agent end

mutable struct TruthSeeker <: Agent
    ϵ::Float64
    α::Float64
    op::Float64
end

const ϵ = .1
const α = .2
const τ = .7
const numb_agents = 50

ppl = [ TruthSeeker(ϵ, α, rand()) for _ in 1:numb_agents ]

opinions(pop::Array{TruthSeeker,1}) = Float64[ pop[i].op for i in 1:numb_agents ]

peers(pop::Array{TruthSeeker,1}) = Bool[ abs(pop[j].op - pop[i].op) ≤ pop[j].ϵ for i in 1:numb_agents, j in 1:numb_agents ] # the peers for agent j are in the j-th column

function update!(pop::Array{TruthSeeker,1})
    prs = peers(pop)
    ops = opinions(pop)
    soc = sum(ops .* prs, dims=1) ./ sum(prs, dims=1)
    @inbounds for i in 1:numb_agents
        pop[i].op = pop[i].α*τ + (1 - pop[i].α)*soc[i]
    end
end

function run_model(pop::Array{TruthSeeker,1}, numb_updates::Int)
    pp = deepcopy(pop)
    res = zeros(numb_agents, numb_updates + 1)
    res[:, 1] = opinions(pp)
    @inbounds for i in 1:numb_updates
        update!(pp)
        res[:, i + 1] = opinions(pp)
    end
    return res
end

function run_model(pop::Array{TruthSeeker,1})
    pp = deepcopy(pop)
    res = Array{Float64,1}[]
    push!(res, opinions(pp))
    while true
        update!(pp)
        push!(res, opinions(pp))
        if all(isapprox.(res[end], res[end - 1], atol=10^-5))
            break
        end
    end
    return reduce(hcat, res)
end

res = run_model(ppl, 30)
res = run_model(ppl)

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
    Guide.title("ϵ = $ϵ / α = $α / τ = $τ"),
    Theme(key_position=:none, point_size=1.5pt))
