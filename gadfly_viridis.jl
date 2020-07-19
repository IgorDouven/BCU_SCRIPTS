using DataFrames
using Gadfly
using Colors
using ColorSchemes
using GLM
using Compose

using Distributed
addprocs()
@everywhere using StatsBase
@everywhere using Distributions

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

@everywhere mutable struct TruthSeeker
    ϵ::Float64
    α::Float64
    op::Float64
end

eps_vals(pop::Array{TruthSeeker,1}) = Float64[ pop[i].ϵ for i in eachindex(pop) ]
alp_vals(pop::Array{TruthSeeker,1}) = Float64[ pop[i].α for i in eachindex(pop) ]
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

# we can already some interesting statistics: record the agents' ϵ and α values, and measure their accuracy, and look whether the former are good predictors of the latter

function conc(ar::AbstractArray, numb_simulations::Int)
    x = ar[:, :, 1]
    for i in 2:numb_simulations
        w = vcat(x, ar[:, :, i])
        x = w
    end
    return x
end

sse(ar::Array{Float64,2}, τ::Float64) = sum((ar .- τ).^2, dims=2)

function run_model_stats(ϵ_inf::Float64, ϵ_sup::Float64, α_inf::Float64, α_sup::Float64, τ::Float64, numb_agents::Int)
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
    out = reduce(hcat, res)
    return hcat(sse(out, τ), eps_vals(pp), alp_vals(pp))
end

function run_sims(ϵ_inf::Float64, ϵ_sup::Float64, α_inf::Float64, α_sup::Float64, numb_agents::Int, numb_simulations::Int)
    v = Array{Float64,3}(undef, numb_agents, 4, numb_simulations)
    @inbounds for i in 1:numb_simulations
        t = rand()
        out = run_model_stats(ϵ_inf, ϵ_sup, α_inf, α_sup, t, numb_agents)
        v[:, :, i] = hcat(out, fill(t, numb_agents))
    end
    return conc(v, numb_simulations)
end

res = run_sims(.0, .25, .0, .5, 50, 1000)
res = DataFrame(SSE = res[:, 1], ϵ = res[:, 2], α = res[:, 3], τ = res[:, 4])
res[!, :ϵ] = standardize(ZScoreTransform, res[!, :ϵ], dims=1)
res[!, :α] = standardize(ZScoreTransform, res[!, :α], dims=1)

lm(@formula(SSE ~ α + ϵ + τ), res)

# further types of agents

@everywhere mutable struct FreeRider
    ϵ::Float64
    α::Float64
    op::Float64
end

@everywhere mutable struct Campaigner
    ϵ::Float64
    α::Float64
    op::Float64
end

@everywhere function ppl(ϵ_tr::Float64, α_tr::Float64, ϵ_fr::Float64, α_fr::Float64, ϵ_ca::Float64, α_ca::Float64, numb_ts::Int, numb_fr::Int, numb_ca::Int, ρ::Float64)
    ts = [ TruthSeeker(ϵ_tr , α_tr, rand()) for _ in 1:numb_ts ]
    fr = [ FreeRider(ϵ_fr, α_fr, rand()) for _ in 1:numb_fr ]
    ca = [ Campaigner(ϵ_ca, α_ca, ρ) for _ in 1:numb_ca ]
    un = Array{Union{TruthSeeker,FreeRider,Campaigner},1}[ts, fr, ca]
    return reduce(vcat, un)
end

@everywhere opinions(pop::Array{Union{TruthSeeker,FreeRider,Campaigner},1}) = Float64[ pop[i].op for i in eachindex(pop) ]

@everywhere function peers(pop::Array{Union{TruthSeeker,FreeRider,Campaigner},1})
    numb_ts = count(x->typeof(x) == TruthSeeker, pop)
    numb_fr = count(x->typeof(x) == FreeRider, pop)
    return Bool[ abs(pop[j].op - pop[i].op) ≤ pop[j].ϵ for i in eachindex(pop), j in 1:(numb_ts + numb_fr) ]
end

@everywhere function update!(pop::Array{Union{TruthSeeker,FreeRider,Campaigner},1}, τ::Float64)
    prs = peers(pop)
    ops = opinions(pop)
    soc = sum(ops .* prs, dims=1) ./ sum(prs, dims=1)
    @inbounds for i in eachindex(pop)
        if typeof(pop[i]) == TruthSeeker
            pop[i].op = pop[i].α*τ + (1 - pop[i].α)*soc[i]
        elseif typeof(pop[i]) == FreeRider
            pop[i].op = soc[i]
        end
    end
end

function run_model(ϵ_tr::Float64, α_tr::Float64, ϵ_fr::Float64, α_fr::Float64, ϵ_ca::Float64, α_ca::Float64, numb_ts::Int, numb_fr::Int, numb_ca::Int, τ::Float64, ρ::Float64, numb_updates::Int)
    numb_agents = numb_ts + numb_fr + numb_ca
    pp = ppl(ϵ_tr, α_tr, ϵ_fr, α_fr, ϵ_ca, α_ca, numb_ts, numb_fr, numb_ca, ρ)
    res = zeros(numb_agents, numb_updates + 1)
    res[:, 1] = opinions(pp)
    @inbounds for i in 1:numb_updates
        update!(pp, τ)
        res[:, i + 1] = opinions(pp)
    end
    return res
end

@everywhere function run_model(ϵ_tr::Float64, α_tr::Float64, ϵ_fr::Float64, α_fr::Float64, ϵ_ca::Float64, α_ca::Float64, numb_ts::Int, numb_fr::Int, numb_ca::Int, τ::Float64, ρ::Float64)
    pp = ppl(ϵ_tr, α_tr, ϵ_fr, α_fr, ϵ_ca, α_ca, numb_ts, numb_fr, numb_ca, ρ)
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

const numb_ts = 50
const numb_fr = 10
const numb_ca = 5

res = run_model(.1, .2, .1, .0, .0, .0, numb_ts, numb_fr, numb_ca, .7, .9)

df₀ = res' |> DataFrame
rename!(df₀, [Symbol("$i") for i in 1:size(df₀, 2)])
df = df₀ |> stack
df[!, :steps] = repeat(0:size(df₀, 1) - 1, outer=size(df₀, 2))
df[!, :agent] = vcat(fill("Truth seeker", size(df₀, 1)*numb_ts), fill("Free rider", size(df₀, 1)*numb_fr), fill("Campaigner", size(df₀, 1)*numb_ca))

palette = [colorant"darksalmon", colorant"gold", colorant"darkseagreen"]

Gadfly.plot(df, x=:steps, y=:value, group=:variable, color=:agent,
        Geom.point, Geom.line, Scale.color_discrete_manual(palette..., order=[3, 2, 1]),
        Coord.cartesian(xmax=maximum(df.steps)),
        Guide.xlabel("Update"), Guide.ylabel("Opinion"),
        Guide.colorkey(title="Group", labels=["Campaigners", "Free riders", "Truth seekers"]),
        yintercept=[.7], Geom.hline(style=:dot, color=colorant"grey"),
        Guide.annotation(compose(context(), Compose.text(maximum(df.steps) - 1, 0.735, "τ"), fontsize(11pt))))

# conversion

@everywhere begin
    const ϵ_fr = .0
    const ϵ_ca = .0
    const α_tr = .0
    const α_fr = .0
    const α_ca = .0
    const τ    = .0
    const ρ    = .8

    const numb_ts = 50
    const numb_fr = 0
end

@everywhere function perc_conv(eps::Float64, nc::Int)
    res = run_model(eps, α_tr, ϵ_fr, α_fr, ϵ_ca, α_ca, numb_ts, numb_fr, nc, τ, ρ)
    return sum(≈(ρ, atol=1e-3), res[1:numb_ts, end])
end

@everywhere function perc_sim(eps::Float64, nc::Int)
    return Float64[ perc_conv(eps, nc) for _ in 1:100 ] |> mean
end

function conv_sims()
    perc_mat = Matrix{Float64}(undef, 50, 50)
    @inbounds for j in 1:50
        perc_mat[:, j] = pmap(i->perc_sim(i/100, j), 1:50)
    end
    return perc_mat
end

res = conv_sims()

function xlabelname(x)
    n = x/100
    return "$n"
end

function ylabelname(x)
    n = abs(x - 50)
    return "$n"
end

ticks = collect(0:10:50)

Gadfly.spy(rotl90(res),
    Guide.ColorKey(title="Converted\ntruth seekers"),
    Guide.xlabel("ϵ"), Guide.ylabel("Campaigners"),
    Guide.title("ρ = $ρ"), # given that alpha is zero, the value of tau doesn't matter here
    Guide.xticks(ticks=ticks), Scale.x_continuous(labels=xlabelname),
    Guide.yticks(ticks=ticks), Scale.y_continuous(labels=ylabelname),
    Scale.ContinuousColorScale(p->get(ColorSchemes.viridis, p), minvalue=0, maxvalue=50),
    Theme(grid_color=colorant"white", panel_fill="white"))
