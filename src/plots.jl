using RecipesBase
using Random: GLOBAL_RNG
export SamplingPlot

# function xs_ys_barline(walls, ys)
#     @argcheck length(walls) == length(ys) + 1
#     xs_ret = Float64[]
#     ys_ret = Float64[]
# 
#     push!(xs_ret, walls[1])
#     push!(ys_ret, 0)
# 
#     for i in 1:length(walls)-1
#         x_left = walls[i]
#         x_right = walls[i+1]
#         y = ys[i]
#         push!(xs_ret, x_left)
#         push!(ys_ret, y)
#         push!(xs_ret, x_right)
#         push!(ys_ret, y)
#     end
#     push!(xs_ret, walls[end])
#     push!(ys_ret, 0)
# 
#     xs_ret, ys_ret
# end
function xs_ys_barline(walls, ys)
    @argcheck length(walls) == length(ys) + 1
    xs_ret = Float64[]
    ys_ret = Float64[]

    push!(xs_ret, walls[1])
    push!(ys_ret, 0)

    for i in 1:length(walls)-1
        x_left = walls[i]
        x_right = walls[i+1]
        y = ys[i]
        push!(xs_ret, x_left)
        push!(ys_ret, y)
        push!(xs_ret, x_right)
        push!(ys_ret, y)
        push!(xs_ret, x_right)
        push!(ys_ret, 0)
    end

    xs_ret, ys_ret
end

function spacings(xs)
    xs[2:end] - xs[1:end-1]
end

@recipe function plot(grid::VegasGrid{1})
    # seriestype --> :vline
    # walls = first(grid.boundaries)
    xs = first(grid.boundaries)
    if get(plotattributes, :seriestype, nothing) == :vline
        xs
    else
        ys = 1 ./ spacings(xs)
        xs_ys_barline(xs, ys)
    end
end

@qstruct SamplingPlot(f, domain; neval::Int=100, axis::Int=1)

@recipe function plot(p::SamplingPlot)
    seriestype --> :scatter
    axis = p.axis
    f = p.f; dom = p.domain
    s = draw(GLOBAL_RNG, dom)
    pt = s.position
    x = pt[p.axis]
    y = f(pt) * s.weight
    xs = typeof(x)[]
    ys = typeof(y)[]
    for _ in 1:p.neval
        s = draw(GLOBAL_RNG, dom)
        pt =  s.position
        y = f(x) * s.weight
        x = pt[p.axis]
        push!(xs, x)
        push!(ys, y)
    end
    xs, ys
end
