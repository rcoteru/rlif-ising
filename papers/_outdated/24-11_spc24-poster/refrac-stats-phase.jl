using DrWatson
@quickactivate "BioIsing"

include(srcdir("discrete-maps/ising.jl"));
include(srcdir("misc.jl"));

using LinearAlgebra: eigen
using ProgressBars
using CairoMakie

begin
    J = 6
    R = 4
    θs = range(-2, 2, length=201);
    βs = range( 0, 6, length=201);
    x0 = [1, zeros(R)...]
    equi_stp = 50000
    meas_stp = 20000
    fname = datadir("uniform/refrac-full.jld2")
end;

trajs = zeros(Float64, length(βs), 1000, 5)
Threads.@threads for (i,) in ProgressBar(idx_combinations([βs])) 
    s = DiscreteIsingMF(x0, J, θs[61], βs[i], R; 
        active=true, refrac=true, lif=false)
    forward!(s, equi_stp)
    trajs[i,:,:] = trajectory!(s, 1000, ft=true)
end


begin # run the eigenvalue calculation
    stats = zeros(Float64, length(θs),length(βs), 10);
    entropy = zeros(Float64, length(θs),length(βs), 3);
    phases = zeros(Int64, length(θs), length(βs));
    Threads.@threads for (i,j) in ProgressBar(idx_combinations([θs, βs]))    
        # create the system
        s = DiscreteIsingMF(x0, J, θs[i], βs[j], R; 
            active=true, refrac=true, lif=false)
        # equilibrate
        forward!(s, equi_stp)
        # measure statistics
        stats[i,j,:] = stats!(s, meas_stp, ft=true)
        # measure entropy production
        entropy[i,j,:] = active_refrac_ising_entropy!(s, meas_stp)
        # place the system in the fixed point
        s.x[:] .= s.pc[:fxp]
        # check which phase it is
        thold = 1e-6
        if sum(abs.(eigen(jac(s)).values) .< (1-thold)) == R
            # phase 1 - stable phase
            phases[i,j] = 1
        else
            # phase 2 - oscillatory phase
            phases[i,j] = 2
        end
    end
    # save the data
    save(fname, "θs", θs, "βs", βs, "stats", stats, 
        "entropy", entropy, "phases", phases)
end

begin # load the data
    phases = load(fname, "phases")
    stats = load(fname, "stats")
    entropy = load(fname, "entropy")
end;

# function to get sorted border coordinates
function find_borders(image:: Matrix) :: Array{Tuple{Int, Int}, 1}
    rows, cols = size(image)
    border_coords = []
    # Check each element if it's a 1 and has a neighboring 0
    for i in 2:rows-1
        for j in 2:cols-1
            if image[i, j] == 1
                neighbors = [
                image[i-1, j],image[i+1, j+1],
                image[i+1, j],image[i+1, j-1], 
                image[i, j-1],image[i-1, j+1], 
                image[i, j+1],image[i-1, j-1]]
                if 2 in neighbors  # If any neighbor is 2, it's part of the border
                    push!(border_coords, (i, j))
                end
            end
        end
    end
    # Sort the border coordinates, first point is the one with the highest x
    start_idx = argmax([p[1] for p in border_coords])
    sorted_coords = [border_coords[start_idx]]
    available_coords = copy(border_coords)
    deleteat!(available_coords, start_idx)
    while !isempty(available_coords)
        current_point = last(sorted_coords)
        # Find the closest point
        dists = [norm(current_point .- p) for p in available_coords]
        closest_point = argmin(dists)
        push!(sorted_coords, available_coords[closest_point]) # save the closest point
        deleteat!(available_coords, closest_point)  # Mark as visited
    end
    return sorted_coords
end


begin # condensed in two subplots
    f = Figure(size=(700,700))
    colors = [:blue, :orange, :green]
    fsize = 20
    lwidth = 2
    msize = 15

    cutoffs = 1.65 # cutoffs for the second-to-first phase transition
    
    function draw_border(ax)
        borders = find_borders(Matrix(phases[:,:]'))
        bounds = []
        for x in borders[1:end]
            push!(bounds, (βs[x[1]], θs[x[2]]))
        end
        # find first point over the cutoff
        boundix = 0
        for (ix, b) in enumerate(bounds)
            if b[1] < cutoffs
                boundix = ix
                break
            end
        end
        # plot the first part continuously
        lines!(ax, bounds[1:boundix], color=:black, 
            linewidth=lwidth, linestyle = :dash)
        # plot the second part dashed
        lines!(ax, bounds[boundix:end], color=:black, 
            linewidth=lwidth)
        # mark the transition point
        scatter!(ax, [bounds[boundix][1]], [bounds[boundix][2]], color=:black, 
        markersize = msize)
    end
    
    cmap = :OrRd

    # mean firing rate
    ax = Axis(f[1,2], xlabel=L"\beta",
    xticks = 0:1:maximum(βs), yticks = minimum(θs):0.5:maximum(θs),
    yaxisposition=:right, yticklabelsvisible = false,
    limits=(minimum(βs), maximum(βs), minimum(θs), 0.5))
    hm = heatmap!(ax, βs, θs, stats[:,:,2]', 
        colormap = cmap, colorrange = (0, 0.4),
        )
    Colorbar(f[1,1], hm, labelsize=fsize, flipaxis=false, label=L"\langle N_0 \rangle")
    draw_border(ax)

    # std firing rate
    ax = Axis(f[2,2],
    xticks = 0:1:maximum(βs), yticks = minimum(θs):0.5:maximum(θs), 
    yaxisposition=:right, yticklabelsvisible = false,
    xaxisposition=:top, xticklabelsvisible = false,
    limits=(minimum(βs), maximum(βs), minimum(θs), 0.5))
    hm = heatmap!(ax, βs, θs, stats[:,:,7]', 
        colormap = cmap, colorrange = (0, 0.4))
    Colorbar(f[2,1], hm, labelsize=fsize, flipaxis=false, label=L"std(N_0)")
    draw_border(ax)

    # transition 
    ax = Axis(f[3,1:2], xlabel=L"\beta", ylabel=L"|N-N_{fxp}|",
    limits=(minimum(βs), 3, nothing, nothing), xticks = 0:0.5:maximum(βs))

    θidx = 61
    npoints = 1000
    for i in 1:1:length(βs)
        scatter!(ax, fill(βs[i], npoints), 
        trajs[i,1:npoints,5], color=:red, markersize=2)
    end
    text!(0.5, 0.7, text = "θ=-0.8", align = (:center, :center), fontsize = fsize)
    lines!(ax, βs, stats[θidx,:,5], color=:black, label="mean")


    # forwards conditional entropy
    ax = Axis(f[1,3], xlabel=L"\beta", ylabel=L"\theta",
    xticks = 0:1:maximum(βs)-1, yticks = minimum(θs):0.5:maximum(θs), 
    limits=(minimum(βs), maximum(βs), minimum(θs), 0.5))
    hm = heatmap!(ax, βs, θs, entropy[:,:,1]', 
        colormap = cmap, colorrange = (0, 0.3))
    Colorbar(f[1,4], hm, labelsize=fsize, label=L"\langle S_{t|t-1}\rangle")
    draw_border(ax)

    # backwards conditional entropy
    ax = Axis(f[2,3], ylabel=L"\theta",
    xticks = 0:1:maximum(βs), yticks = minimum(θs):0.5:maximum(θs), 
    xaxisposition=:top, xticklabelsvisible=false,
    limits=(minimum(βs), maximum(βs), minimum(θs), 0.5))
    hm = heatmap!(ax, βs, θs, entropy[:,:,2]', 
        colormap = cmap, colorrange= (0,2))
    Colorbar(f[2,4], hm, labelsize=fsize, label=L"\langle S_{t-1|t}\rangle")
    draw_border(ax)

    # steady state entropy
    ax = Axis(f[3,3], xlabel=L"\beta", ylabel=L"\theta",
    xticks = 0:1:maximum(βs), yticks = minimum(θs):0.5:maximum(θs), 
    limits=(minimum(βs), maximum(βs), minimum(θs), 0.5))
    hm = heatmap!(ax, βs, θs, entropy[:,:,2]'-entropy[:,:,1]', 
        colormap = cmap, colorrange = (0, 2))
    Colorbar(f[3,4], hm, labelsize=fsize, label=L"\langle\sigma_t\rangle")
    draw_border(ax)
    

    # hm = heatmap!(ax1, βs, θs, entropy[:,:,3]', 
    #     colormap = :OrRd, colorrange = (0, 0.3))
    # Colorbar(f[1,2], hm, label="Entropy production", labelsize=fsize)

    colgap!(f.layout, 4)
    rowgap!(f.layout, 12)

    save(plotsdir("papers/spc/refrac-stats-phase.png"), px_per_unit=5,  f)
    save(plotsdir("papers/spc/refrac-stats-phase.pdf"), px_per_unit=5, f)

    f

end
