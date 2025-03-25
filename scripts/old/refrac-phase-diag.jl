using DrWatson
@quickactivate "BioIsing"

include(srcdir("discrete-maps/ising.jl"));
include(srcdir("misc.jl"));

using LinearAlgebra: eigen
using ProgressBars

begin # set parameters
    Js = [1, 6, 12];
    Rs = [5, 10, 15];
    θs = range(-3, 3, length=501);
    βs = range( 0, 6, length=501);
    fname = datadir("uniform/refrac-phase-diag.jld2");
end;

begin # run the eigenvalue calculation
    phases = zeros(Int64, length(Rs), length(Js), length(θs), length(βs));
    Threads.@threads for (i,j,k,l) in ProgressBar(idx_combinations([Rs, Js, θs, βs]))    
        # threshold
        thold = 1e-6
        # generate fixed point
        x0 = _active_refrac_ising_fxp(Js[j], θs[k], βs[l], Rs[i])
        # create the system
        s = DiscreteIsingMF(x0, Js[j], θs[k], βs[l], Rs[i]; 
            active=true, refrac=true, lif=false)
        # check which phase it is
        if sum(abs.(eigen(jac(s)).values) .< (1-thold)) == Rs[i]
            # phase 1 - stable phase
            phases[i,j,k,l] = 1
        else
            # phase 2 - oscillatory phase
            phases[i,j,k,l] = 2
        end
    end
    # save the data
    save(fname, "phases", phases, "θs", θs, "βs", βs, "Js", Js, "Rs", Rs)
end

begin # load the data
    phases = load(fname, "phases")
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


using CairoMakie
begin # condensed in two subplots
    f = Figure(size=(700,350))
    colors = [:blue, :orange, :green]
    fsize = 20
    lwidth = 2
    msize = 15

    # fixed J
    Jix = 1
    Jcutoffs = [3.7, 1.65, 2.15] # cutoffs for the second-to-first phase transition
    ax1 = Axis(f[1,1], title="J = $(Js[Jix])", xlabel=L"\beta", ylabel=L"\theta",
    xticks = 0:1:maximum(βs)-1, yticks = minimum(θs):1:maximum(θs), 
    limits=(minimum(βs), maximum(βs), minimum(θs), maximum(θs)))
    for Rix in [1,3,2]
        borders = find_borders(Matrix(phases[Rix,Jix,:,:]'))
        bounds = []
        for x in borders[1:end]
            push!(bounds, (βs[x[1]], θs[x[2]]))
        end
        # find first point over the cutoff
        boundix = 0
        for (ix, b) in enumerate(bounds)
            if b[1] < Jcutoffs[Rix]
                boundix = ix
                break
            end
        end
        # plot the first part continuously
        lines!(ax1, bounds[1:boundix], label="R = $(Rs[Rix])", color=colors[Rix], 
            linewidth=lwidth, linestyle = :dash)
        # plot the second part dashed
        lines!(ax1, bounds[boundix:end], color=colors[Rix], 
            linewidth=lwidth)
        # mark the transition point
        scatter!(ax1, [bounds[boundix][1]], [bounds[boundix][2]], color=colors[Rix], 
        markersize = msize)
    end
    text!(3, -2, text = "(1)", align = (:center, :center), fontsize = fsize)
    text!(3.5, 1, text = "(2)", align = (:center, :center), fontsize = fsize)
    axislegend(ax1)

    # fixed R
    Rix = 1
    Rcutoffs = [3.7, 1.65, 0.85] # cutoffs for the second-to-first phase transition
    ax2 = Axis(f[1,2], title="R = $(Rs[Rix])", xlabel=L"\beta", #ylabel=L"\theta",
    xticks = 0:1:maximum(βs), yticks = minimum(θs):1:maximum(θs), 
    limits=(minimum(βs), maximum(βs), minimum(θs), maximum(θs)))
    for Jix in 1:3
        borders = find_borders(Matrix(phases[Rix,Jix,:,:]'))
        bounds = []
        for x in borders[1:end]
            push!(bounds, (βs[x[1]], θs[x[2]]))
        end
        # find first point over the cutoff
        boundix = 0
        for (ix, b) in enumerate(bounds)
            if b[1] < Rcutoffs[Jix]
                boundix = ix
                break
            end
        end
        # plot the first part continuously
        lines!(ax2, bounds[1:boundix], label="J = $(Js[Jix])", color=colors[Jix], 
            linewidth=lwidth, linestyle = :dash)
        # plot the second part dashed
        lines!(ax2, bounds[boundix:end], color=colors[Jix], 
            linewidth=lwidth)
        # mark the transition point
        scatter!(ax2, [bounds[boundix][1]], [bounds[boundix][2]], color=colors[Jix], 
            markersize = msize)
    end
    text!(3, -2, text = "(1)", align = (:center, :center), fontsize = fsize)
    text!(3.5, 1, text = "(2)", align = (:center, :center), fontsize = fsize)
    axislegend(ax2)
    
    # hide axis labels
    ax2.yticksvisible = false
    ax2.yticklabelsvisible = false
    colgap!(f.layout, 5)
    save(plotsdir("uniform/refrac-phase-diag.pdf"), f)
    save(plotsdir("uniform/refrac-phase-diag.png"), f)
    f
end