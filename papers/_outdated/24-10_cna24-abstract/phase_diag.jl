begin # imports
    using ProgressBars;
    using LinearAlgebra;
    using Statistics;
    using Zygote;
    using JLD;
    using GLMakie;
    using CairoMakie;
end 
begin # long imports
    using Revise;
    push!(LOAD_PATH, pwd());
    using RefractiveIsing;
end

begin # set parameters
    J = [1, 6, 12]
    R = [5, 10, 15]
    npoints = [size(R)[1], size(J)[1], 500, 500]
    H = LinRange(-3, 3, npoints[3]);
    B = LinRange(0,  6, npoints[4]);
    fname = "data/article/phase_diag.jld";
end;

# begin # run the eigenvalue calculation

#     # phase 1 - stable phase
#     # phase 2 - oscillatory phase
#     phases = zeros(Int8, npoints...)
#     thold = 1e-6
    
#     it = collect(Base.product(1:npoints[1], 1:npoints[2], 1:npoints[3], 1:npoints[4]))
#     Threads.@threads for (i, j, k, l) in ProgressBar(it)
#         # define the system parameters
#         pars = RefracParams(J[j], H[k], B[l], R[i])
#         # Calculate fixed point
#         fpx = mf_fixed_point(pars)
#         # Calculate Jacobian and diagonalize
#         res = eigen(Zygote.jacobian(p -> mf_step(p, pars), fpx)[1])
#         # Phase rules
#         if sum(abs.(res.values) .< (1-thold)) == R[i]-1
#             phases[i,j,k,l] = 1
#         else
#             phases[i,j,k,l] = 2
#         end
#     end
# end

# begin # save the data
#     save(fname, "phases", phases, "H", H, "B", B, "J", J, "R", R)
# end

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


# backend choice
GLMakie.activate!()
CairoMakie.activate!()

begin # spread over subplots
    f = Figure(size=(1000,1000))
    # ax = Axis(f[1,1], xlabel=L"\beta", ylabel=L"\theta",
    # xticks = 0:1:9, yticks = -2:0.5:2, limits=(0, 4, -2, 2))
    # heatmap!(B, H, phases[3,3,:,:]')
    for Jix in 1:3
        for Rix in 1:3
            ax = Axis(f[Rix,Jix], xlabel=L"\beta", ylabel=L"\theta",
            xticks = 0:1:maximum(B), yticks = minimum(H):1:maximum(H), 
            limits=(minimum(B), maximum(B), minimum(H), maximum(H)),
            title = "J = $(J[Jix]), R = $(R[Rix])")
            heatmap!(B, H, phases[Rix,Jix,:,:]')
        end
    end
    #save("figures/article/phase_diag_spread.pdf", f)
    f
end

begin # condensed in two subplots
    f = Figure(size=(700,350))
    colors = [:blue, :orange, :green]
    fsize = 20
    lwidth = 2
    msize = 15

    # fixed J
    Jix = 1
    Jcutoffs = [3.7, 1.65, 2.15] # cutoffs for the second-to-first phase transition
    ax1 = Axis(f[1,1], title="J = $(J[Jix])", xlabel=L"\beta", ylabel=L"\theta",
    xticks = 0:1:maximum(B)-1, yticks = minimum(H):1:maximum(H), 
    limits=(minimum(B), maximum(B), minimum(H), maximum(H)))
    for Rix in [1,3,2]
        borders = find_borders(Matrix(phases[Rix,Jix,:,:]'))
        bounds = []
        for x in borders[1:end]
            push!(bounds, (B[x[1]], H[x[2]]))
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
        lines!(ax1, bounds[1:boundix], label="R = $(R[Rix])", color=colors[Rix], 
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
    ax2 = Axis(f[1,2], title="R = $(R[Rix])", xlabel=L"\beta", #ylabel=L"\theta",
    xticks = 0:1:maximum(B), yticks = minimum(H):1:maximum(H), 
    limits=(minimum(B), maximum(B), minimum(H), maximum(H)))
    for Jix in 1:3
        borders = find_borders(Matrix(phases[Rix,Jix,:,:]'))
        bounds = []
        for x in borders[1:end]
            push!(bounds, (B[x[1]], H[x[2]]))
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
        lines!(ax2, bounds[1:boundix], label="J = $(J[Jix])", color=colors[Jix], 
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

    f
end
save("figures/article/phase_diag.pdf", f)