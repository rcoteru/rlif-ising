using DrWatson
@quickactivate "BioIsing"

using CairoMakie

# Firing probability / sigmoid function
begin
    x = -5:0.01:5
    sigmoid(x, β) = (exp(β*x))/(2*cosh(β*x))
    f = Figure(size=(400, 300))
    ax = Axis(f[1, 1], xticks=([0]), yticks=([0, 0.5, 1]), 
        xgridvisible=false, ygridvisible=true,
        xlabel=L"I_{t}", title=L"p_t(s_i=1)")
    hidespines!(ax, :t, :r, :l)
    lines!(ax, x, sigmoid.(x, 10), color=:red, label=L"\beta=10")
    lines!(ax, x, sigmoid.(x, 1), color=:blue, label=L"\beta=1")
    ax.xticks = ([0], [L"\theta"])
    
    axislegend(ax, position = :rc)
    save(plotsdir("sigmoid.pdf"), f)
    display(f)
end

# Complete model schematic
begin
    J = 0.2
    θ = 0.4
    α = 0.2
    R = 2

    # spike function
    spike(x, A, α, x0) = x < x0 ? 0 : A*exp(-α*(x-x0));

        
    spike_train1 = [1.2,3,4.8]

    npoints = 100
    x = range(0, 10, length=npoints)
    y = zeros(npoints)
    for sx in spike_train1
        y += spike.(x, J, α, sx)
    end

    psp = x[argmax(y.>θ)]

    y[argmax(y.>θ)+1:end] .= 0

    # refracted spikes
    spike_train2 = [5.2, 6.6]

    # refracted spikes
    spike_train3 = [7.8, 9.2]

    for sx in spike_train3
        y += spike.(x, J, α, sx)
    end

    with_theme(theme_latexfonts()) do

        fig = Figure()
    
        ax1 = Axis(fig[1:5,2],
        limits=(0,10,nothing,0.6))
        hidedecorations!(ax1)
        hidespines!(ax1, :t, :r, :l)
    
        vspan!(ax1, [psp], [psp+R],
            color=(:red, 0.2))
    
        lines!(ax1, x, y, color=(:blue,1))
        band!(ax1, x, zeros(npoints), y, color=(:blue,0.1))
        
        hlines!(ax1, [θ], color=:black, linestyle=:dash)
        vlines!(ax1, [psp], color=:red)
    
        text!(0, θ*1.01, text = L"\theta", fontsize=20)
        
        text!(psp+R/2, θ*1.25, text = "Refractory\nperiod", 
        align=(:center, :center), color=:red,
        fontsize=20)
    
        Label(fig[1:5,1], L"I_t", 
            fontsize=20, rotation=pi/2)
    
    
        ax2 = Axis(fig[6,2], xlabel=L"t", 
        xlabelsize=20, limits=(0,10,nothing,nothing))
        hidedecorations!(ax2, label=false)
        hidespines!(ax2, :t, :r, :l)
    
        spikewidth = 2.5
        vlines!(ax2, spike_train1, color=:blue, linewidth=spikewidth)
        vlines!(ax2, spike_train2, color=:red, linewidth=spikewidth)
        vlines!(ax2, spike_train3, color=:blue, linewidth=spikewidth)
    
        Label(fig[6,1], "Input", 
            fontsize=20, rotation=pi/2)
    
        save(plotsdir("schematic.pdf"), fig)
        display(fig)
    end
end;




