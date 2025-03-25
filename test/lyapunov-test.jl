include(srcdir("lyapunov.jl"));


# Henon map tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# DiscreteMap implementation of the Henon ma``
function HenonMap(x0::Vector{<:Real}, a::Real, b::Real) :: DiscreteMap
    # check x0 is a vector of length 2
    @assert length(x0) == 2 "IC x0 must be a vector of length 2"
    # update function
    _henon_map(x, p) = [1 - p[:a]*x[1]*x[1] + x[2], p[:b]*x[1]]
    return DiscreteMap{2}("Henon Map", 
        _henon_map, x0, x0,
        Dict(:a => a, :b => b),
        (x, p, pc) -> x, 2, Dict()
    )
end
# test the Henon map implementation
@testset "Henon map tests" begin

    # test initialization
    s = HenonMap([0.0, 0], 0, 0)
    @test s.x == [0.0, 0.0]
    @test length(s.x) == 2
    s = HenonMap([0.0, 0], 1, 1)
    @test s.x == [0.0, 0.0]

    # test fixed point
    a, b = rand(), 1
    s = HenonMap([1/sqrt(a), b/sqrt(a)], a, b)
    @test step(s) ≈ s.x0
    step!(s)
    @test s.x ≈ s.x0     
   
end


# test Lyapunov spectrum
@testset "Lyapunov" begin
    
    dx = MMatrix{2,2, Float64}([1 0; 0 1])
    s = HenonMap([0.1, 0.1], 1.4, 0.3)
    equi_stp = 1000
    L, conv_info = lyap_spectrum(s, dx)
    @test length(L) == 2

end



# test Lyapunov spectrum

# s = HenonMap([0.1, 0.1], 1.4, 0.3)

# equi_stp = 1000
# meas_stp = 1000

# begin # run the simulation    
#     # equilibration
#     forward!(s, equi_stp)
#     # sample trajectory
#     traj = trajectory!(s,meas_stp)
#     # initial perturbation
#     dx = MMatrix{2,2, Float64}([1 0; 0 1])
#     L, conv_info = lyap_spectrum(s, dx)
# end;