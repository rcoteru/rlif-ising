using DrWatson, Test
@quickactivate "BioIsing"

# Run test suite
println("Starting tests")
ti = time()

@testset "BioIsing" begin
    @testset "Auxiliary" begin include("auxiliary-test.jl") end
#    @testset "Mean Fields" begin include("mean-field-test.jl") end
    @testset "Spin Models" begin include("spin-model-test.jl") end
#    @testset "Lyapunov" begin include("lyapunov-test.jl") end
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")