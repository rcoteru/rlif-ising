using DrWatson
@quickactivate "BioIsing"

begin # load project environment
    include(srcdir("spin-model.jl"));
    include(srcdir("mean-field.jl"));
    include(srcdir("auxiliary.jl"));     
end

using DataStructures: CircularBuffer
using Distributions
using CairoMakie

function dm(m, S, J, β, θ, N) :: Float64
    return -(m+1) + (1-S)*(1+tanh(β*(J*(m+1)/2+θ)))
end

J = 6
β = 1
θ = 0
N = 100
R = 4
dt = 0.001
m = -1

Pn = CircularBuffer{Float64}(Int((R+1)/dt))
for i in 1:Int((R+1)/dt)
    push!(Pn, 0)
end

dm(m, sum(Pn), J, β, θ, N)

traj = []
for _ in 1:30*(R+1)/dt
    m = m + dm(m, sum(Pn), J, β, θ, N)*dt
    pushfirst!(Pn, (m+1)/2)
    push!(traj, m)
end

sum(Pn)

lines(Pn)

lines(traj)