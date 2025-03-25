# Struct and functions for discrete maps

using Statistics: mean, std
using LinearAlgebra: eigen
using StaticArrays
import Zygote

struct DiscreteMap{N}
    name::String            # name of the model
    f ::Function            # f(x, p) -> x
    x ::MVector{N, Float64} # current state
    x0::SVector{N, Float64} # initial state
    p ::Dict{Symbol, Any}   # parameters
    ft::Function            # featurizer d(x, pc) -> x
    nft::Int                # number of features
    pc::Dict{Symbol, Any}   # precomputed stuff
end

function Base.show(io::IO, dm::DiscreteMap)
    println(io, "$(dm.name): dim=$(length(dm.x))")
end

function step(dm::DiscreteMap) :: Vector
    return dm.f(dm.x, dm.p)
end

function step!(dm::DiscreteMap)
    dm.x[:] = dm.f(dm.x, dm.p)
end

function jac(dm::DiscreteMap) :: Matrix{Float64}
    return Zygote.jacobian(dm.f, dm.x, dm.p)[1]
end

function eigvals(dm::DiscreteMap) :: Vector
    return eigen(jac(dm)).values
end

function eigvecs(dm::DiscreteMap) :: Matrix
    return eigen(jac(dm)).vectors
end

function dstep!(dm::DiscreteMap, dx::MMatrix) :: MMatrix
    dx .= jac(dm)*dx
    step!(dm)
    return dx
end

function state(dm::DiscreteMap) :: Vector
    return dm.x
end

function features(dm::DiscreteMap) :: Vector
    return dm.ft(dm.x, dm.p, dm.pc)
end

function set_state!(dm::DiscreteMap, x) :: Nothing
    dm.x[:] .= x
end

function reset_state!(dm::DiscreteMap) :: Nothing
    dm.x[:] .= dm.x0
end

function forward!(dm::DiscreteMap, nsteps::Int) :: Nothing
    for _ in 1:nsteps step!(dm) end
end

function trajectory!(dm::DiscreteMap, nsteps::Int; ft::Bool = false) :: Matrix{Float64}
    if ft == false || isnothing(dm.ft)
        traj = zeros(Float64, nsteps, length(dm.x0))
        for i in 1:nsteps
            step!(dm)
            traj[i,:] = dm.x
        end
        return traj
    end
    traj = zeros(Float64, nsteps, dm.nft)
    for i in 1:nsteps
        step!(dm)
        traj[i,:] = features(dm)
    end
    return traj
end

function stats!(dm::DiscreteMap, nsteps::Int; ft::Bool = false) :: Vector{Float64}
    traj = trajectory!(dm, nsteps, ft=ft)
    return [mean(traj, dims=1)[:]..., std(traj, dims=1)...]
end