include("../discrete-maps.jl")

# Vanilla Ising model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

function vanilla_map(x, p)
    return [0.5 + 0.5*tanh(p[:β]*(p[:J]*x[1]+p[:I]-p[:θ]))]   
end

"""
VanillaIMF(x0, J, θ, β)

Returns a DiscreteMap object of the regular Ising model.

# Arguments
- `x0::Real`: initial condition
- `J::Real`: synaptic coupling
- `θ::Real`: external field
- `β::Real`: inverse temperature

# Returns
- `DiscreteMap`: object representing the regular Ising model
"""
function VanillaIMF(x0::Vector, J::Real, θ::Real, β::Real, I::Real) :: DiscreteMap
    # create parameters
    p = Dict(:J => J, :θ => θ, :β => β, :I => I)
    # create featurizer
    ft = (x, p, pc) -> x
    nft = 1
    # precompute stuff
    pc = Dict()
    return DiscreteMap{1}("Vanilla Ising",
        vanilla_map, x0, x0, p, ft, nft, pc)
end
