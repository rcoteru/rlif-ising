# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Lyapunov exponents functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

include("discrete-maps.jl");

using LinearAlgebra: norm, qr, diag

function lyap_step!(dm::DiscreteMap, dx::MMatrix, S::Vector)
    # update the state and the perturbation
    dx = dstep!(dm, dx)
    # QR decomposition
    Q, R = qr(dx)
    # update the perturbation vectors
    dx .= Q
    # update the Lyapunov exponent sum
    S += log.(abs.(diag(R)))
end

function lyap_spectrum(s:: DiscreteMap, dx :: MMatrix;
    tol :: Real = 1e-6, itermax :: Integer = Integer(1E5)
    ) :: Tuple{Vector, Tuple{Vector, Integer, Bool}}
    # vector dimension
    npert = size(dx, 2) # number of perturbations
    # trace sum for the Lyapunov exponent
    S = @MVector zeros(Float64, npert)
    prev_S = @MVector zeros(Float64, npert)
    # ensure the perturbation is normalized
    [dx[:,j] = dx[:,j]/norm(dx[:,j]) for j in 1:npert]
    # iterate
    error, conv_flag, conv_it = 0, false, 0
    for i in 1:itermax
        # update the state and the perturbation
        dx = dstep!(s, dx)
        # QR decomposition
        Q, R = qr(dx)
        # update the perturbation vectors
        dx .= Q
        # update the Lyapunov exponent sum
        S += log.(abs.(diag(R)))
        # calculate the Lyapunov exponents
        # check for convergence
        error = abs.(S./i - prev_S)
        if maximum(abs.(error)) < tol
            conv_it, conv_flag = i, true
            break
        end
        prev_S = copy(S./i)
    end
    if !conv_flag 
        @warn "Lyapunov spectrum did not converge"
        conv_it=itermax 
    end
    return (S./conv_it, (error, conv_it, conv_flag))
end
