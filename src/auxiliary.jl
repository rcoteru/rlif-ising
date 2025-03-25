using LinearAlgebra: qr, diag, norm
using Random

"""
    simplex_orthobasis(n::Int) :: Matrix

    Returns a basis of n-1 orthogonal vectors
    of the n-dimensional standard simplex.
"""
function simplex_orthobasis(n::Int) :: Matrix
    Q = zeros(n, n)
    # the first vector is the normalized vector of ones
    Q[:,1] = ones(n)/sqrt(n)
    # create the rest of the vectors using Gram-Schmidt
    for i in 2:n
        Q[:,i] = rand(n)
        for j in 1:i-1
            Q[:,i] -= Q[:,j]'*Q[:,i]*Q[:,j]
        end
        Q[:,i] = Q[:,i]/norm(Q[:,i])
    end
    return Q[:,2:end]
end

"""
    idx_combinations(x::AbstractVector) :: Vector

    Returns a vector of all possible combinations
    of indices of the elements of x.
"""
function idx_combinations(x::AbstractVector)
    Nvals(x) = 1:length(x)
    return collect(Base.product(Nvals.(x)...))
end


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

function exponential_weights(Q::Int, α::Real)
    return exp.(-α*(1:Q))/exp.(-α)
end