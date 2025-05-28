##########################################
####
#### Julia Functions for ESAG Distribution
####
##########################################

###
### Set-up 
###

## parallel computing
using Distributed
addprocs()

## load required packages
@everywhere using LinearAlgebra
@everywhere using Distributions
@everywhere using Statistics
@everywhere using StaticArrays
using QuadGK # for mdmo integration
using Cubature # for CDF integration
using Base.Threads # for parallel computing

###
### PDF of ESAG(+)
###

## evaluates M_{d-1}(t) for ESAG density
function mdmo(d::Int, t::Float64)
    f(x) = x^(d-1) * exp(-(x-t)^2 / 2)
    result, error = quadgk(f, 0, Inf)
    return (2*pi)^(-1/2) * result
end

## evaluates ESAG pdf; allows for logarithms and positive orthant truncation
function dESAG(y, mu, Vinv; logm = false, trunc = false, nc = 1)
    d = length(mu)
    ytVinvy = dot(y, Vinv * y) # y'*Vinv*y
    ytmu = dot(y, mu)
    if (logm == false) # no logarithm
        if (trunc == false) # no truncation
            result = (2*pi)^(-(d-1)/2)/ytVinvy^(d/2)*exp((ytmu^2/ytVinvy - dot(mu, mu))/2) * mdmo(d, ytmu/sqrt(ytVinvy))
        else # truncation to positive orthant
            if all(y .>= 0) # observation is in positive orthant
                result = (2*pi)^(-(d-1)/2)/ytVinvy^(d/2)*exp((ytmu^2/ytVinvy - dot(mu, mu))/2) * mdmo(d, ytmu/sqrt(ytVinvy))/nc
            else # observation is outside support
                result = missing
            end
        end
    else # logarithm
        if (trunc == false) # no truncation
            result = -(d-1)/2*log(2*pi) - d/2*log(ytVinvy) + (ytmu^2/ytVinvy - dot(mu, mu))/2 + log(mdmo(d, ytmu/sqrt(ytVinvy)))
        else # truncation to positive orthant
            if all(y .>= 0) # observation is in positive orthant
                result = -(d-1)/2*log(2*pi) - d/2*log(ytVinvy) + (ytmu^2/ytVinvy - dot(mu, mu))/2 + log(mdmo(d, ytmu/sqrt(ytVinvy))) - log(nc)
            else # observation is outside support
                result = missing
            end
        end
    end
    return result
end

## fast normalizing constant calculation
@everywhere function normConstant(mu, V, nSamp)
    samps = rESAG(nSamp, mu, V) # simulate from ESAG(μ, V)
    result = sum(all(samps .>= 0, dims=2))/nSamp # Monte Carlo integration
    if result != 0
        return result   
    else
        return 0.0000001 # for numerical stability
    end
end

## alternative parallelization approach for normalizing constant
function NCP(mu_list, V_list, nSamp)
    n = length(mu_list)
    results = Vector{Float64}(undef, n)
    Threads.@threads for i in 1:n
        results[i] = normConstant(mu_list[i], V_list[:, :, i], nSamp)
    end
    return results
end

###
### ESAG Random Draws
###

## fast simulation using StaticArrays and vectorized programming
## NOTE: if (d != 3), then all SVectors and SMatrices must be manually changed
@everywhere function rESAG(n, mu, V)
    mu_st = SVector{3}(mu)
    V_st = SMatrix{length(mu_st), length(mu_st)}(V)
    L = cholesky(Hermitian(V_st)).U'
    X = zeros(n, length(mu_st))
    for i in 1:n 
        X[i, :] = normalize(mu_st + L * @SVector(randn(length(mu_st))))
    end
    return X
end

## randomly samples from an ESAG⁺ density
function rESAG_trunc(n, mu, V)
    ## check if constraints are satisfied
    if !isapprox(V * mu, mu) || !isapprox(det(V), 1)
        println("ESAG conditions are not met.")
        return
    end

    ## truncation to positive orthant ; rejection sampling
    samples = Matrix{Float64}(undef, 0, length(mu))
    while size(samples, 1) < n # until we have enough samples
        X = rand(MvNormal(mu, Hermitian(V)), 1)'
        while !all(X .>= 0) # while outside the positive orthant
            X = rand(MvNormal(mu, Hermitian(V)), 1)'
        end
        samples = vcat(samples, X/norm(X))
    end
    return samples
end

###
### CDF of ESAG+
###

## evaluates CDF of ESAG+ at a point
function cdfESAG(test, mu, Vinv; parallel = true)
    if parallel
        test = vec(test)
        upper_bounds = vec(test)
        lower_bounds = zeros(length(test))
        function truncatedPDF_v(test, result = Vector{Float64,1})
            for i in 1:length(result)
                result[i] = dESAG(test[:, i], mu, Vinv, trunc = true)
            end
        end
        return hcubature_v(truncatedPDF_v, lower_bounds, upper_bounds, reltol = 1)
    else
        test = vec(test)
        upper_bounds = vec(test)
        lower_bounds = zeros(length(test))
        function truncatedPDF(test)
            return dESAG(test, mu, Vinv, trunc = true)
        end
        return hcubature(truncatedPDF, lower_bounds, upper_bounds, reltol = 1)
    end
end

###
### Functions to get V from μ and γ 
###

## compute radial parameters r from gamma
function get_r(gamma, d)
    r = zeros(d-2)
    for i in 1:(d-2)
        r[i] = norm(gamma[i])
    end
    return r
end

## get eigenvalues using formula
function get_eigenvalues(r, d)
    lambda_vec = ones(d)
    lambda_vec[1] = prod((r .+ 1) .^ (d .- (1:(d-2)) .- 1)) ^ (-1 / (d-1))
    for j in 2:(d-1)
        lambda_vec[j] = lambda_vec[1] * prod(r[1:(j-1)] .+ 1)
    end
    return lambda_vec
end

## compute longitude angles theta from gamma
function get_theta(gamma, d)
    theta = zeros(d-2)
    theta[1] = atan(gamma[1][2], gamma[1][1])
    for j in 2:(d-2)
        if gamma[j][j]^2 + gamma[j][j+1]^2 == 0
            theta[j] = 0
        elseif gamma[j][j+1] >= 0
            theta[j] = acos(gamma[j][j] / sqrt(gamma[j][j]^2 + gamma[j][j+1]^2))
        else
            theta[j] = -acos(gamma[j][j] / sqrt(gamma[j][j]^2 + gamma[j][j+1]^2))
        end
    end
    return theta
end

## compute latitude angles phi from gamma
function get_phi(gamma, d)
    phi = Vector{Float64}[]
    push!(phi, [NaN]) # place-holder for non-existent phi[1]
    for j in 2:(d-2)
        group_phi = zeros(j-1)
        for k in 1:(j-1)
            if sum(gamma[j][k:(j+1)].^2) == 0
                group_phi[k] = 0
            else
                group_phi[k] = acos(gamma[j][k] / sqrt(sum(gamma[j][k:(j+1)].^2)))
            end
        end
        push!(phi, group_phi)
    end
    return phi
end

## computes orthonormal basis for step (ii) in Yu & Huang (2024)
function get_OB(mu, d)
    u = zeros(d, d)
    u[1, 1] = -mu[2]
    u[1, 2] = mu[1]
    
    for j in 2:(d-1)
        for i in 1:j
            u[j, i] = mu[i] * mu[j+1]
        end
        u[j, j+1] = -sum(mu[1:j].^2)
    end
    u[d, :] = mu

    ## edge case below eq. (2.6) in Yu & Huang (2024)
    for j in 1:d
        if all(iszero, u[j, :])
            u[j, j] = 1
        end
    end
    u_SA = SMatrix{3, 3}(u)

    ## compute orthonormal basis
    xi_tilde = zeros(d, d)
    for j in 1:d
        xi_tilde[:, j] = u_SA[j, :] / norm(u_SA[j, :], 2)
    end

    return xi_tilde
end

## compute rotation matrix R
function get_R(theta, phi, d)
    R = Matrix(1.0*I, d-1, d-1)
    phi_unlist = reverse(vcat(phi[2:end]...))
    phi_counter = 1
    for m in 1:(d-3)
        tmpR = get_Rstar(1, 2, theta[d-m-1], d)
        for j in 1:(d-m-2)
            tmpR *= get_Rstar(j+1, j+2, phi_unlist[phi_counter], d)
            phi_counter += 1
        end
        R *= tmpR
    end
    R *= get_Rstar(1, 2, theta[1], d)
    return R
end

## compute the (d-1)-dimensional plane rotation matrices
function get_Rstar(j, k, theta, d)
    Rstar = Matrix(1.0*I, d-1, d-1)
    Rstar[j, j] = cos(theta)
    Rstar[j, k] = -sin(theta)
    Rstar[k, j] = sin(theta)
    Rstar[k, k] = cos(theta)
    return Rstar
end

## compute the eigenvectors to characterize V
function get_eigenvectors(mu, theta, phi, d)
    xi_tilde = get_OB(mu, d)
    R = get_R(theta, phi, d)
    xi = xi_tilde[:, 1:(d-1)] * R
    xi = hcat(xi, xi_tilde[:, d])
    return xi
end

## compute V given μ and γ; potentially return ξ, too
function get_V(mu, gamma, d; get_xi = false)
    r = get_r(gamma, d)
    lambda = get_eigenvalues(r, d)
    theta = get_theta(gamma, d)
    phi = get_phi(gamma, d)
    xi = get_eigenvectors(mu, theta, phi, d)
    xi_SA = SMatrix{3, 3}(xi) # remember to change all SVector and SMatrix objects if (d != 3)
    V = zeros(d, d)
    for j in 1:d
        V += lambda[j] * xi_SA[:, j] * xi_SA[:, j]'
    end
    if (get_xi == false)
        return V
    else
        return V, xi
    end
end

## construct γ with a vector of values
function getGamma(vec, d)

    ## intialize structure
    gamma = [zeros(j+1) for j in 1:(d-2)]
    
    ## populate gamma
    counter = 1
    for (index, subarray) in enumerate(gamma)
        for inner_index in eachindex(subarray)
            gamma[index][inner_index] = vec[counter]
            counter += 1
        end
    end
    return gamma
end

###
### Evaluate ESAG mode
###

## construct a Fibonacci lattice on positive orthant of sphere
function fiblat(N)
    golden_ratio = (1 + sqrt(5)) / 2
    i = 0:(N-1)
    theta = 2*pi*i/golden_ratio
    phi = acos.(1 .- 2 .* (i ./ N))
    x = cos.(theta) .* sin.(phi)
    y = sin.(theta) .* sin.(phi)
    z = cos.(phi)
    points = [x y z]
    valid_indices = (0 .<= x .<= 1) .& (0 .<= y .<= 1) .& (0 .<= z .<= 1)
    valid_indices_array = findall(valid_indices) 
    positive_orthant_points = points[valid_indices_array, :] 

    return positive_orthant_points
end

## grid search for mode
function ESAG_mode(mu, Vinv, nc, FibLat; trunc = false)
    modeVal = 0
    curr_mode = zeros(length(mu))
    for i in 1:size(FibLat, 1)
        densVal = dESAG(FibLat[i, :], mu, Vinv, logm = true, trunc = trunc, nc = nc)
        if densVal > modeVal
            modeVal = copy(densVal)
            curr_mode = FibLat[i, :]
        end
    end
    return curr_mode, modeVal
end 

###
### Evaluate Mean Direction
###

## function to compute mean direction given μ and V
function MCmean(mu, V, num_samples; trunc = false)
    d = length(mu) # dimension of the distribution
    samples = zeros(d, num_samples) # matrix to store generated samples
    if trunc 
        for i in 1:num_samples
            samples[:, i] = vec(rESAG_trunc(1, mu, V))
        end
    else
        for i in 1:num_samples
            samples[:, i] = vec(rESAG(1, mu, V))
        end
    end

    ## compute the mean of the samples
    mean_sample = mean(samples, dims = 2)
    return mean_sample
end

###
### Estimate Mean Composition
###

## function to estimate composition given μ and V
function estComp(mu, V, num_samples; trunc = false)
    d = length(mu) # dimension of the distribution
    samples = zeros(d, num_samples) # matrix to store generated samples
    if trunc 
        for i in 1:num_samples
            samples[:, i] = (vec(rESAG_trunc(1, mu, V))).^2
        end
    else
        for i in 1:num_samples
            samples[:, i] = (vec(rESAG(1, mu, V))).^2
        end
    end

    ## compute the mean of the samples
    mean_sample = mean(samples, dims = 2)
    return mean_sample
end

###
### Compute χ² measure of distance for model comparison
###

## chi-squared measure of distance 
function CS_dist(y1, y2)
    d = length(y1)
    r = zeros(d)
    for j in 1:d
        if (y1[j] == 0 && y2[j] == 0)
            r[j] = 0
        else
            r[j] = (y1[j]/sum(y1) - y2[j]/sum(y2))^2/(y1[j]/sum(y1) + y2[j]/sum(y2))
        end
    end
    return sqrt(2*d)*(sum(r))^(1/2)
end