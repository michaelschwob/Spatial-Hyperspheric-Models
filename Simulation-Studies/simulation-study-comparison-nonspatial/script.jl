###############################################
####
#### Simulation Study to Compare ESAG and ESAG+ 
#### Models w/ ESAG+ generated data
####
###############################################

###
### Set-up
###

using Random, Distributions, Plots, DataFrames
using Distances, Kronecker
using ProgressBars, JLD2
using Plots.PlotMeasures
using IterTools
using Base.Threads

Random.seed!(512)
include("../../esag_functions.jl")
include("mcmc.jl")
include("mcmc_base.jl")
nMCMC = 150000
nBurn = Int(0.2*nMCMC)
nKeep = 1000
nSamp = 10000
ngridpoints = 40
trunc_flag = true

## link function for mean direction μ
function link(x)
    return log(1 + exp(x))
end

## transform to range function
function ttr(vec::Vector{Float64}, a::Float64, b::Float64)
    min_val = minimum(vec)
    max_val = maximum(vec)
    
    ## normalize to [0, 1]
    normalized_vec = (vec .- min_val) ./ (max_val - min_val)
    
    ## scale to [a, b]
    scaled_vec = a .+ (b - a) .* normalized_vec
    
    return scaled_vec
end

## directional standardization function ; Scealy & Wood (2019), Yu & Huang (2024+)
function dirStandard(vec::Vector{Float64})
    min_val = minimum(vec)
    max_val = maximum(vec)
    
    ## normalize to [0, 1]
    normalized_vec = (vec .- min_val) ./ (max_val - min_val) .+ 1
    
    return normalized_vec
end

## standardize grid covariates using scaling of original covariates
function dirStandard_grid(X_grid_vec, vec::Vector{Float64})
    min_val = minimum(vec)
    max_val = maximum(vec)
    
    ## normalize to original covariates' scale
    normalized_X_grid_vec = (X_grid_vec .- min_val) ./ (max_val - min_val) .+ 1
    
    return normalized_X_grid_vec
end

###
### Simulate Parameters & Data 
###

## sizes and dimensions
n = 100
d = 3
p = 3
q = 1

## locations, covariates, and coefficients
locs = rand(Uniform(0, 1), n, 2) # study domain
spatial_idx = 1:n
X = [ttr(abs.(locs[:, 1] .- 0.5).^(1.2), 0.5, 1.0) [norm(locs[i, :]) for i in 1:n]] # west-easting ; southwest-northeasting
X = [ones(n) X]
[X[:, i] = dirStandard(X[:, i]) for i in 2:p]

## regression coefficients ; simulate with baseline regression coefficients
B = [2.0 2*0.8 2*0.5; 2.5 2.3 0.5; 2.2 1.1 3.3].*0.5

## load same parameter values from first simulation study (w/ eta)
@load "simVars.jld2"

## create grid across spatial domain
x = collect(LinRange(0, 1, ngridpoints))
y = collect(LinRange(0, 1, ngridpoints))
grid = vec(collect(IterTools.product(x, y)))
grid = hcat([p[1] for p in grid], [p[2] for p in grid])
n_grid = size(grid, 1)
locs_aug = [locs ; grid]
n_aug = size(locs_aug, 1)

## simulate phis
phi_vec = [0.25, 0.2, 0.15]

## pre-compute all R(phi)s for observed locations
phi_list = 0.01:0.01:0.3
distMat = pairwise(Euclidean(), locs_aug') # pairwise Euclidean distance matrix
R_list = zeros(n, n, length(phi_list))
for k in 1:length(phi_list)
    for i in 1:n
        for j in 1:n
            R_list[i, j, k] = exp(-distMat[i, j]/phi_list[k])
        end
    end
end

## get the true R(phis) for observed and gridded locations
phiIdx_vec = [25, 20, 15]
R_aug_true = zeros(n_aug, n_aug, 3)
for k in 1:3
    for i in 1:n_aug
        for j in 1:n_aug
            R_aug_true[i, j, k] = exp(-distMat[i, j]/phi_list[phiIdx_vec[k]])
        end
    end
end

## get the true augmented Sig_eta and the observed Sig_eta
Sig_eta_aug = zeros(n_aug*d, n_aug*d)
for j in 1:d 
    global Sig_eta_aug += Matrix(kronecker(R_aug_true[:, :, j], Am[:, j]*Am[:, j]'))
end
Sig_eta = Sig_eta_aug[1:(n*d), 1:(n*d)]

## simulate η (augmented and observed)
eta_vec_aug = zeros(n_aug*d)
eta_mat_aug = reshape(eta_vec_aug, d, n_aug)
eta_vec = eta_vec_aug[1:(n*d)]
eta_mat = reshape(eta_vec, d, n)

## compute mean
mu = [link.(B*X[i, :] + eta_mat[:, i] + A*z[i, :]) for i in 1:n]
norm_mu = [norm(mu[i]) for i in 1:n]
println("Concentration of mu diagnostics: ")
println(describe(norm_mu))

## compute V
V = zeros(d, d, n)
Vinv = zeros(d, d, n)
xi = zeros(d, d, n)
[(V[:, :, i], xi[:, :, i]) = get_V(mu[i], gammas, d, get_xi = true) for i in 1:n]
[Vinv[:, :, i] = inv(V[:, :, i]) for i in 1:n]

## simulate data from ESAG⁺
Y = zeros(n, d)
for i in 1:n
    Y[i, :] = rESAG_trunc(1, mu[i], V[:, :, i])
end

###
### Save Simulation Variables
###

@save "simulation_vars.jld2" locs X Y R_aug_true Sig_eta_aug eta_mat_aug z A mu gammas V Vinv xi Am

###
### Truth w/ Observations Overlaid
###

## set-up
pgfplotsx()
etaGrid_mat = eta_mat_aug[:, (n+1):n_aug]

###
### Compute true normalized mean directions
###

## covariate grid
X_grid = [ttr(abs.(grid[:, 1] .- 0.5).^(1/2), 0.5, 1.0) [norm(grid[i, :]) for i in 1:n_grid]] # get spatial covariates for grid
X_grid = [ones(n_grid) X_grid] # add intercept
[X_grid[:, i] = dirStandard_grid(X_grid[:, i], X[:, i]) for i in 2:p]
XG = copy(X_grid)

## true mu
mu_list = zeros(d, n_grid)
[mu_list[:, i] = link.(B*XG[i, :] + etaGrid_mat[:, i]) for i in 1:n_grid]

## true V, V_inv, and nc
V_list = zeros(d, d, n_grid)
Vinv_list = zeros(d, d, n_grid)
[V_list[:, :, i] = get_V(mu_list[:, i], gammas, d, get_xi = false) for i in 1:n_grid]
[Vinv_list[:, :, i] = inv(V_list[:, :, i]) for i in 1:n_grid]

## true NMD
nmd_true = zeros(d, n_grid)
println("Computing NMD:")
[nmd_true[:, i] = MCmean(mu_list[:, i], V_list[:, :, i], 50000, trunc = false) for i in ProgressBar(1:n_grid)]
[nmd_true[:, i] = nmd_true[:, i] ./ norm(nmd_true[:, i]) for i in 1:n_grid]

###
### Fit ESAG MCMC
###

println("Nonspatial ESAG MCMC:")
out = mcmc_nonspatial(Y, X, z, spatial_idx, nMCMC, nBurn, nSamp, 1, 5, true, R_list, phiIdx_vec)

###
### ESAG Traces
###

## β trace plots
plots = []
for i in 1:d
    for j in 1:p
        tmpPlot = plot(out["B"][i, j, :], title = "β[$i, $j]")
        push!(plots, tmpPlot)
    end
end
plot(plots..., layout = (d, p), size = (900, 600), left_margin = 10mm, legend = false)
savefig("betaTraces_ESAG.pdf")

## α trace plots
plots = []
for i in 1:q
    tmpPlot = plot(out["alpha"][i, :], title = "α[$i]")
    push!(plots, tmpPlot)
end
plot(plots..., layout = (q, 1), size = (900, 600), left_margin = 10mm, legend = false)
savefig("alphaTraces_ESAG.pdf")

## number of ESS iterations per MCMC iteration
plot(out["ESS_counts"], title = "ESS Counts (ESAG)", left_margin = 10mm, legend = false)
savefig("ESS_ESAG.pdf")

## eta traces 
plots = []
for i in 1:d
    for j in 1:5
        tmpPlot = plot(out["eta"][i, j, :], title = "eta[$i, $j]")
        push!(plots, tmpPlot)
    end
end
plot(plots..., layout = (d, 5), size = (900, 600), left_margin = 10mm, legend = false)
savefig("etaTraces_ESAG.pdf")

## C (coded as Am) traces 
plots = []
for i in 1:d
    for j in 1:d
        tmpPlot = plot(out["Am"][i, j, :], title = "Am[$i, $j]")
        push!(plots, tmpPlot)
    end
end
plot(plots..., layout = (d, d), size = (900, 600), left_margin = 10mm, legend = false)
savefig("AmTraces_ESAG.pdf")

## phi indices
plots = []
for i in 1:d
    tmpPlot = plot(out["phiIdx"][i, :], title = "phi_index[$i]")
    push!(plots, tmpPlot)
end
plot(plots..., layout = (d, 1), size = (900, 600), left_margin = 10mm, legend = false)
savefig("phiTraces_ESAG.pdf")

###
### Predicted η Surface
###

## computes the appropriate exponential decay covariance matrix across given locations
function sigw(distMat, phi)
    sigw = zeros(size(distMat))
    for i in 1:size(distMat, 1)
        for j in 1:size(distMat, 2)
            sigw[i, j] = exp(-distMat[i, j]/phi)
        end
    end
    return sigw
end

etaPred = zeros(d*n_grid)
etaPred_mat = reshape(etaPred, d, n_grid)

###
### Mean Direction Maps
###

## compute normalized mean directions
mu_grid = [link.(B*X_grid[i, :] + etaGrid_mat[:, i]) for i in 1:n_grid]
mu_grid_norm = zeros(d, n_grid)
for i in 1:n_grid
    mu_grid_norm[:, i] = normalize(mu_grid[i])
end

## plot heatmaps of normalized mean directions with observed spherical coordinates
nmd1 = heatmap(x, y, reshape(mu_grid_norm[1, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "Normalized Mean Directions w/ Observations", label = "", widen = false,
    clim = (0, 1), alpha = 0.75, legend = false,
    xlims = extrema(x), ylims = extrema(y), colorbar = :left, colorbar_title = "||μ₁||"
)
scatter!(twinx(), eachcol(locs)..., marker_z = Y[:, 1], xmirror = true, widen = false,
    clim = (0, 1), color = :viridis, msw = 0.25, legend = false, colorbar = :right,
    xlims = extrema(locs[:, 1]), ylims = extrema(locs[:, 2]), colorbar_title = "Y₁"
)
nmd2 = heatmap(x, y, reshape(mu_grid_norm[2, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "", label = "", widen = false,
    clim = (0, 1), alpha = 0.75, legend = false,
    xlims = extrema(x), ylims = extrema(y), colorbar = :left, colorbar_title = "||μ₂||"
)
scatter!(twinx(), eachcol(locs)..., marker_z = Y[:, 2], xmirror = true, widen = false,
    clim = (0, 1), color = :viridis, msw = 0.25, legend = false, colorbar = :right,
    xlims = extrema(locs[:, 1]), ylims = extrema(locs[:, 2]), colorbar_title = "Y₂"
)
nmd3 = heatmap(x, y, reshape(mu_grid_norm[3, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "", label = "", widen = false,
    clim = (0, 1), alpha = 0.75, legend = false,
    xlims = extrema(x), ylims = extrema(y), colorbar = :left, colorbar_title = "||μ₃||"
)
scatter!(twinx(), eachcol(locs)..., marker_z = Y[:, 3], xmirror = true, widen = false,
    clim = (0, 1), color = :viridis, msw = 0.25, legend = false, colorbar = :right,
    xlims = extrema(locs[:, 1]), ylims = extrema(locs[:, 2]), colorbar_title = "Y₃"
)

nmd_obs = plot(nmd1, nmd2, nmd3, layout = (3, 1), size = (600, 1000))
savefig(nmd_obs, "nmd_obs.pdf")

###
### Side-by-side comparison of true vs predicted mean directions
###

mu_grid = [link.(B*X_grid[i, :] + etaGrid_mat[:, i]) for i in 1:n_grid]
mu_grid_norm = zeros(d, n_grid)
for i in 1:n_grid
    mu_grid_norm[:, i] = normalize(mu_grid[i])
end
mu_grid_mat = copy(mu_grid_norm)

B_post = mean(out["B"], dims = 3)[:, :, 1]
mu_pred = [link.(B_post*X_grid[i, :] + etaPred_mat[:, i]) for i in 1:n_grid]
mu_pred_norm = zeros(d, n_grid)
for i in 1:n_grid
    mu_pred_norm[:, i] = normalize(mu_pred[i])
end
mu_pred_mat = copy(mu_pred_norm)

## spatial field #1
truth1 = heatmap(x, y, reshape(mu_grid_mat[1, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "True NMD", titlefontsize = 25, label = "", widen = false,
    clim = extrema(mu_grid_mat[1, :]), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), colorbar_title = "μ₁", colorbar_titlefontsize = 20
)
pred1 = heatmap(x, y, reshape(mu_pred_mat[1, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "Estimated NMD (Non-Spatial)", titlefontsize = 25, label = "", widen = false,
    clim = extrema(mu_pred_mat[1, :]), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), colorbar_title = "μ₁", colorbar_titlefontsize = 20
)

## spatial field #2
truth2 = heatmap(x, y, reshape(mu_grid_mat[2, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "", label = "", widen = false,
    clim = extrema(mu_grid_mat[2, :]), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), colorbar_title = "μ₂", colorbar_titlefontsize = 20
)
pred2 = heatmap(x, y, reshape(mu_pred_mat[2, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "", label = "", widen = false,
    clim = extrema(mu_pred_mat[2, :]), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), colorbar_title = "μ₂", colorbar_titlefontsize = 20
)

## spatial field #3
truth3 = heatmap(x, y, reshape(mu_grid_mat[3, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "", label = "", widen = false,
    clim = extrema(mu_grid_mat[3, :]), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), colorbar_title = "μ₃", colorbar_titlefontsize = 20
)
pred3 = heatmap(x, y, reshape(mu_pred_mat[3, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "", label = "", widen = false,
    clim = extrema(mu_pred_mat[3, :]), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), colorbar_title = "μ₃", colorbar_titlefontsize = 20
)

mean_dir = plot(truth1, pred1, truth2, pred2, truth3, pred3, layout = (3, 2), size = (1100, 1000), left_margin = 25mm)
savefig(mean_dir, "tog_norm_mean_dir.pdf")

###
### Logarithmic Score (ESAG)
###

## posterior mus
B_post = mean(out["B"], dims = 3)[:, :, 1]
alpha_post = mean(out["alpha"], dims = 2)[:, 1]
A_post = Matrix(kronecker(ones(d), alpha_post'))
mu_ps = [link.(B_post*X[i, :] + etaPred_mat[:, i] + A_post*z[i, :]) for i in 1:n]

## posterior gammas
gamma_post = [[0.0, 0.0]]
iterCounter = 1
for j in 1:(d-2)
    for i in 1:(j+1)
        for k in 1:(nMCMC-nBurn+1)
            gamma_post[j][i] += out["gamma"][k][j][i]
        end
        gamma_post[j][i] = gamma_post[j][i]/(nMCMC-nBurn+1)
    end
end

## compute V
V_ps = zeros(d, d, n)
Vinv_ps = zeros(d, d, n)
xi_ps = zeros(d, d, n)
[(V_ps[:, :, i], xi_ps[:, :, i]) = get_V(mu_ps[i], gamma_post, d, get_xi = true) for i in 1:n]
[Vinv_ps[:, :, i] = inv(V_ps[:, :, i]) for i in 1:n]

## compute logS
logS = [-dESAG(Y[i, :], mu_ps[i], Vinv_ps[:, :, i], logm = true, trunc = true) for i in 1:n]
logS_mean = mean(logS)

###
### Compute predicted normalized mean directions
###

## posterior mu
mu_list = zeros(d, n_grid)
[mu_list[:, i] = link.(B_post*XG[i, :] + etaPred_mat[:, i]) for i in 1:n_grid]

## posterior V, V_inv, and nc
V_list = zeros(d, d, n_grid)
Vinv_list = zeros(d, d, n_grid)
[V_list[:, :, i] = get_V(mu_list[:, i], gamma_post, d, get_xi = false) for i in 1:n_grid]
[Vinv_list[:, :, i] = inv(V_list[:, :, i]) for i in 1:n_grid]

## posterior NMD
nmd_pred = zeros(d, n_grid)
println("Predicting NMD (Non-Spatial):")
[nmd_pred[:, i] = MCmean(mu_list[:, i], V_list[:, :, i], 50000, trunc = true) for i in ProgressBar(1:n_grid)]
[nmd_pred[:, i] = nmd_pred[:, i] ./ norm(nmd_pred[:, i]) for i in 1:n_grid]

###
### Save MCMC output and eta predictions
###

@save "MCMCoutput.jld2" out etaPred_mat

####
#### Run MCMC with ESAG+
####

println("Spatial ESAG MCMC:")
out_trunc = mcmc_base(Y, X, z, spatial_idx, nMCMC, nBurn, nSamp, 1, 5, true, R_list, phiIdx_vec)

###
### ESAG⁺ Traces
###

## β trace plots
plots = []
for i in 1:d
    for j in 1:p
        tmpPlot = plot(out_trunc["B"][i, j, :], title = "β[$i, $j]")
        hline!(tmpPlot, [B[i, j]], color = :red)
        push!(plots, tmpPlot)
    end
end
plot(plots..., layout = (d, p), size = (900, 600), left_margin = 10mm, legend = false)
savefig("betaTraces_ESAG_trunc.pdf")

## α trace plots
plots = []
for i in 1:q
    tmpPlot = plot(out_trunc["alpha"][i, :], title = "α[$i]")
    hline!(tmpPlot, [alphas[i]], color = :red)
    push!(plots, tmpPlot)
end
plot(plots..., layout = (q, 1), size = (900, 600), left_margin = 10mm, legend = false)
savefig("alphaTraces_ESAG_trunc.pdf")

## number of ESS iterations per MCMC iteration
plot(out_trunc["ESS_counts"], title = "ESS Counts (ESAG)", left_margin = 10mm, legend = false)
savefig("ESS_ESAG_trunc.pdf")

## eta traces 
plots = []
for i in 1:d
    for j in 1:5
        tmpPlot = plot(out_trunc["eta"][i, j, :], title = "eta[$i, $j]")
        hline!(tmpPlot, [eta_mat[i, j]], color = :red)
        push!(plots, tmpPlot)
    end
end
plot(plots..., layout = (d, 5), size = (900, 600), left_margin = 10mm, legend = false)
savefig("etaTraces_ESAG_trunc.pdf")

## C (coded as Am) traces 
plots = []
for i in 1:d
    for j in 1:d
        tmpPlot = plot(out_trunc["Am"][i, j, :], title = "Am[$i, $j]")
        hline!(tmpPlot, [Am[i, j]], color = :red)
        push!(plots, tmpPlot)
    end
end
plot(plots..., layout = (d, d), size = (900, 600), left_margin = 10mm, legend = false)
savefig("AmTraces_ESAG_trunc.pdf")

## phi indices
plots = []
for i in 1:d
    tmpPlot = plot(out_trunc["phiIdx"][i, :], title = "phi_index[$i]")
    hline!(tmpPlot, [phiIdx_vec[i]], color = :red)
    push!(plots, tmpPlot)
end
plot(plots..., layout = (d, 1), size = (900, 600), left_margin = 10mm, legend = false)
savefig("phiTraces_ESAG_trunc.pdf")

###
### Predicted η Surface
###

## computes the appropriate exponential decay covariance matrix across given locations
function sigw(distMat, phi)
    sigw = zeros(size(distMat))
    for i in 1:size(distMat, 1)
        for j in 1:size(distMat, 2)
            sigw[i, j] = exp(-distMat[i, j]/phi)
        end
    end
    return sigw
end

## Kriging
etaPred = zeros(n_grid*d, nKeep)
println("Kriging: ")
for k in ProgressBar((nMCMC-nBurn+1-nKeep+1):(nMCMC-nBurn+1)) # for each iteration
    Sig_eta_oo = zeros(n*d, n*d)
    Sig_eta_no = zeros((n_aug-n)*d, n*d)
    Sig_eta_nn = zeros((n_aug-n)*d, (n_aug-n)*d)
    for j in 1:d 
        AAt = out_trunc["Am"][:, j, k]*out_trunc["Am"][:, j, k]'
        R_oo = sigw(distMat[1:n, 1:n], phi_list[Int(out_trunc["phiIdx"][j, k])])
        R_no = sigw(distMat[(n+1):n_aug, 1:n], phi_list[Int(out_trunc["phiIdx"][j, k])])
        R_nn = sigw(distMat[(n+1):n_aug, (n+1):n_aug], phi_list[Int(out_trunc["phiIdx"][j, k])])
        Sig_eta_oo += Matrix(kronecker(R_oo, AAt))
        Sig_eta_no += Matrix(kronecker(R_no, AAt))
        Sig_eta_nn += Matrix(kronecker(R_nn, AAt))
    end
    Sig_eta_oo_i = inv(Sig_eta_oo)
    muStar = Sig_eta_no*Sig_eta_oo_i*vec(out_trunc["eta"][:, :, k])
    SigStar = Sig_eta_nn - Sig_eta_no*Sig_eta_oo_i*Sig_eta_no'
    if !isposdef(SigStar)
        SigStar = (SigStar + SigStar') / 2
    end
    etaPred[:, k-((nMCMC-nBurn+1-nKeep+1))+1] = rand(MvNormal(muStar, Hermitian(SigStar)), 1)
end
etaPred = mean(etaPred, dims = 2)  
etaPred_mat = reshape(etaPred, d, n_grid)

###
### Predicted w/ Observations Overlaid
###

## spatial field #1
pred1_trunc = heatmap(x, y, reshape(etaPred_mat[1, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "Predicted η Grid w/ Observations", label = "", widen = false,
    clim = extrema(etaPred_mat[1, :]), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), colorbar_title = "η₁"
)
scatter!(twinx(), eachcol(locs)..., marker_z = Y[:, 1], xmirror = true, widen = false,
    clim = extrema(Y[:, 1]), color = :viridis, msw = 0.25, legend = false, colorbar = :right,
    xlims = extrema(locs[:, 1]), ylims = extrema(locs[:, 2]), colorbar_title = "Y₁"
)

## spatial field #2
pred2_trunc = heatmap(x, y, reshape(etaPred_mat[2, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "", label = "", widen = false,
    clim = extrema(etaPred_mat[2, :]), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), colorbar_title = "η₂"
)
scatter!(twinx(), eachcol(locs)..., marker_z = Y[:, 2], xmirror = true, widen = false,
    clim = extrema(Y[:, 2]), color = :viridis, msw = 0.25, legend = false, colorbar = :right,
    xlims = extrema(locs[:, 1]), ylims = extrema(locs[:, 2]), colorbar_title = "Y₂"
)

## spatial field #3
pred3_trunc = heatmap(x, y, reshape(etaPred_mat[3, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "", label = "", widen = false,
    clim = extrema(etaPred_mat[3, :]), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), colorbar_title = "η₃"
)
scatter!(twinx(), eachcol(locs)..., marker_z = Y[:, 3], xmirror = true, widen = false,
    clim = extrema(Y[:, 3]), color = :viridis, msw = 0.25, legend = false, colorbar = :right,
    xlims = extrema(locs[:, 1]), ylims = extrema(locs[:, 2]), colorbar_title = "Y₃"
)

pred_obs = plot(pred1_trunc, pred2_trunc, pred3_trunc, layout = (3, 1), size = (600, 1000))
savefig(pred_obs, "pred_obs_trunc.pdf")

###
### Mean Direction Maps
###

## compute normalized mean directions
mu_grid = [link.(B*X_grid[i, :] + etaGrid_mat[:, i]) for i in 1:n_grid]
mu_grid_norm = zeros(d, n_grid)
for i in 1:n_grid
    mu_grid_norm[:, i] = normalize(mu_grid[i])
end

## plot heatmaps of normalized mean directions with observed spherical coordinates
nmd1 = heatmap(x, y, reshape(mu_grid_norm[1, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "Normalized Mean Directions w/ Observations", label = "", widen = false,
    clim = (0, 1), alpha = 0.75, legend = false,
    xlims = extrema(x), ylims = extrema(y), colorbar = :left, colorbar_title = "||μ₁||"
)
scatter!(twinx(), eachcol(locs)..., marker_z = Y[:, 1], xmirror = true, widen = false,
    clim = (0, 1), color = :viridis, msw = 0.25, legend = false, colorbar = :right,
    xlims = extrema(locs[:, 1]), ylims = extrema(locs[:, 2]), colorbar_title = "Y₁"
)
nmd2 = heatmap(x, y, reshape(mu_grid_norm[2, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "", label = "", widen = false,
    clim = (0, 1), alpha = 0.75, legend = false,
    xlims = extrema(x), ylims = extrema(y), colorbar = :left, colorbar_title = "||μ₂||"
)
scatter!(twinx(), eachcol(locs)..., marker_z = Y[:, 2], xmirror = true, widen = false,
    clim = (0, 1), color = :viridis, msw = 0.25, legend = false, colorbar = :right,
    xlims = extrema(locs[:, 1]), ylims = extrema(locs[:, 2]), colorbar_title = "Y₂"
)
nmd3 = heatmap(x, y, reshape(mu_grid_norm[3, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "", label = "", widen = false,
    clim = (0, 1), alpha = 0.75, legend = false,
    xlims = extrema(x), ylims = extrema(y), colorbar = :left, colorbar_title = "||μ₃||"
)
scatter!(twinx(), eachcol(locs)..., marker_z = Y[:, 3], xmirror = true, widen = false,
    clim = (0, 1), color = :viridis, msw = 0.25, legend = false, colorbar = :right,
    xlims = extrema(locs[:, 1]), ylims = extrema(locs[:, 2]), colorbar_title = "Y₃"
)

nmd_obs = plot(nmd1, nmd2, nmd3, layout = (3, 1), size = (600, 1000))
savefig(nmd_obs, "nmd_obs_trunc.pdf")

###
### Side-by-side comparison of true vs predicted mean directions
###

B_post = mean(out_trunc["B"], dims = 3)[:, :, 1]
mu_pred = [link.(B_post*X_grid[i, :] + etaPred_mat[:, i]) for i in 1:n_grid]
mu_pred_norm = zeros(d, n_grid)
for i in 1:n_grid
    mu_pred_norm[:, i] = normalize(mu_pred[i])
end
mu_pred_mat = copy(mu_pred_norm)

## spatial field #1
pred1_trunc = heatmap(x, y, reshape(mu_pred_mat[1, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "Estimated NMD (Spatial)", titlefontsize = 25, label = "", widen = false,
    clim = extrema(mu_pred_mat[1, :]), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), colorbar_title = "μ₁", colorbar_titlefontsize = 20
)

## spatial field #2
pred2_trunc = heatmap(x, y, reshape(mu_pred_mat[2, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "", label = "", widen = false,
    clim = extrema(mu_pred_mat[2, :]), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), colorbar_title = "μ₂", colorbar_titlefontsize = 20
)

## spatial field #3
pred3_trunc = heatmap(x, y, reshape(mu_pred_mat[3, :], ngridpoints, ngridpoints)',
    color = :viridis, title = "", label = "", widen = false,
    clim = extrema(mu_pred_mat[3, :]), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), colorbar_title = "μ₃", colorbar_titlefontsize = 20
)

mean_dir = plot(truth1, pred1_trunc, truth2, pred2_trunc, truth3, pred3_trunc, layout = (3, 2), size = (1100, 1000), left_margin = 25mm)
savefig(mean_dir, "tog_norm_mean_dir_trunc.pdf")

mean_dir = plot(truth1, pred1, pred1_trunc, truth2, pred2, pred2_trunc, truth3, pred3, pred3_trunc, layout = (3, 3), size = (2000, 1300), left_margin = 25mm)
savefig(mean_dir, "tog_norm_mean_dir_all.pdf")

###
### Compute normalized mean directions (truncated)
###

## poterior betas 
B_post = mean(out_trunc["B"], dims = 3)[:, :, 1]

## posterior gammas
gamma_post = [[0.0, 0.0]]
iterCounter = 1
for j in 1:(d-2)
    for i in 1:(j+1)
        for k in 1:(nMCMC-nBurn+1)
            gamma_post[j][i] += out_trunc["gamma"][k][j][i]
        end
        gamma_post[j][i] = gamma_post[j][i]/(nMCMC-nBurn+1)
    end
end

## posterior mu
mu_list = zeros(d, n_grid)
[mu_list[:, i] = link.(B_post*XG[i, :] + etaPred_mat[:, i]) for i in 1:n_grid]

## posterior V, V_inv, and nc
V_list = zeros(d, d, n_grid)
Vinv_list = zeros(d, d, n_grid)
[V_list[:, :, i] = get_V(mu_list[:, i], gamma_post, d, get_xi = false) for i in 1:n_grid]
[Vinv_list[:, :, i] = inv(V_list[:, :, i]) for i in 1:n_grid]

## posterior NMD
nmd_pred_trunc = zeros(d, n_grid)
println("Predicting NMD (Spatial):")
[nmd_pred_trunc[:, i] = MCmean(mu_list[:, i], V_list[:, :, i], 50000, trunc = true) for i in ProgressBar(1:n_grid)]
[nmd_pred_trunc[:, i] = nmd_pred_trunc[:, i] ./ norm(nmd_pred_trunc[:, i]) for i in 1:n_grid]

###
### Save MCMC output and eta predictions
###

@save "MCMCoutput_trunc.jld2" out_trunc etaPred_mat

###
### Logarithmic Score (ESAG+)
###

## posterior mus
B_post = mean(out_trunc["B"], dims = 3)[:, :, 1]
alpha_post = mean(out_trunc["alpha"], dims = 2)[:, 1]
A_post = Matrix(kronecker(ones(d), alpha_post'))
mu_ps = [link.(B_post*X[i, :] + etaPred_mat[:, i] + A_post*z[i, :]) for i in 1:n]

## posterior gammas
gamma_post = [[0.0, 0.0]]
iterCounter = 1
for j in 1:(d-2)
    for i in 1:(j+1)
        for k in 1:(nMCMC-nBurn+1)
            gamma_post[j][i] += out_trunc["gamma"][k][j][i]
        end
        gamma_post[j][i] = gamma_post[j][i]/(nMCMC-nBurn+1)
    end
end

## compute V
V_ps = zeros(d, d, n)
Vinv_ps = zeros(d, d, n)
xi_ps = zeros(d, d, n)
[(V_ps[:, :, i], xi_ps[:, :, i]) = get_V(mu_ps[i], gamma_post, d, get_xi = true) for i in 1:n]
[Vinv_ps[:, :, i] = inv(V_ps[:, :, i]) for i in 1:n]

## compute logS
logS_trunc = [-dESAG(Y[i, :], mu_ps[i], Vinv_ps[:, :, i], logm = true, trunc = true) for i in 1:n]
logS_mean_trunc = mean(logS_trunc)

###
### Plot Predicted NMD Together
###

## GR-backend
gr() 
ms = 4

## first component
nmd11 = heatmap(x, y, reshape(nmd_true[1, :], ngridpoints, ngridpoints),
color = :viridis, title = "Truth", titlefontsize = 15, label = "", widen = false,
clim = extrema([nmd_true[1, :] nmd_pred[1, :] nmd_pred_trunc[1, :]]), alpha = 0.75, legend = false, 
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
scatter!(locs[:, 1], locs[:, 2], color = :black, markersize = ms, label = "") 

nmd12 = heatmap(x, y, reshape(nmd_pred_trunc[1, :], ngridpoints, ngridpoints),
color = :viridis, title = "Prediction (Spatial)", titlefontsize = 15, label = "", widen = false,
clim = extrema([nmd_true[1, :] nmd_pred[1, :] nmd_pred_trunc[1, :]]), alpha = 0.75, legend = false, 
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false
)
scatter!(locs[:, 1], locs[:, 2], color = :black, markersize = ms, label = "") 

nmd13 = heatmap(x, y, reshape(nmd_pred[1, :], ngridpoints, ngridpoints),
color = :viridis, title = "Prediction (Non-Spatial)", titlefontsize = 15, label = "", widen = false,
clim = extrema([nmd_true[1, :] nmd_pred[1, :] nmd_pred_trunc[1, :]]), alpha = 0.75, legend = false, colorbar = :right, 
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false, colorbar_title = "Component 1", colorbar_titlefontsize = 15
)
scatter!(locs[:, 1], locs[:, 2], color = :black, markersize = ms, label = "") 

## second component
nmd21 = heatmap(x, y, reshape(nmd_true[2, :], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 30, label = "", widen = false,
clim = extrema([nmd_true[2, :] nmd_pred[2, :] nmd_pred_trunc[2, :]]), alpha = 0.75, legend = false, 
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
scatter!(locs[:, 1], locs[:, 2], color = :black, markersize = ms, label = "") 

nmd22 = heatmap(x, y, reshape(nmd_pred_trunc[2, :], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 30, label = "", widen = false,
clim = extrema([nmd_true[2, :] nmd_pred[2, :] nmd_pred_trunc[2, :]]), alpha = 0.75, legend = false,  
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false
)
scatter!(locs[:, 1], locs[:, 2], color = :black, markersize = ms, label = "") 

nmd23 = heatmap(x, y, reshape(nmd_pred[2, :], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 30, label = "", widen = false,
clim = extrema([nmd_true[2, :] nmd_pred[2, :] nmd_pred_trunc[2, :]]), alpha = 0.75, legend = false, colorbar = :right,
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false, colorbar_title = "Component 2", colorbar_titlefontsize = 15 
)
scatter!(locs[:, 1], locs[:, 2], color = :black, markersize = ms, label = "") 

## third component
nmd31 = heatmap(x, y, reshape(nmd_true[3, :], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 30, label = "", widen = false,
clim = extrema([nmd_true[3, :] nmd_pred[3, :] nmd_pred_trunc[3, :]]), alpha = 0.75, legend = false, 
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
scatter!(locs[:, 1], locs[:, 2], color = :black, markersize = ms, label = "") 

nmd32 = heatmap(x, y, reshape(nmd_pred_trunc[3, :], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 30, label = "", widen = false,
clim = extrema([nmd_true[3, :] nmd_pred[3, :] nmd_pred_trunc[3, :]]), alpha = 0.75, legend = false, 
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false
)
scatter!(locs[:, 1], locs[:, 2], color = :black, markersize = ms, label = "") 

nmd33 = heatmap(x, y, reshape(nmd_pred[3, :], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 30, label = "", widen = false,
clim = extrema([nmd_true[3, :] nmd_pred[3, :] nmd_pred_trunc[3, :]]), alpha = 0.75, legend = false, colorbar = :right, 
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false, colorbar_title = "Component 3", colorbar_titlefontsize = 15 
)
scatter!(locs[:, 1], locs[:, 2], color = :black, markersize = ms, label = "") 

## combine and save plot
r1_nmd = plot(nmd11, nmd12, nmd13, layout = Plots.grid(1, 3, widths = [0.30, 0.30, 0.40]), size = (2000, 500))
r2_nmd = plot(nmd21, nmd22, nmd23, layout = Plots.grid(1, 3, widths = [0.30, 0.30, 0.40]), size = (2000, 500))
r3_nmd = plot(nmd31, nmd32, nmd33, layout = Plots.grid(1, 3, widths = [0.30, 0.30, 0.40]), size = (2000, 500))
all_nmd = plot(r1_nmd, r2_nmd, r3_nmd, layout = (3, 1), size = (2000, 1500), dpi = 600)
savefig(all_nmd, "nmd_compare.png")

###
### Print LogS scores 
###

println("LogS (non-Spatial): ", logS_mean)
println("LogS (Spatial): ", logS_mean_trunc)

###
### Save Important Information
###

@save "info.jld2" logS_mean logS_mean_trunc out out_trunc etaPred etaPred_mat