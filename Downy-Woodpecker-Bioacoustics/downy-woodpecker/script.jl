############################################################
####
#### Downy Woodpecker Study to Compare ESAG and ESAG+ Models
####
############################################################

###
### Set-up
###

using Random, Distributions, Plots, DataFrames, StatsBase
using Distances, Kronecker
using ProgressBars, JLD2
using Plots.PlotMeasures
using IterTools
using Missings
using Base.Threads
using RCall
using Roots
using MCMCChains

Random.seed!(512)
include("../../esag_functions.jl")
include("mcmc.jl")
include("mcmc_base.jl")
nMCMC = 400000
nBurn = Int(0.4*nMCMC)
nKeep = 1000
nSamp = 10000
trunc_flag = true # whether we fit the ESAG+ model

## link function for mean direction μ
function link(x)
    return log(1 + exp(x))
end

## transform to range function
function ttr(vec::Vector{Float64}, a::Float64, b::Float64)
    min_val = minimum(vec)
    max_val = maximum(vec)
    
    normalized_vec = (vec .- min_val) ./ (max_val - min_val) # normalize to [0, 1]
    scaled_vec = a .+ (b - a) .* normalized_vec # scale to [a, b]
    
    return scaled_vec
end

## directional standardization function ; Scealy & Wood (2019), Yu & Huang (2024+)
function dirStandard(vec::Vector{Float64})
    min_val = minimum(vec)
    max_val = maximum(vec)
    
    normalized_vec = (vec .- min_val) ./ (max_val - min_val) .+ 1
    
    return normalized_vec
end

## standardize grid covariates using scaling of original covariates
function dirStandard_grid(X_grid_vec, vec::Vector{Float64})
    min_val = minimum(vec)
    max_val = maximum(vec)
    
    normalized_X_grid_vec = (X_grid_vec .- min_val) ./ (max_val - min_val) .+ 1
    
    return normalized_X_grid_vec
end

## element-wise square-root transformation
function c2d(u::AbstractMatrix)
    return sqrt.(u)
end

## Matérn 3/2 function
function m32(d, phi)
    rho = (1 + sqrt(3)*d/phi)*exp(-(sqrt(3)*d)/phi) 
    return rho
end

## function to find appropriate upper-bound for m32 given data
function ubPhi(d)
    phi_list = 0.01:0.01:100
    answers = m32.(d, phi_list)
    idx = argmin(abs.(answers .- 0.05))
    return phi_list[idx]
end

## function to solve for gamma parameters given variance v and upper-bound u
function solve_gamma(v, u)
    ## define the function whose root we wish to find
    f(k) = quantile(Gamma(k, sqrt(v/k)), 0.95) - u

    ## find a root for k ; choose an interval that we expect contains the solution and adjust the interval as necessary
    k_est = find_zero(f, (1e-3, 1e3))
    
    ## once k is known, compute θ from the variance equation
    θ_est = sqrt(v / k_est)
    
    return k_est, θ_est
end

###
### Load Data via R
###

R"""
library(tidyverse)

## load and format observed data
load("NE_3_5.RData")
head(ne_data_3_5_stars)

data = ne_data_3_5_stars %>% filter(season %in% c("spring"))
Y = data %>% select(laugh, drum, pik)
X = data %>% select(avg_temp_deg_c, avg_precip_mm)
X = cbind(rep(1, dim(X)[1]), X)
z1 = data %>% mutate(z1 = ifelse(avg_rating == 4, 1, 0)) %>% select(z1)
z2 = data %>% mutate(z2 = ifelse(avg_rating == 5, 1, 0)) %>% select(z2)
Z = cbind(z1, z2)
obs_locs = data %>% select(Latitude, Longitude)
unique_locs = unique(obs_locs)

## load gridded data
load("grid_data/grid_spring_60_2020.RData")
load("grid_data/grid_spring_60_2021.RData")
load("grid_data/grid_spring_60_2022.RData")
load("grid_data/grid_spring_60_2023.RData")
grid2020 = grid_spring_60_2020
grid2021 = grid_spring_60_2021
grid2022 = grid_spring_60_2022
grid2023 = grid_spring_60_2023
grid_coords = grid2020[, 1:2]
n_grid = dim(grid_coords)[1]
"""
@rget Y X Z obs_locs unique_locs grid2020 grid2021 grid2022 grid2023 grid_coords n_grid

## format data for Julia
Y = Matrix(Y)
Y = c2d(Y)
X = Matrix(X)
z = Matrix(Z)
grid_coords = Matrix(grid_coords)
obs_locs = Matrix(obs_locs)
unique_locs = Matrix(unique_locs)

###
### Prepare data
###

## sizes and dimensions
n, d = size(Y)
p = size(X, 2)
q = size(z, 2)
m = size(unique_locs, 1)

## find unique spatial location indices
row_to_index = Dict{Vector{eltype(obs_locs)}, Int}()
for (i, row) in enumerate(eachrow(unique_locs))
    row_to_index[row] = i
end
spatial_idx = Vector{Int}(undef, size(obs_locs, 1))
for (i, row) in enumerate(eachrow(obs_locs))
    spatial_idx[i] = row_to_index[row]
end

## standardize X
X_orig = copy(X)
[X[:, i] = dirStandard(X[:, i]) for i in 2:p]

## create grid across spatial domain
locs_aug = [unique_locs ; grid_coords]
n_aug = size(locs_aug, 1)

## pre-compute phi and Sig_w lists for MCMC
distMat = pairwise(Euclidean(), locs_aug')

## find appropriate phi hyperparameters for gamma distribution 
maxdist = maximum(distMat)
ub = ubPhi(0.25*maxdist) # get the upper-bound for 95% support
v = 0.003
phi_alpha, phi_theta = solve_gamma(v, ub)
cdf(Gamma(phi_alpha, phi_theta), ub) # should be approximately 0.95
var(Gamma(phi_alpha,phi_theta)) # should equal v
m32(0.25*maxdist, ub) # should be approximately 0.05
pdf(Gamma(phi_alpha,phi_theta), ub)

###
### Fit ESAG Model via MCMC
###

println("ESAG MCMC:")
out = mcmc(Y, X, z, spatial_idx, nMCMC, nBurn, nSamp, 1, 5, false, distMat[1:m, 1:m], phi_alpha, phi_theta)

## solve for thinning factor
max_lag = Int(0.5*nMCMC) 
ac = StatsBase.autocor(out["phi"][1, :], 0:max_lag)
lags = 0:length(ac)-1 # lags for the full autocorrelation
lags_limited = 0:max_lag
thin = findfirst(x -> x <= 0.1, ac)
println("Suggested thin: ", thin)
phi_chains = Chains(out["phi"]')
println("Phi diagnostics: ")
display(ess_rhat(phi_chains))
eta_chains_1 = [Chains(out["eta"][1, i, :]) for i in 1:5]
eta_chains_2 = [Chains(out["eta"][2, i, :]) for i in 1:5]
eta_chains_3 = [Chains(out["eta"][3, i, :]) for i in 1:5]
println("Eta diagnostics: ")
[display(ess_rhat(eta_chains_1[i])) for i in 1:5]
[display(ess_rhat(eta_chains_2[i])) for i in 1:5]
[display(ess_rhat(eta_chains_3[i])) for i in 1:5]

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
    tmpPlot = plot(out["phi"][i, :], title = "phi[$i]")
    push!(plots, tmpPlot)
end
plot(plots..., layout = (d, 1), size = (900, 600), left_margin = 10mm, legend = false)
savefig("phiTraces_ESAG.pdf")

###
### Predicted η Surface
###

## Kriging
etaPred = zeros(n_grid*d, nKeep)
let
    Sig_eta_oo = zeros(m*d, m*d)
    Sig_eta_no = zeros((n_aug-m)*d, m*d)
    Sig_eta_nn = zeros((n_aug-m)*d, (n_aug-m)*d)
    println("Kriging: ")
    for k in ProgressBar((nMCMC-nBurn+1-nKeep+1):(nMCMC-nBurn+1))
        if k % 10 == 0
            GC.gc()
        end
        fill!(Sig_eta_oo, 0.0)
        fill!(Sig_eta_no, 0.0)
        fill!(Sig_eta_nn, 0.0)
        for j in 1:d 
            AAt = out["Am"][:, j, k]*out["Am"][:, j, k]'
            R_oo = m32.(distMat[1:m, 1:m], out["phi"][j, k])
            R_no = m32.(distMat[(m+1):n_aug, 1:m], out["phi"][j, k])
            R_nn = m32.(distMat[(m+1):n_aug, (m+1):n_aug], out["phi"][j, k])
            Sig_eta_oo += kronecker(R_oo, AAt)
            Sig_eta_no += kronecker(R_no, AAt)
            Sig_eta_nn += kronecker(R_nn, AAt)
        end
        Sig_eta_oo_i = inv(Sig_eta_oo)
        muStar = Sig_eta_no*Sig_eta_oo_i*vec(out["eta"][:, :, k])
        SigStar = Sig_eta_nn - Sig_eta_no*Sig_eta_oo_i*Sig_eta_no'
        if !isposdef(SigStar)
            SigStar = (SigStar + SigStar') / 2
        end
        etaPred[:, k-((nMCMC-nBurn+1-nKeep+1))+1] = rand(MvNormal(muStar, Hermitian(SigStar)), 1)
    end
end
etaPred = mean(etaPred, dims = 2)  
etaPred_mat = reshape(etaPred, d, n_grid)

###
### Predicted w/ Observations Overlaid
###

ngridpoints = Int(sqrt(n_grid))
x = unique(grid_coords[:, 2])
y = unique(grid_coords[:, 1])

## spatial field #1
pred1 = heatmap(x, y, reshape(etaPred_mat[1, :], ngridpoints, ngridpoints),
    color = :viridis, title = "Predicted η Grid w/ Observations", label = "", widen = false,
    clim = extrema(etaPred_mat[1, :]), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), colorbar_title = "η₁"
)
scatter!(twinx(), eachcol(obs_locs)..., marker_z = Y[:, 1], xmirror = true, widen = false,
    clim = extrema(Y[:, 1]), color = :viridis, msw = 0.25, legend = false, colorbar = :right,
    xlims = extrema(obs_locs[:, 1]), ylims = extrema(obs_locs[:, 2]), colorbar_title = "Y₁"
)

## spatial field #2
pred2 = heatmap(x, y, reshape(etaPred_mat[2, :], ngridpoints, ngridpoints),
    color = :viridis, title = "", label = "", widen = false,
    clim = extrema(etaPred_mat[2, :]), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), colorbar_title = "η₂"
)
scatter!(twinx(), eachcol(obs_locs)..., marker_z = Y[:, 2], xmirror = true, widen = false,
    clim = extrema(Y[:, 2]), color = :viridis, msw = 0.25, legend = false, colorbar = :right,
    xlims = extrema(obs_locs[:, 1]), ylims = extrema(obs_locs[:, 2]), colorbar_title = "Y₂"
)

## spatial field #3
pred3 = heatmap(x, y, reshape(etaPred_mat[3, :], ngridpoints, ngridpoints),
    color = :viridis, title = "", label = "", widen = false,
    clim = extrema(etaPred_mat[3, :]), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), colorbar_title = "η₃"
)
scatter!(twinx(), eachcol(obs_locs)..., marker_z = Y[:, 3], xmirror = true, widen = false,
    clim = extrema(Y[:, 3]), color = :viridis, msw = 0.25, legend = false, colorbar = :right,
    xlims = extrema(obs_locs[:, 1]), ylims = extrema(obs_locs[:, 2]), colorbar_title = "Y₃"
)

pred_obs = plot(pred1, pred2, pred3, layout = (3, 1), size = (600, 1000))
savefig(pred_obs, "pred_obs.pdf")

###
### Predicted compositions
###

## posterior betas
B_post = mean(out["B"], dims = 3)[:, :, 1]

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

## predict compositions
year_vec = 2020:2023
comp_pred = missings(Float64, d, n_grid, 4)
for j in 1:4

    ## preparing grid
    if j == 1
        XG = [ones(n_grid) Matrix(grid2020)[:, 4:5]]
    elseif j == 2
        XG = [ones(n_grid) Matrix(grid2021)[:, 4:5]]
    elseif j == 3
        XG = [ones(n_grid) Matrix(grid2022)[:, 4:5]]
    else
        XG = [ones(n_grid) Matrix(grid2023)[:, 4:5]]
    end
    [XG[:, i] = dirStandard_grid(XG[:, i], X_orig[:, i]) for i in 2:p]
    validIdx = findall(row -> !any(ismissing, row), eachrow(XG))
    validNo = length(validIdx)

    ## posterior mu (grid)
    mu_list = zeros(d, n_grid)
    [mu_list[:, validIdx[i]] = link.(B_post*XG[validIdx[i], :] + etaPred_mat[:, validIdx[i]]) for i in 1:validNo]

    ## posterior V, V_inv, and nc
    V_list = zeros(d, d, n_grid)
    Vinv_list = zeros(d, d, n_grid)
    [V_list[:, :, validIdx[i]] = get_V(mu_list[:, validIdx[i]], gamma_post, d, get_xi = false) for i in 1:validNo]
    [Vinv_list[:, :, validIdx[i]] = inv(V_list[:, :, validIdx[i]]) for i in 1:validNo]
    nc = zeros(n_grid)
    
    ## posterior compositional predictions
    println("Predicting Compositions for $(year_vec[j]):")
    [comp_pred[:, validIdx[i], j] = estComp(mu_list[:, validIdx[i]], V_list[:, :, validIdx[i]], 50000, trunc = false) for i in ProgressBar(1:validNo)]
end

###
### Predicted Compositions (d colorbars)
###

## set-up
ngridpoints = Int(sqrt(n_grid))
x = unique(grid_coords[:, 2])
y = unique(grid_coords[:, 1])

## first component
comp11 = heatmap(x, y, reshape(comp_pred[1, :, 1], ngridpoints, ngridpoints),
color = :viridis, title = "$(year_vec[1])", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[1, :, :])), alpha = 0.75, legend = false,
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false
)
comp12 = heatmap(x, y, reshape(comp_pred[1, :, 2], ngridpoints, ngridpoints),
color = :viridis, title = "$(year_vec[2])", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[1, :, :])), alpha = 0.75, legend = false, 
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
comp13 = heatmap(x, y, reshape(comp_pred[1, :, 3], ngridpoints, ngridpoints),
color = :viridis, title = "$(year_vec[3])", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[1, :, :])), alpha = 0.75, legend = false,
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
comp14 = heatmap(x, y, reshape(comp_pred[1, :, 4], ngridpoints, ngridpoints),
color = :viridis, title = "$(year_vec[4])", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[1, :, :])), alpha = 0.75, legend = false, colorbar = :left,
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false, colorbar_title = "Laugh", colorbar_titlefontsize = 20
)

## second component
comp21 = heatmap(x, y, reshape(comp_pred[2, :, 1], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[2, :, :])), alpha = 0.75, legend = false, 
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
comp22 = heatmap(x, y, reshape(comp_pred[2, :, 2], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[2, :, :])), alpha = 0.75, legend = false, 
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
comp23 = heatmap(x, y, reshape(comp_pred[2, :, 3], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[2, :, :])), alpha = 0.75, legend = false,
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
comp24 = heatmap(x, y, reshape(comp_pred[2, :, 4], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[2, :, :])), alpha = 0.75, legend = false, colorbar = :left,
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false, colorbar_title = "Drum", colorbar_titlefontsize = 20
)

## third component
comp31 = heatmap(x, y, reshape(comp_pred[3, :, 1], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[3, :, :])), alpha = 0.75, legend = false, 
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
comp32 = heatmap(x, y, reshape(comp_pred[3, :, 2], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[3, :, :])), alpha = 0.75, legend = false,
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
comp33 = heatmap(x, y, reshape(comp_pred[3, :, 3], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[3, :, :])), alpha = 0.75, legend = false,
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
comp34 = heatmap(x, y, reshape(comp_pred[3, :, 4], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[3, :, :])), alpha = 0.75, legend = false, colorbar = :left,
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false, colorbar_title = "Pik", colorbar_titlefontsize = 20
)

## posterior compositions plots
r1 = plot(comp11, comp12, comp13, comp14, layout = grid(1, 4, widths = [0.225, 0.225, 0.225, 0.325]), size = (2000, 500))
r2 = plot(comp21, comp22, comp23, comp24, layout = grid(1, 4, widths = [0.225, 0.225, 0.225, 0.325]), size = (2000, 500))
r3 = plot(comp31, comp32, comp33, comp34, layout = grid(1, 4, widths = [0.225, 0.225, 0.225, 0.325]), size = (2000, 500))
comp_preds = plot(r1, r2, r3, layout = (3, 1), size = (2000, 1000), dpi = 600)
savefig(comp_preds, "pred_comp.png")

###
### Predicted Compositions (one colorbar)
###

## first component
comp11_2 = heatmap(x, y, reshape(comp_pred[1, :, 1], ngridpoints, ngridpoints),
color = :viridis, title = "$(year_vec[1])", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false, 
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
comp12_2 = heatmap(x, y, reshape(comp_pred[1, :, 2], ngridpoints, ngridpoints),
color = :viridis, title = "$(year_vec[2])", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false,
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
comp13_2 = heatmap(x, y, reshape(comp_pred[1, :, 3], ngridpoints, ngridpoints),
color = :viridis, title = "$(year_vec[3])", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false,
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false
)
comp14_2 = heatmap(x, y, reshape(comp_pred[1, :, 4], ngridpoints, ngridpoints),
color = :viridis, title = "$(year_vec[4])", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false, 
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false
)

## second component
comp21_2 = heatmap(x, y, reshape(comp_pred[2, :, 1], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false, 
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
comp22_2 = heatmap(x, y, reshape(comp_pred[2, :, 2], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false,
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
comp23_2 = heatmap(x, y, reshape(comp_pred[2, :, 3], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false, 
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
comp24_2 = heatmap(x, y, reshape(comp_pred[2, :, 4], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false,
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false
)

## third component
comp31_2 = heatmap(x, y, reshape(comp_pred[3, :, 1], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false, 
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
comp32_2 = heatmap(x, y, reshape(comp_pred[3, :, 2], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false,
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false
)
comp33_2 = heatmap(x, y, reshape(comp_pred[3, :, 3], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false, 
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
)
comp34_2 = heatmap(x, y, reshape(comp_pred[3, :, 4], ngridpoints, ngridpoints),
color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false,
xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false
)

## posterior compositions plots (one color bar)
r1_2 = plot(comp11_2, comp12_2, comp13_2, comp14_2, layout = grid(1, 4, widths = [0.25, 0.25, 0.25, 0.25]), size = (2000, 500))
r2_2 = plot(comp21_2, comp22_2, comp23_2, comp24_2, layout = grid(1, 4, widths = [0.25, 0.25, 0.25, 0.25]), size = (2000, 500))
r3_2 = plot(comp31_2, comp32_2, comp33_2, comp34_2, layout = grid(1, 4, widths = [0.25, 0.25, 0.25, 0.25]), size = (2000, 500))
comp_preds_2 = plot(r1_2, r2_2, r3_2, layout = (3, 1), size = (2000, 1000), dpi = 600)
cb = scatter([0,0], [0,1], zcolor = [0,3], clims = extrema(skipmissing(comp_pred[:, :, :])), xlims = (1, 1.1), xshowaxis = false, yshowaxis = false, label = "", c = :viridis, colorbar_title = "Posterior Predicted Compositions", grid=false)
comp_preds_2_cb = plot(comp_preds_2, cb, layout = grid(1, 2, widths = [0.875, 0.125]), dpi = 600)
savefig(comp_preds_2_cb, "pred_comp_2.png")

###
### Save ESAG MCMC and eta predictions
###

@save "MCMCoutput.jld2" out etaPred_mat

###
### ESAG+ Code
###

if trunc_flag

    println("Truncated ESAG MCMC:")
    out_trunc = mcmc_base(Y, X, z, spatial_idx, nMCMC, nBurn, nSamp, 1, 5, true, distMat[1:m, 1:m], phi_alpha, phi_theta)

    ## solve for thinning factor
    max_lag = Int(0.5*nMCMC)
    ac = StatsBase.autocor(out_trunc["phi"][1, :], 0:max_lag)
    lags = 0:length(ac)-1 # lags for the full autocorrelation
    lags_limited = 0:max_lag
    println("Suggested thin: ", thin)
    thin = findfirst(x -> x <= 0.1, ac)
    phi_chains = Chains(out_trunc["phi"]')
    println("Phi diagnostics: ")
    display(ess_rhat(phi_chains))
    eta_chains_1 = [Chains(out_trunc["eta"][1, i, :]) for i in 1:5]
    eta_chains_2 = [Chains(out_trunc["eta"][2, i, :]) for i in 1:5]
    eta_chains_3 = [Chains(out_trunc["eta"][3, i, :]) for i in 1:5]
    println("Eta diagnostics: ")
    [display(ess_rhat(eta_chains_1[i])) for i in 1:5]
    [display(ess_rhat(eta_chains_2[i])) for i in 1:5]
    [display(ess_rhat(eta_chains_3[i])) for i in 1:5]

    ###
    ### ESAG⁺ Traces
    ###

    ## β trace plots
    plots = []
    for i in 1:d
        for j in 1:p
            tmpPlot = plot(out_trunc["B"][i, j, :], title = "β[$i, $j]")
            push!(plots, tmpPlot)
        end
    end
    plot(plots..., layout = (d, p), size = (900, 600), left_margin = 10mm, legend = false)
    savefig("betaTraces_ESAG_trunc.pdf")

    ## α trace plots
    plots = []
    for i in 1:q
        tmpPlot = plot(out_trunc["alpha"][i, :], title = "α[$i]")
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
            push!(plots, tmpPlot)
        end
    end
    plot(plots..., layout = (d, d), size = (900, 600), left_margin = 10mm, legend = false)
    savefig("AmTraces_ESAG_trunc.pdf")

    ## phi indices
    plots = []
    for i in 1:d
        tmpPlot = plot(out_trunc["phi"][i, :], title = "phi[$i]")
        push!(plots, tmpPlot)
    end
    plot(plots..., layout = (d, 1), size = (900, 600), left_margin = 10mm, legend = false)
    savefig("phiTraces_ESAG_trunc.pdf")

    ###
    ### Predicted η Surface
    ###

    etaPred = zeros(n_grid*d, nKeep)
    let
        Sig_eta_oo = zeros(m*d, m*d)
        Sig_eta_no = zeros((n_aug-m)*d, m*d)
        Sig_eta_nn = zeros((n_aug-m)*d, (n_aug-m)*d)
        println("Kriging: ")
        for k in ProgressBar((nMCMC-nBurn+1-nKeep+1):(nMCMC-nBurn+1))
            if k % 10 == 0 # periodically clean memory
                GC.gc()
            end
            fill!(Sig_eta_oo, 0.0) # reset matrices to 0
            fill!(Sig_eta_no, 0.0)
            fill!(Sig_eta_nn, 0.0)
            for j in 1:d 
                AAt = out_trunc["Am"][:, j, k]*out_trunc["Am"][:, j, k]'
                R_oo = m32.(distMat[1:m, 1:m], out_trunc["phi"][j, k])
                R_no = m32.(distMat[(m+1):n_aug, 1:m], out_trunc["phi"][j, k])
                R_nn = m32.(distMat[(m+1):n_aug, (m+1):n_aug], out_trunc["phi"][j, k])
                Sig_eta_oo += kronecker(R_oo, AAt)
                Sig_eta_no += kronecker(R_no, AAt)
                Sig_eta_nn += kronecker(R_nn, AAt)
            end
            Sig_eta_oo_i = inv(Sig_eta_oo)
            muStar = Sig_eta_no*Sig_eta_oo_i*vec(out_trunc["eta"][:, :, k])
            SigStar = Sig_eta_nn - Sig_eta_no*Sig_eta_oo_i*Sig_eta_no'
            if !isposdef(SigStar)
                SigStar = (SigStar + SigStar') / 2
            end
            etaPred[:, k-((nMCMC-nBurn+1-nKeep+1))+1] = rand(MvNormal(muStar, Hermitian(SigStar)), 1)
        end
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
    scatter!(twinx(), eachcol(obs_locs)..., marker_z = Y[:, 1], xmirror = true, widen = false,
        clim = extrema(Y[:, 1]), color = :viridis, msw = 0.25, legend = false, colorbar = :right,
        xlims = extrema(obs_locs[:, 1]), ylims = extrema(obs_locs[:, 2]), colorbar_title = "Y₁"
    )

    ## spatial field #2
    pred2_trunc = heatmap(x, y, reshape(etaPred_mat[2, :], ngridpoints, ngridpoints)',
        color = :viridis, title = "", label = "", widen = false,
        clim = extrema(etaPred_mat[2, :]), alpha = 0.75, legend = false, colorbar = :left,
        xlims = extrema(x), ylims = extrema(y), colorbar_title = "η₂"
    )
    scatter!(twinx(), eachcol(obs_locs)..., marker_z = Y[:, 2], xmirror = true, widen = false,
        clim = extrema(Y[:, 2]), color = :viridis, msw = 0.25, legend = false, colorbar = :right,
        xlims = extrema(obs_locs[:, 1]), ylims = extrema(obs_locs[:, 2]), colorbar_title = "Y₂"
    )

    ## spatial field #3
    pred3_trunc = heatmap(x, y, reshape(etaPred_mat[3, :], ngridpoints, ngridpoints)',
        color = :viridis, title = "", label = "", widen = false,
        clim = extrema(etaPred_mat[3, :]), alpha = 0.75, legend = false, colorbar = :left,
        xlims = extrema(x), ylims = extrema(y), colorbar_title = "η₃"
    )
    scatter!(twinx(), eachcol(obs_locs)..., marker_z = Y[:, 3], xmirror = true, widen = false,
        clim = extrema(Y[:, 3]), color = :viridis, msw = 0.25, legend = false, colorbar = :right,
        xlims = extrema(obs_locs[:, 1]), ylims = extrema(obs_locs[:, 2]), colorbar_title = "Y₃"
    )

    pred_obs = plot(pred1_trunc, pred2_trunc, pred3_trunc, layout = (3, 1), size = (600, 1000))
    savefig(pred_obs, "pred_obs_trunc.pdf")

    ###
    ### Predicted compositions
    ###

    ## posterior betas
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

    ## predict compositions
    year_vec = 2020:2023
    comp_pred = missings(Float64, d, n_grid, 4)
    for j in 1:4

        ## preparing grid
        if j == 1
            XG = [ones(n_grid) Matrix(grid2020)[:, 4:5]]
        elseif j == 2
            XG = [ones(n_grid) Matrix(grid2021)[:, 4:5]]
        elseif j == 3
            XG = [ones(n_grid) Matrix(grid2022)[:, 4:5]]
        else
            XG = [ones(n_grid) Matrix(grid2023)[:, 4:5]]
        end
        [XG[:, i] = dirStandard_grid(XG[:, i], X_orig[:, i]) for i in 2:p]
        validIdx = findall(row -> !any(ismissing, row), eachrow(XG))
        validNo = length(validIdx)

        ## posterior mu (grid)
        mu_list = zeros(d, n_grid)
        [mu_list[:, validIdx[i]] = link.(B_post*XG[validIdx[i], :] + etaPred_mat[:, validIdx[i]]) for i in 1:validNo]

        ## posterior V, V_inv, and nc
        V_list = zeros(d, d, n_grid)
        Vinv_list = zeros(d, d, n_grid)
        [V_list[:, :, validIdx[i]] = get_V(mu_list[:, validIdx[i]], gamma_post, d, get_xi = false) for i in 1:validNo]
        [Vinv_list[:, :, validIdx[i]] = inv(V_list[:, :, validIdx[i]]) for i in 1:validNo]
        nc = zeros(n_grid)
        [nc[validIdx[i]] = normConstant(mu_list[:, validIdx[i]], V_list[:, :, validIdx[i]], nSamp) for i in 1:validNo]

        ## posterior compositional predictions
        println("Predicting Compositions for $(year_vec[j]):")
        [comp_pred[:, validIdx[i], j] = estComp(mu_list[:, validIdx[i]], V_list[:, :, validIdx[i]], 50000, trunc = true) for i in ProgressBar(1:validNo)]
    end

    ###
    ### Predicted Compositions (d colorbars)
    ###

    ## set-up
    ngridpoints = Int(sqrt(n_grid))
    x = unique(grid_coords[:, 2])
    y = unique(grid_coords[:, 1])

    ## first component
    comp11 = heatmap(x, y, reshape(comp_pred[1, :, 1], ngridpoints, ngridpoints),
    color = :viridis, title = "$(year_vec[1])", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[1, :, :])), alpha = 0.75, legend = false,
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false
    )
    comp12 = heatmap(x, y, reshape(comp_pred[1, :, 2], ngridpoints, ngridpoints),
    color = :viridis, title = "$(year_vec[2])", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[1, :, :])), alpha = 0.75, legend = false, 
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
    )
    comp13 = heatmap(x, y, reshape(comp_pred[1, :, 3], ngridpoints, ngridpoints),
    color = :viridis, title = "$(year_vec[3])", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[1, :, :])), alpha = 0.75, legend = false,
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
    )
    comp14 = heatmap(x, y, reshape(comp_pred[1, :, 4], ngridpoints, ngridpoints),
    color = :viridis, title = "$(year_vec[4])", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[1, :, :])), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false, colorbar_title = "Laugh", colorbar_titlefontsize = 20
    )

    ## second component
    comp21 = heatmap(x, y, reshape(comp_pred[2, :, 1], ngridpoints, ngridpoints),
    color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[2, :, :])), alpha = 0.75, legend = false, 
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
    )
    comp22 = heatmap(x, y, reshape(comp_pred[2, :, 2], ngridpoints, ngridpoints),
    color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[2, :, :])), alpha = 0.75, legend = false, 
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
    )
    comp23 = heatmap(x, y, reshape(comp_pred[2, :, 3], ngridpoints, ngridpoints),
    color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[2, :, :])), alpha = 0.75, legend = false,
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
    )
    comp24 = heatmap(x, y, reshape(comp_pred[2, :, 4], ngridpoints, ngridpoints),
    color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[2, :, :])), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false, colorbar_title = "Drum", colorbar_titlefontsize = 20
    )

    ## third component
    comp31 = heatmap(x, y, reshape(comp_pred[3, :, 1], ngridpoints, ngridpoints),
    color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[3, :, :])), alpha = 0.75, legend = false, 
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
    )
    comp32 = heatmap(x, y, reshape(comp_pred[3, :, 2], ngridpoints, ngridpoints),
    color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[3, :, :])), alpha = 0.75, legend = false,
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
    )
    comp33 = heatmap(x, y, reshape(comp_pred[3, :, 3], ngridpoints, ngridpoints),
    color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[3, :, :])), alpha = 0.75, legend = false,
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
    )
    comp34 = heatmap(x, y, reshape(comp_pred[3, :, 4], ngridpoints, ngridpoints),
    color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[3, :, :])), alpha = 0.75, legend = false, colorbar = :left,
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false, colorbar_title = "Pik", colorbar_titlefontsize = 20
    )

    ## posterior compositions plots
    r1 = plot(comp11, comp12, comp13, comp14, layout = grid(1, 4, widths = [0.225, 0.225, 0.225, 0.325]), size = (2000, 500))
    r2 = plot(comp21, comp22, comp23, comp24, layout = grid(1, 4, widths = [0.225, 0.225, 0.225, 0.325]), size = (2000, 500))
    r3 = plot(comp31, comp32, comp33, comp34, layout = grid(1, 4, widths = [0.225, 0.225, 0.225, 0.325]), size = (2000, 500))
    comp_preds = plot(r1, r2, r3, layout = (3, 1), size = (2000, 1000), dpi = 600)
    savefig(comp_preds, "pred_comp_trunc.png")

    ###
    ### Predicted Compositions (one colorbar)
    ###

    ## first component
    comp11_2 = heatmap(x, y, reshape(comp_pred[1, :, 1], ngridpoints, ngridpoints),
    color = :viridis, title = "$(year_vec[1])", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false, 
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
    )
    comp12_2 = heatmap(x, y, reshape(comp_pred[1, :, 2], ngridpoints, ngridpoints),
    color = :viridis, title = "$(year_vec[2])", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false,
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
    )
    comp13_2 = heatmap(x, y, reshape(comp_pred[1, :, 3], ngridpoints, ngridpoints),
    color = :viridis, title = "$(year_vec[3])", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false,
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false
    )
    comp14_2 = heatmap(x, y, reshape(comp_pred[1, :, 4], ngridpoints, ngridpoints),
    color = :viridis, title = "$(year_vec[4])", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false, 
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false
    )

    ## second component
    comp21_2 = heatmap(x, y, reshape(comp_pred[2, :, 1], ngridpoints, ngridpoints),
    color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false, 
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
    )
    comp22_2 = heatmap(x, y, reshape(comp_pred[2, :, 2], ngridpoints, ngridpoints),
    color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false,
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
    )
    comp23_2 = heatmap(x, y, reshape(comp_pred[2, :, 3], ngridpoints, ngridpoints),
    color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false, 
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
    )
    comp24_2 = heatmap(x, y, reshape(comp_pred[2, :, 4], ngridpoints, ngridpoints),
    color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false,
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false
    )

    ## third component
    comp31_2 = heatmap(x, y, reshape(comp_pred[3, :, 1], ngridpoints, ngridpoints),
    color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false, 
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
    )
    comp32_2 = heatmap(x, y, reshape(comp_pred[3, :, 2], ngridpoints, ngridpoints),
    color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false,
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false
    )
    comp33_2 = heatmap(x, y, reshape(comp_pred[3, :, 3], ngridpoints, ngridpoints),
    color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false, 
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false 
    )
    comp34_2 = heatmap(x, y, reshape(comp_pred[3, :, 4], ngridpoints, ngridpoints),
    color = :viridis, title = "", titlefontsize = 15, label = "", widen = false,
    clim = extrema(skipmissing(comp_pred[:, :, :])), alpha = 0.75, legend = false,
    xlims = extrema(x), ylims = extrema(y), xticks = false, yticks = false
    )

    ## posterior compositions plots (one color bar)
    r1_2 = plot(comp11_2, comp12_2, comp13_2, comp14_2, layout = grid(1, 4, widths = [0.25, 0.25, 0.25, 0.25]), size = (2000, 500))
    r2_2 = plot(comp21_2, comp22_2, comp23_2, comp24_2, layout = grid(1, 4, widths = [0.25, 0.25, 0.25, 0.25]), size = (2000, 500))
    r3_2 = plot(comp31_2, comp32_2, comp33_2, comp34_2, layout = grid(1, 4, widths = [0.25, 0.25, 0.25, 0.25]), size = (2000, 500))
    comp_preds_2 = plot(r1_2, r2_2, r3_2, layout = (3, 1), size = (2000, 1000), dpi = 600)
    cb = scatter([0,0], [0,1], zcolor = [0,3], clims = extrema(skipmissing(comp_pred[:, :, :])), xlims = (1, 1.1), xshowaxis = false, yshowaxis = false, label = "", c = :viridis, colorbar_title = "Posterior Predicted Compositions", grid=false)
    comp_preds_2_cb = plot(comp_preds_2, cb, layout = grid(1, 2, widths = [0.875, 0.125]), dpi = 600)
    savefig(comp_preds_2_cb, "pred_comp_trunc_2.png")

    ###
    ### Logarithmic Score (ESAG+)
    ###

    ## posterior mus
    alpha_post = mean(out_trunc["alpha"], dims = 2)[:, 1]
    A_post = Matrix(kronecker(ones(d), alpha_post'))
    mu_ps = [link.(B_post*X[i, :] + etaPred_mat[:, i] + A_post*z[i, :]) for i in 1:n]

    ## compute V
    V_ps = zeros(d, d, n)
    Vinv_ps = zeros(d, d, n)
    xi_ps = zeros(d, d, n)
    [(V_ps[:, :, i], xi_ps[:, :, i]) = get_V(mu_ps[i], gamma_post, d, get_xi = true) for i in 1:n]
    [Vinv_ps[:, :, i] = inv(V_ps[:, :, i]) for i in 1:n]

    ## compute logS
    logS_trunc = [-dESAG(Y[i, :], mu_ps[i], Vinv_ps[:, :, i], logm = true, trunc = trunc) for i in 1:n]
    logS_mean_trunc = mean(logS_trunc)
    println("logS_trunc: ", logS_mean_trunc)

    ###
    ### Save ESAG+ MCMC and eta predictions
    ###

    @save "MCMCoutput_trunc.jld2" out_trunc etaPred_mat logS_mean_trunc

end
