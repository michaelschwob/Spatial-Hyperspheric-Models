#############################################################
####
#### Julia MCMC Script to Estimate ESAG Regression Parameters
#### Without Latent Spatial Random Effects
####
#############################################################

function mcmc_nonspatial(Y, X, z, spatial_idx, nMCMC, nBurn, nSamp, startIdx, endIdx, truncation, R_list, phiIdx_vec)

    ## set-up
    n, d = size(Y)
    m = size(R_list, 1)
    p = size(X, 2)
    q = size(z, 2)
    non_base = [2, 3] # indices for components that are not the base

    ## for beta update
    sd_beta = 0.5
    mu_beta = zeros(p)
    Sig_beta = Diagonal(repeat([1e1], p))

    ## for gamma update
    sd_gamma = 0.1
    mu_gamma = 0.0
    Sig_gamma = 1e1

    ## for alpha update 
    sd_alpha = 0.1
    mu_alpha = zeros(q)
    Sig_alpha = 1e1*diagm(ones(q))

    ## initializations
    B = rand(Uniform(1, 2), d, p)
    B[1, :] = zeros(p)
    alpha = rand(Uniform(1, 2), q)
    A = Matrix(kronecker(ones(d), alpha'))
    eta_vec = zeros(d*m)
    eta_mat = reshape(eta_vec, d, m)

    ## rest of parameters
    mu = [link.(B*X[i, :] + eta_mat[:, spatial_idx[i]] + A*z[i, :]) for i in 1:n]
    gamma = [rand(Truncated(Normal(0, 1), 0, Inf), j+1) for j in 1:(d-2)] # normal prior truncated to R⁺
    V = zeros(d, d, n)
    Vinv = zeros(d, d, n)
    xi = zeros(d, d, n)
    [(V[:, :, i], xi[:, :, i]) = get_V(mu[i], gamma, d, get_xi = true) for i in 1:n]
    [Vinv[:, :, i] = inv(V[:, :, i]) for i in 1:n]
    VStar = deepcopy(V)
    VinvStar = deepcopy(Vinv)
    xiStar = deepcopy(xi)
    nc = zeros(n)
    if truncation
        nc = NCP(mu, V, nSamp)
    end
    ncStar = copy(nc)

    ## save matrices
    B_save = zeros(d, p, (nMCMC-nBurn+1))
    eta_save = zeros(d, m, (nMCMC-nBurn+1))
    alpha_save = zeros(q, (nMCMC-nBurn+1))
    gamma_save = []
    V_save = zeros(endIdx-startIdx+1, d, d, (nMCMC-nBurn+1))
    Vinv_save = zeros(endIdx-startIdx+1, d, d, (nMCMC-nBurn+1))
    xi_save = zeros(endIdx-startIdx+1, d, d, (nMCMC-nBurn+1))
    essCounts = zeros(nMCMC)
    Am_save = zeros(d, d, (nMCMC-nBurn+1))
    phiIdx_save = zeros(d, (nMCMC-nBurn+1))

    ###
    ### MCMC Iterations
    ###

    for k in ProgressBar(2:nMCMC)
        
        ###
        ### B update
        ###

        for i in 1:d # for each row in B matrix

            if i in non_base # for non-base components, update all betas
                BStar = copy(B)
                BStar[i, :] = rand(MvNormal(B[i, :], (sd_beta^2)*I), 1) # random-walk proposal
            else # for base component, only update the intercept
                BStar = copy(B)
                BStar[i, 1] = rand(Normal(B[i, 1], sd_beta), 1)[1] # random-walk proposal
            end

            ## downstream computations
            muStar = [link.(BStar*X[j, :] + eta_mat[:, spatial_idx[j]] + A*z[j, :]) for j in 1:n]
            [(VStar[:, :, j], xiStar[:, :, j]) = get_V(muStar[j], gamma, d, get_xi = true) for j in 1:n]
            [VinvStar[:, :, j] = inv(VStar[:, :, j]) for j in 1:n]
            if truncation
                ncStar = NCP(muStar, VStar, nSamp) # normalizing constants in parallel
            end

            ## Metropolis-Hastings ratio
            mh1 = logpdf(MvNormal(mu_beta, Sig_beta), BStar[i, :])
            mh2 = logpdf(MvNormal(mu_beta, Sig_beta), B[i, :])
            for j in 1:n
                mh1 += dESAG(Y[j, :], muStar[j], VinvStar[:, :, j], logm = true, trunc = truncation, nc = ncStar[j])
                mh2 += dESAG(Y[j, :], mu[j], Vinv[:, :, j], logm = true, trunc = truncation, nc = nc[j])
            end
            if exp(mh1 - mh2) > rand()
                B = copy(BStar)
                mu = copy(muStar)
                V = copy(VStar)
                Vinv = copy(VinvStar)
                xi = copy(xiStar)
                nc = copy(ncStar)
            end
        end

        ###
        ### α update
        ###

        ## random-walk proposal
        alphaStar = rand(MvNormal(vec(alpha), sd_alpha*diagm(ones(q))), 1)

        ## downstream computations
        AStar = kronecker(ones(d), alphaStar')
        muStar = [link.(B*X[j, :] + eta_mat[:, spatial_idx[j]] + AStar*z[j, :]) for j in 1:n]
        [(VStar[:, :, j], xiStar[:, :, j]) = get_V(muStar[j], gamma, d, get_xi = true) for j in 1:n]
        [VinvStar[:, :, j] = inv(VStar[:, :, j]) for j in 1:n]
        if truncation
            ncStar = NCP(muStar, VStar, nSamp) # normalizing constants in parallel
        end

        ## Metropolis-Hastings ratio 
        mh1 = logpdf(MvNormal(mu_alpha, Sig_alpha), alphaStar)[1]
        mh2 = logpdf(MvNormal(mu_alpha, Sig_alpha), alpha)[1]
        for j in 1:n
            mh1 += dESAG(Y[j, :], muStar[j], VinvStar[:, :, j], logm = true, trunc = truncation, nc = ncStar[j])
            mh2 += dESAG(Y[j, :], mu[j], Vinv[:, :, j], logm = true, trunc = truncation, nc = nc[j])
        end
        if exp(mh1 - mh2) > rand()
            alpha = copy(alphaStar)
            A = copy(AStar)
            mu = copy(muStar)
            V = copy(VStar)
            Vinv = copy(VinvStar)
            xi = copy(xiStar)
            nc = copy(ncStar)
        end

        ###
        ### γ update
        ###
        
        for j in 1:(d-2)
            for i in 1:(j+1)

                ## random-walk proposal
                gammaStar = deepcopy(gamma) # to copy the structure of gamma
                gammaStar[j][i] = rand(Normal(gamma[j][i], sd_gamma), 1)[1]
                
                ## downstream computations
                [(VStar[:, :, l], xiStar[:, :, l]) = get_V(mu[l], gammaStar, d, get_xi = true) for l in 1:n]
                [VinvStar[:, :, l] = inv(VStar[:, :, l]) for l in 1:n]
                if truncation
                    ncStar = NCP(mu, VStar, nSamp) # normalizing constants in parallel
                end

                ## Metropolis-Hastings ratio
                mh1 = logpdf(Normal(mu_gamma, sqrt(Sig_gamma)), gammaStar[j][i])
                mh2 = logpdf(Normal(mu_gamma, sqrt(Sig_gamma)), gamma[j][i])
                for l in 1:n
                    mh1 += dESAG(Y[l, :], mu[l], VinvStar[:, :, l], logm = true, trunc = truncation, nc = ncStar[l])
                    mh2 += dESAG(Y[l, :], mu[l], Vinv[:, :, l], logm = true, trunc = truncation, nc = nc[l])
                end
                if exp(mh1 - mh2) > rand()
                    gamma = copy(gammaStar)
                    V = copy(VStar)
                    Vinv = copy(VinvStar)
                    xi = copy(xiStar)
                    nc = copy(ncStar)
                end
            end
        end

        ###
        ### Save Values
        ###

        if k >= nBurn
            B_save[:, :, (k-nBurn+1)] = copy(B)
            eta_save[:, :, (k-nBurn+1)] = copy(eta_mat)
            alpha_save[:, (k-nBurn+1)] = copy(alpha)
            [V_save[(i-startIdx+1), :, :, (k-nBurn+1)] = V[:, :, i] for i in startIdx:endIdx]
            [Vinv_save[(i-startIdx+1), :, :, (k-nBurn+1)] = Vinv[:, :, i] for i in startIdx:endIdx]
            [xi_save[(i-startIdx+1), :, :, (k-nBurn+1)] = xi[:, :, i] for i in startIdx:endIdx]
            push!(gamma_save, gamma)
            Am_save[:, :, (k-nBurn+1)] = copy(Am)
            phiIdx_save[:, (k-nBurn+1)] = copy(phiIdx_vec)

            ## save MCMC output every 2500 iterations
            if (k%2500 == 0)
                tempDict = Dict("B" => B_save, "eta"  => eta_save, "alpha" => alpha_save, "V" => V_save, "Vinv" => Vinv_save, "xi" => xi_save, "gamma" => gamma_save, "ESS_counts" => essCounts, "phiIdx" => phiIdx_save, "Am" => Am_save)
                lastK = copy(k)
                if truncation
                    @save "tempOutput_trunc.jld2" tempDict lastK
                else
                    @save "tempOutput.jld2" tempDict lastK
                end
            end
        end
    end

    return Dict("B" => B_save, "eta"  => eta_save, "alpha" => alpha_save, "V" => V_save, "Vinv" => Vinv_save, "xi" => xi_save, "gamma" => gamma_save, "ESS_counts" => essCounts, "phiIdx" => phiIdx_save, "Am" => Am_save)
end