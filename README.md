# Spatial-Hyperspheric-Models
Code and Data for Schwob, Hooten, Calzada, Keitt, "Spatial Hyperspheric Models for Compositional Data," in review.

Code in `Bird_Calls` was developed by [Nicholas Calzada](https://github.com/nick-calzada/Bird_Calls).

## Directory Structure

```txt

Bird_Calls/          # Python and R code for the data processing of bioacoustic and climatic information used in the downy woodpecker case study; borrowed from https://github.com/nick-calzada/Bird_Calls
│   
├── 01_data/
│   │   └──all_wav_files                 # Test set 
│   ├── bioacoustic/
│   │   ├── raw_train_data               
│   │   ├── clean                        # Train set (trimmed 1-second recordings)
│   │   ├── spects                       # Mel-spectrograms of cleaned samples                          
│   │   ├── crop_wav_files.ipynb         # Crop categorized .wav files into 1-second clips
│   │   └── get_high_rated_recordings.R  # Extract recordings with a 3-5 star rating
│   │
│   └── covariates/                      
│       ├── grid_code.R                  # Create a regular grid across region of interest with extracted covariate information for predictions of bioacoustic compositions
│       ├── make_datasets.R              # Make datasets of observed calls and associated covariate information
│       └── make_seasonal_data.R         # Make datasets corresponding to the spring of each year analyzed (2020-2023)
│
├── 02_train_and_predict/              
│   ├── train.ipynb                      # Train the model on classified samples
│   └── predict.ipynb                    # Run predictions using trained models on out-of-sample recordings
│
└── 03_summarize/                      
    ├── eta_plot.jpeg                            # Plot of eta coefficients
    ├── plot_etas.R                              # R script to create eta_plot.jpeg
    ├── spect_samples.jpg                        # Example spectrograms used in modeling (Fig. 2)
    ├── spect_fig_and_sample_nums.ipynb          # Plot random sample of spectrograms for each call type (Fig. 2) 
    ├── spatial_prediction_maps.jpeg             # Visual summary of spatial predictions (Fig. 5)
    ├── prediction_map.R                         # R script to generate prediction maps (Fig. 5)
    ├── covs_across_years.jpeg                   # Precipitation and temperature trends from 2020-2023 (Fig. 6) 
    └── plot_avg_temp_and_precip_across_years.R  # R script for temp/precip trends (Fig. 6)

Downy-Woodpecker-Bioacoustics/       # Julia and R code for the downy woodpecker case study
│   
├── downy-woodpecker/          # the case study presented in the manuscript
│   │   ├── script.jl          # the Julia script that prepares the downy woodpecker data, runs the MCMC algorithm, and processes the MCMC output 
│   │   ├── mcmc.jl            # the MCMC algorithm without a baseline component
│   │   ├── mcmc_base.jl       # the complete MCMC algorithm
│   │   └── NE_3_5.RData       # the compositional responses and bioclimatic information obtained from the data processing steps in Bird_Calls 
│   │
│   └── grid_data/                      
│       ├── grid_spring_60_2020.RData         # bioclimatic data across a fine grid spanning the study domain for 2020
│       ├── grid_spring_60_2021.RData         # bioclimatic data across a fine grid spanning the study domain for 2021
│       ├── grid_spring_60_2022.RData         # bioclimatic data across a fine grid spanning the study domain for 2022
│       └── grid_spring_60_2023.RData         # bioclimatic data across a fine grid spanning the study domain for 2023
│
└── downy-woodpecker-nonspatial/    # the modified case study presented in Appendix F, where we did not include latent spatial random effects in our model         
    │   ├── script.jl               # the Julia script that prepares the downy woodpecker data, runs the MCMC algorithm, and processes the MCMC output (without latent spatial random effects)
    │   ├── mcmc.jl                 # the MCMC algorithm without latent spatial random effects and without a baseline component
    │   ├── mcmc_base.jl            # the MCMC algorithm without latent spatial random effects 
    │   └── NE_3_5.RData            # the compositional responses and bioclimatic information obtained from the data processing steps in Bird_Calls
    │
    └── grid_data/                      
        ├── grid_spring_60_2020.RData         # bioclimatic data across a fine grid spanning the study domain for 2020
        ├── grid_spring_60_2021.RData         # bioclimatic data across a fine grid spanning the study domain for 2021
        ├── grid_spring_60_2022.RData         # bioclimatic data across a fine grid spanning the study domain for 2022
        └── grid_spring_60_2023.RData         # bioclimatic data across a fine grid spanning the study domain for 2023

Simulation-Studies/        # Julia and R code for the simulation studies
│   
├── simulation-study/                # the simulation study presented in the manuscript
│       ├── script.jl                # the Julia script that simulates directional data, runs the MCMC algorithm, and processes the MCMC output
│       ├── mcmc_spatial.jl          # the complete MCMC algorithm
│       └── mcmc_nonspatial.jl       # the MCMC algorithm without latent spatial random effects (inference is presented in Appendix F)
│
├── simulation-study-comparison-spatial/        # the simulation study for model comparison in Appendix F (data simulated with latent spatial fields)
│      ├── script.jl                            # the Julia script that simulates directional data with latent spatial fields, runs the MCMC algorithm, and processes the  MCMC output
│      ├── mcmc.jl                              # the MCMC algorithm without latent spatial random effects
│      └── mcmc_base.jl                         # the complete MCMC algorithm
│
└── simulation-study-comparison-nonspatial/     # the simulation study for model comparison in Appendix F (data simulated without latent spatial fields) 
       ├── script.jl                            # the Julia script that simulates directional data without latent spatial fields, runs the MCMC algorithm, and processes the  MCMC output
       ├── mcmc.jl                              # the MCMC algorithm without latent spatial random effects
       ├── mcmc_base.jl                         # the complete MCMC algorithm
       └── simVars.jld2                         # simulation parameters saved from simulation-study-comparison-spatial to be used in this simulation

esag_functions.jl        # a collection of Julia functions for the ESAG and ESAG+ distribution (density evaluation, simulation, mean direction, mode direction, etc.)
```
