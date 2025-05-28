library(tidyverse)
library(sf)
library(tigris)
library(patchwork)
library(viridis)
library(here)
load('eta_pred.RData')

ne_region <-
  states(cb = TRUE, class = 'sf') %>%
  filter(STUSPS %in% c('CT', 'ME', 'MA', 'NH', 'NJ', 'NY', 'RI', 'VT', 'PA', 'DE', 'MD'))

SIZE <- 60 # dimensions of grid 
min_lat <- 37.912222
max_lat <- 47.459722
min_long <- -80.519444
max_long <- -66.945278
lat_seq = seq(from = min_lat,
              to = max_lat,
              length.out = SIZE)
long_seq = seq(from = min_long,
               to = max_long,
               length.out = SIZE)
grid = expand.grid(lat_seq, long_seq)


eta_plots <- c()

for(i in c(1:3)){
  eta <- etaPred_mat[i, 1:3600]
  grid$eta <- eta
  grid_sf <- st_as_sf(grid,
                      coords = c("Var2", "Var1"),
                      crs = st_crs(ne_region))
  intersecting_indices <- st_intersects(grid_sf, ne_region, sparse = FALSE)
  grid_filtered <- grid_sf[rowSums(intersecting_indices) > 0, ]
  grid_filtered <- grid_filtered %>%
    mutate(x = st_coordinates(geometry)[, 1], y = st_coordinates(geometry)[, 2])
  p <- ggplot() +
    geom_tile(data = grid_filtered, aes(x = x, y = y, fill = eta)) +
    scale_fill_viridis(
      na.value = 'white', 
      name = '',
      breaks = seq(min(grid$eta, na.rm = TRUE), 
                   max(grid$eta, na.rm = TRUE), 
                   length.out = 5), # Add 5 breaks in the color bar
      labels = round(seq(min(grid$eta, na.rm = TRUE), 
                         max(grid$eta, na.rm = TRUE), 
                         length.out = 5), 2)
      ) +
    geom_sf(
      data = ne_region,
      fill = NA,
      color = 'black',
      linewidth = 0.6
    ) + theme_void() + 
    labs(title = parse(text = paste("eta[", i, "]", sep = ""))) +
    theme(plot.title = element_text(hjust = 0.5, size = 25, face = 'bold'), legend.position = 'none')
  
  
  eta_plots[[length(eta_plots) + 1]] <- p

}


eta_row_plot <- wrap_plots(eta_plots[1:3], nrow = 1, ncol = 3) +
  plot_layout(guides = "collect") +
  theme(legend.position = c(1, 0.5), legend.byrow = TRUE) +
  guides(fill = guide_colourbar(theme = theme(
    legend.key.width  = unit(1.5, "lines"),
    legend.key.height = unit(10, "lines")
  )))

# eta_row_plot
ggsave(here('FINAL_PIPELINE','03_summarize','eta_plot.jpeg'), plot = eta_row_plot, dpi = 600)
