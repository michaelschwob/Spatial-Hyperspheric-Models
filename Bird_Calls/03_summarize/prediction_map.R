library(tidyverse)
library(sf)
library(tigris)
library(patchwork)
library(viridis)
library(here)
#setwd('~/birdsongs/')

# Northeastern US state boundaries ---------------------------------------------
ne_region <-
  states(cb = TRUE, class = 'sf') %>%
  filter(STUSPS %in% c('CT', 'ME', 'MA', 'NH', 'NJ', 'NY', 'RI', 'VT', 'PA', 'DE', 'MD'))

# Grid boundaries --------------------------------------------------------------
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

# Load in data -----------------------------------------------------------------
# setwd('birdsongs/')
load('../Downloads/data_locs.RData') # year and coordinates of data points
load('../Downloads/comp_pred_spatial.RData')

# Plotting function ------------------------------------------------------------

plot_prediction <- function(d, t, comp_pred, grid_data, dimen) {
  grid_data$value <- comp_pred[d, 1:dimen ** 2, t]
  
  grid_sf <- st_as_sf(grid_data,
                      coords = c("Var2", "Var1"),
                      crs = st_crs(ne_region))
  intersecting_indices <- st_intersects(grid_sf, ne_region, sparse = FALSE)
  grid_filtered <- grid_sf[rowSums(intersecting_indices) > 0, ]
  grid_filtered <- grid_filtered %>%
    mutate(x = st_coordinates(geometry)[, 1], y = st_coordinates(geometry)[, 2])
  
  years <- c('2020', '2021', '2022', '2023')
  calls <- c('Whinny', 'Drum', 'Pik')
  
  plot <- ggplot() +
    geom_tile(data = grid_filtered, aes(x = x, y = y, fill = value)) +
    geom_point(
      data = data_locs,
      aes(x = Longitude, y = Latitude),
      color = 'black',
      size = 2,
      alpha = 0.8
    ) +
    geom_sf(
      data = ne_region,
      fill = NA,
      color = 'black',
      linewidth = 0.6
    ) +
    # scale_fill_viridis(na.value = 'white', name = '') +
    scale_fill_viridis(
      na.value = 'white', 
      name = '',
      breaks = seq(min(grid_data$value, na.rm = TRUE), 
                   max(grid_data$value, na.rm = TRUE), 
                   length.out = 6), # Add 5 breaks in the color bar
      labels = round(seq(min(grid_data$value, na.rm = TRUE), 
                         max(grid_data$value, na.rm = TRUE), 
                         length.out = 6), 2)) + 
    theme_void() +
    theme(legend.position = 'none')
  
  # if (d == 1) {
  #   plot <- plot + labs(title = years[t]) + theme(plot.title = element_text(hjust = 0.5, size = 20))
  # }
  if (d == 1) {
    plot <- plot + 
      labs(title = years[t]) + 
      theme(plot.title = element_text(hjust = 0.5, size = 20, face = 'bold')) 
      # scale_fill_viridis(
      #   na.value = 'white', 
      #   name = '',
      #   breaks = seq(min(grid_data$value, na.rm = TRUE), 
      #                max(grid_data$value, na.rm = TRUE), 
      #                length.out = 5), # Add 5 breaks in the color bar
      #   labels = round(seq(min(grid_data$value, na.rm = TRUE), 
      #                      max(grid_data$value, na.rm = TRUE), 
      #                      length.out = 5), 2)  # Format the labels to 2 decimal places
      
  }
  if (t == 4) {
    plot <- plot +
      labs(tag = calls[d]) +
      theme(
        plot.tag = element_text(angle = 90, size = 15, face = 'bold'),
        plot.tag.position = c(1.05, 0.5)
      )
  }
  return(plot)
}

# Plot by row (call type) and then stack vertically (by year) ------------------

plot_list <- list()

for (d in seq(1, 3)) {
  for (t in seq(1, 4)) {
    plot_list[[length(plot_list) + 1]] <- plot_prediction(d, t, comp_pred, grid, SIZE)
  }
}

# Whinny
row_1 <- wrap_plots(plot_list[1:4], nrow = 1, ncol = 4) +
  plot_layout(guides = "collect") +
  theme(legend.position = c(1, 0.5), legend.byrow = TRUE) +
  guides(fill = guide_colourbar(theme = theme(
    legend.key.width  = unit(1.5, "lines"),
    legend.key.height = unit(10, "lines")
  )))

# Drum
row_2 <- wrap_plots(plot_list[5:8], nrow = 1, ncol = 4) +
  plot_layout(guides = "collect") +
  theme(legend.position = "right", legend.byrow = TRUE) +
  guides(fill = guide_colourbar(theme = theme(
    legend.key.width  = unit(1.5, "lines"),
    legend.key.height = unit(10, "lines")
  )))

# Pik
row_3 <- wrap_plots(plot_list[9:12], nrow = 1, ncol = 4) +
  plot_layout(guides = "collect") +
  theme(legend.position = "right", legend.byrow = TRUE) +
  guides(fill = guide_colourbar(theme = theme(
    legend.key.width  = unit(1.5, "lines"),
    legend.key.height = unit(10, "lines")
  )))

combined_plot <- (row_1 / row_2 / row_3)
combined_plot

# Save plot for export ---------------------------------------------------------
ggsave(here('FINAL_PIPELINE','03_summarize','spatial_prediction_maps.jpeg'), plot = combined_plot, width = 12, height = 8, units = 'in', dpi = 600)

