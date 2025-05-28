library(ggplot2)
library(patchwork)

# Northeastern US state boundaries ---------------------------------------------
ne_region <-
  states(cb = TRUE, class = 'sf') %>%
  filter(STUSPS %in% c('CT', 'ME', 'MA', 'NH', 'NJ', 'NY', 'RI', 'VT', 'PA', 'DE', 'MD'))

# Load in covariates and stack -------------------------------------------------
load('grid_spring_60_2020.RData')
load('grid_spring_60_2021.RData')
load('grid_spring_60_2022.RData')
load('grid_spring_60_2023.RData')
all_years_grid_data <- rbind(
  grid_spring_60_2020,
  grid_spring_60_2021,
  grid_spring_60_2022,
  grid_spring_60_2023
)

# initialize lists with different parameters to iterate through ----------------
covs <- c('avg_temp', 'avg_precip')
years <- c(2020, 2021, 2022, 2023)
labels <- c('Average Temperature (\u00B0C)', 'Average Precipitation (mm)')
plot_list <- c()

# plot each cov for each year --------------------------------------------------
for (i in seq_along(covs)) {
  for (yr in years) {
    # Filter for specific year and crop
    grid_data <- all_years_grid_data %>% filter(year == yr)
    grid_sf <- st_as_sf(grid_data,
                        coords = c("Var2", "Var1"),
                        crs = st_crs(ne_region))
    intersecting_indices <- st_intersects(grid_sf, ne_region, sparse = FALSE)
    grid_filtered <- grid_sf[rowSums(intersecting_indices) > 0, ]
    grid_filtered <- grid_filtered %>%
      mutate(x = st_coordinates(geometry)[, 1], y = st_coordinates(geometry)[, 2])
    
    # Fix: Ensure `cov` correctly references the numeric column
    plot <- ggplot() +
      geom_tile(data = grid_filtered, aes(
        x = x,
        y = y,
        fill = !!sym(covs[i])
      )) +
      geom_sf(
        data = ne_region,
        fill = NA,
        color = 'black',
        linewidth = 0.6
      ) +
      scale_fill_viridis(
        na.value = 'white',
        name = '',
        breaks = seq(
          min(grid_data[, covs[i]], na.rm = TRUE),
          max(grid_data[, covs[i]], na.rm = TRUE),
          length.out = 5
        ),
        # Add 5 breaks in the color bar
        labels = round(seq(
          min(grid_data[, covs[i]], na.rm = TRUE),
          max(grid_data[, covs[i]], na.rm = TRUE),
          length.out = 5
        ))
      ) +
      theme_void() +
      theme(legend.position = 'none')
    
    if (covs[i] == 'avg_temp') {
      plot <- plot +
        labs(title = yr) +
        theme(plot.title = element_text(hjust = 0.5, size = 15, face = 'bold'))
    }
    if (yr == 2023) {
      plot <- plot +
        labs(tag = labels[i]) +
        theme(
          plot.tag = element_text(angle = 90, size = 10, face = 'bold'),
          plot.tag.position = c(1.05, 0.45)
        )
    }
    plot_list[[length(plot_list) + 1]] <- plot
  }
}

# gather each covariate into a separate row
row_1 <- wrap_plots(plot_list[1:4], nrow = 1, ncol = 4) +
  plot_layout(guides = "collect") +
  theme(legend.position = c(1, 0.5), legend.byrow = TRUE) +
  guides(fill = guide_colourbar(theme = theme(
    legend.key.width  = unit(1, "lines"),
    legend.key.height = unit(7, "lines")
  )))

row_2 <- wrap_plots(plot_list[5:8], nrow = 1, ncol = 4) +
  plot_layout(guides = "collect") +
  theme(legend.position = c(1, 0.5), legend.byrow = TRUE) +
  guides(fill = guide_colourbar(theme = theme(
    legend.key.width  = unit(1, "lines"),
    legend.key.height = unit(7, "lines")
  )))

# Combine into a 2x4 plot
covs_combined <- (row_1 / row_2) +
  plot_annotation() &
  theme(plot.margin = margin(0, 20, 0, 0)) +  # Adjust right margin
  plot_layout(heights = c(1, 1)) 

# covs_combined

ggsave('covs_across_years.jpeg', plot = covs_combined, dpi = 600)
