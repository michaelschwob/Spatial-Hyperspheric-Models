library(raster)

# Summarize monthly information into seasonal information----------------------

# List of monthly files 
files <- list.files(path = "prism_data/precip/PRISM_ppt_stable_4kmM3_2022_all_bil", 
                         pattern = "PRISM_ppt_stable_4kmM3_2022[0-9][0-9]_bil\\.bil$", 
                         full.names = TRUE)

# Read raster files
monthly_rasters <- lapply(files, raster)

# Combine into a single raster stack
raster_stack <- stack(monthly_rasters)

# Create a function to calculate seasonal average
calc_seasonal_avg <- function(month_indices) {
  seasonal_stack <- subset(raster_stack, month_indices)
  seasonal_avg <- calc(seasonal_stack, fun = mean, na.rm = TRUE)
  return(seasonal_avg)
}

# Define month indices for each season
winter_indices <- c(12, 1, 2)    # Dec, Jan, Feb
spring_indices <- c(3, 4, 5)     # Mar, Apr, May
summer_indices <- c(6, 7, 8)     # Jun, Jul, Aug
fall_indices <- c(9, 10, 11)     # Sept, Oct, Nov 

# Calculate seasonal averages
winter_avg <- calc_seasonal_avg(winter_indices)
spring_avg <- calc_seasonal_avg(spring_indices)
summer_avg <- calc_seasonal_avg(summer_indices)
fall_avg <- calc_seasonal_avg(fall_indices)

# Save for export --------------------------------------------------------------
# writeRaster(winter_avg, "seasonal/temp/2023/winter_2023_avg_temp", format = "GTiff", overwrite=TRUE)
# writeRaster(spring_avg, "seasonal/temp/2023/spring_2023_avg_temp", format = "GTiff", overwrite=TRUE)
# writeRaster(summer_avg, "seasonal/temp/2023/summer_2023_avg_temp", format = "GTiff", overwrite=TRUE)
# writeRaster(fall_avg, "seasonal/temp/2023/fall_2023_avg_temp", format = "GTiff", overwrite=TRUE)