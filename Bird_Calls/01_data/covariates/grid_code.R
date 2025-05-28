library(tidyverse)

# Function to extract average Spring climate information for each coordinate ---
get_spring_clim_for_year <- function(climate_type, lat, lon, year) {
  file_path <- paste0(
    "seasonal/",
    climate_type,
    "/",
    year,
    "/",
    "spring_",
    year,
    "_avg_",
    climate_type,
    ".tif"
  )
  seasonal_avg <- raster(file_path)
  point <- SpatialPoints(cbind(lon, lat), proj4string = CRS("+proj=longlat +datum=WGS84"))
  point_transformed <- spTransform(point, crs(seasonal_avg))
  return(extract(seasonal_avg, point_transformed))
}


# Establishing boundaries for prediction region --------------------------------

# Northernmost boundary of Maine --> 47°27′35″N 69°13′28″W
# Southernmost boundary of Maryland  --> 37°54′44″N 75°53′01″W
# Easternmost boundary of Maine --> 44°48′44″N 66°56′43″W
# Westernmost boundary of Pennsylvania --> 39°44′39″N 80°31′10″W

min_lat2 <- 37.912222 # southernmost boundary of Maryland
max_lat2 <- 47.459722 # northernmost boundary of Maine
min_long2 <- -80.519444 # westernmost boundary of Pennsylvania
max_long2 <- -66.945278 # easternmost boundary of Maine

size <- 75 # Dimension of grid
clims <- c('temp', 'precip')
years <- c(2020, 2021, 2022, 2023)

lat_seq2 = seq(from = min_lat2,
               to = max_lat2,
               length.out = size)
long_seq2 = seq(from = min_long2,
                to = max_long2,
                length.out = size)
grid2 = expand.grid(long_seq2, lat_seq2)

# Iterate through year and climate type to create four separate grids-----------

for (year in years) {
  grid_current <- grid2
  grid_current$year <- year
  
  for (climate_type in clims) {
    col_name <- paste0("avg_", climate_type) # add units later
    grid_current[[col_name]] <- mapply(
      get_spring_clim_for_year,
      climate_type = climate_type,
      lat = grid_current$Var2,
      lon = grid_current$Var1,
      year = year
    )
  }
  assign(paste0("grid_spring", '_', size, '_', year), grid_current)
}

# Save grids for export --------------------------------------------------------

# save(grid_spring_75_2020, file = 'grid_spring_75_2020.RData')
# save(grid_spring_75_2021, file = 'grid_spring_75_2021.RData')
# save(grid_spring_75_2022, file = 'grid_spring_75_2022.RData')
# save(grid_spring_75_2023, file = 'grid_spring_75_2023.RData')
