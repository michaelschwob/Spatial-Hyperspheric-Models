setwd('~/birdsongs')
library(stringr)
library(dplyr)
library(tidyr)
library(sf)
library(raster)
#library(terra)

# Process prediction results and associated meta data---------------------------
pred_and_meta_data_3_5 <- read.csv('app_data/three_five_star_pred_and_meta_data.csv')

seasonal <- pred_and_meta_data_3_5 %>%
  #mutate(county = str_trim(County), state = str_trim(State)) %>%
  mutate(
    season = case_when(
      Month %in%  c(9:11) ~ "fall",
      Month %in%  c(12, 1, 2)  ~ "winter",
      Month %in%  c(3:5)  ~ "spring",
      Month %in% c(6:8) ~ "summer"
    )
  )

cleaned <-
  seasonal %>%
  filter(
    State %in% c(
      "Connecticut",
      "Maine",
      "Massachusetts",
      "New Hampshire",
      "Rhode Island",
      "Vermont",
      "New Jersey",
      "New York",
      "Pennsylvania",
      "Delaware",
      "Maryland"
    )
  ) %>%
  filter(Year %in% c(2020, 2021, 2022, 2023)) %>%
  dplyr::select(
    unique_id,
    Year,
    Month,
    Latitude,
    Longitude,
    County,
    State,
    Average.Community.Rating,
    season,
    laugh,
    drum,
    pik
  ) %>%
  rename(avg_rating = Average.Community.Rating) %>%
  na.omit()


# Function to extract average temperature and precipitation information---------
get_seasonal_clim_for_year <- function(climate_type, lat, lon, season, year) {
  file_path <- paste0(
    "seasonal/",
    climate_type,
    "/",
    year,
    "/",
    season,
    '_',
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

# Extract average values for the observations used to fit the model-------------
cleaned$avg_temp_deg_c <- mapply(
  get_seasonal_clim_for_year,
  'temp',
  cleaned$Latitude,
  cleaned$Longitude,
  cleaned$season,
  cleaned$Year
)

cleaned$avg_precip_mm <- mapply(
  get_seasonal_clim_for_year,
  'precip',
  cleaned$Latitude,
  cleaned$Longitude,
  cleaned$season,
  cleaned$Year
)

cleaned <- cleaned %>%
  dplyr::select(
    unique_id,
    Year,
    Month,
    Latitude,
    Longitude,
    county,
    state,
    season,
    avg_rating,
    pop_density,
    avg_precip_mm,
    avg_temp_deg_c,
    laugh,
    drum,
    pik
  )
