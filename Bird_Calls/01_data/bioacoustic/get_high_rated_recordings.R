library(tidyverse)

meta_data <- read.csv('og_data/ML__2024-06-07T18-02_dowwoo_audio.csv')
ratings <- c(3, 4, 5)

# Filter to get high rated recordings not used in training set------------------
pred_recs_3_5 <- meta_data %>%
  filter(Background.Species == '') %>%
  filter(Average.Community.Rating %in% ratings) %>% 
  filter(Year < 2024) %>% # pre-2024 to ensure there is a recording available
  filter(!(`ML.Catalog.Number` %in% clean_24_all)) %>% # filtering out any training data
  rename(unique_id = ML.Catalog.Number) %>%
  select(unique_id)

# Save for export --------------------------------------------------------------
# write.csv(pred_recs_3_5, file = 'high_rated_recordings_to_predict_3_5_rated.csv')
