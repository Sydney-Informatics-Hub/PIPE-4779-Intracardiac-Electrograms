library(here)
library(tidyverse)
source("paths.R")
# to work on a 3d plot of predictors versus truth
get_paths()

predictions_file <- here::here("orange_exploration","individual_model_predictions.tab")
predictions <- read.delim(predictions_file, header = TRUE,skip=0)
predictions <- predictions[-c(1:2), ]

predictions <- predictions %>% select(depth_label,X,Y,Z,Point_Number,sheep,WaveFront,
                       Gradient.Boosting..1., Gradient.Boosting..1...error.,
                       Gradient.Boosting..1...NoScar.,
                       Gradient.Boosting..1...AtLeastEndo.,
                       Gradient.Boosting..1...AtLeastIntra.,
                       Gradient.Boosting..1...epiOnly.,
                       )
