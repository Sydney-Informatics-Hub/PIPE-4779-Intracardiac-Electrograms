library(here)
library(tidyverse)
library(plotly)
source("paths.R")
# Predictions are from Basic Model - Filtered data
# see individual_model.ows orange file
# to work on a 3d plot of predictors versus truth
get_paths()


#"individual_model_predictions.tab"
get_predictions <- function(predictions_file){
  predictions_file <- here::here("orange_exploration",predictions_file)
  predictions <- read.delim(predictions_file, header = TRUE,skip=0)
  predictions <- predictions[-c(1:2), ]

  predictions <- predictions %>% select(depth_label,X,Y,Z,Point_Number,sheep,WaveFront,
                                        Gradient.Boosting..1.,
                                        Gradient.Boosting..1...NoScar.,
                                        Gradient.Boosting..1...AtLeastEndo.,
                                        Gradient.Boosting..1...AtLeastIntra.,
                                        Gradient.Boosting..1...epiOnly.,
  )

  predictions <- predictions %>% rename(Prediction = Gradient.Boosting..1.) %>%
    select(depth_label,X,Y,Z,Point_Number,sheep,WaveFront,Prediction)

  predictions <- predictions %>%
    mutate(is_correct = ifelse(Prediction == depth_label,"Correct","Incorrect"))

  return(predictions)
}


plot_all_wavefront <- function(data,select_sheep,data_type) {
  data_plot <- data %>% filter(sheep == select_sheep)

  fig <- plot_ly(data_plot, x = ~X, y = ~Y, z = ~Z, type = "scatter3d", mode = "markers", marker = list(size = 3),
                 color = ~is_correct, colors = c("blue", "red")) %>%
    layout(title = paste0("Sheep ",select_sheep," ",data_type))

  fig

}

data <- get_predictions("individual_model_predictions.tab")
plot_all_wavefront(data,"S20","Test data")

data <- get_predictions("predictions_3d_all_data.tab")
plot_all_wavefront(data,"S20"," All data")

