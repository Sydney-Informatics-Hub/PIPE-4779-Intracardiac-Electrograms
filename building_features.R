
library(here)
library(tidyverse)

labelled_data_path <- here::here("data","S18","labelled")

main_data_path <- here::here("data","S18","Export_Analysis-01_05_2024-17-02-34")

LabelledSignalData <- readRDS(file = here::here(labelled_data_path,"LabelledSignalData.rds"))

# Accessing nested data requires something like..
thenumbers <- LabelledSignalData %>%
  filter(WaveFront == "SR",Catheter_Type == "Penta", Point_Number == 4815)  %>% pull(signal) %>% unlist()
thedataframe <- LabelledSignalData %>%
  filter(WaveFront == "SR",Catheter_Type == "Penta", Point_Number == 4815)  %>% pull(signal)

mean_positive_values <- function(list_containing_df) {
  data <- list_containing_df %>% unlist()
  positive_data <- data[data > 0]
  return(mean(positive_data))
}

sum_positive_values <- function(list_containing_df) {
  data <- list_containing_df %>% unlist()
  positive_data <- data[data > 0]
  return(sum(positive_data))
}

# Function to apply Fourier transformation to a numeric vector
fourier_transform <- function(x) {
  fft_result <- fft(x)
  return(fft_result)
}

test <- LabelledSignalData %>% rowwise() %>%
  mutate(mean = mean(signal %>% unlist(),na.rm = T),
         standard_deviation = sd(signal %>% unlist(),na.rm = T),
         sum = sum(signal %>% unlist(),na.rm = T),
         positivesum = sum_positive_values(signal),
         positivemean = mean_positive_values(signal),
         duration = length((signal %>% unlist())),
         positivesumcheck = sapply(signal, function(x) sum(x[x > 0])),
         positivemeancheck = sapply(signal, function(x) mean(x[x > 0])),
         fourier_signal = lapply(signal, fourier_transform))

check <- test %>%
  filter(WaveFront == "SR",Catheter_Type == "Penta", Point_Number == 4815)

