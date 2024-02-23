
library(here)
library(tidyverse)
library(tidyverse)
library(tidymodels)
source("paths.R")
library(pracma)
library(circular)

get_paths()


LabelledSignalData <- readRDS(file = here::here(generated_data_path,"cleaned_aggregate_data.rds"))


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

compute_fourier_transform <- function(vector_data) {
  fft_result <- fft(vector_data)
  magnitude <- Mod(fft_result)
  phase <- Arg(fft_result)
  return(data.frame(magnitude = magnitude, phase = phase))
}


LabelledSignalData <- LabelledSignalData %>% rowwise() %>%
  mutate(mean = mean(signal %>% unlist(),na.rm = T),
         standard_deviation = sd(signal %>% unlist(),na.rm = T),
         sum = sum(signal %>% unlist(),na.rm = T),
         positivesum = sum_positive_values(signal),
         positivemean = mean_positive_values(signal),
         duration = length((signal %>% unlist())),
         positivesumcheck = sapply(signal, function(x) sum(x[x > 0])),
         positivemeancheck = sapply(signal, function(x) mean(x[x > 0])),
         fourier_features = lapply(signal,compute_fourier_transform))


#extract circular mean and variance from fourier features to aggregate info into a feature.



LabelledSignalData <- LabelledSignalData %>% rowwise() %>%
  mutate(phase_mean = circular(fourier_features$magnitude) %>% mean(),
         phase_var = circular(fourier_features$phase) %>% var.circular(),
         magnitude_mean = mean(fourier_features$magnitude))


#prepare for a model.

saveRDS(LabelledSignalData, file = here::here(generated_data_path,"model_data.rds"))



