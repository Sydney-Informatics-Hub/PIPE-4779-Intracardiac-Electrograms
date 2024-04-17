# Choose data_type
library(here)
library(tidyverse)
library(tidyverse)
library(tidymodels)
source("paths.R")
library(pracma)
library(circular)
get_paths()


data_type <- "filtered"
#data_type <- "imputed"

LabelledSignalData <- readRDS(file = here::here(generated_data_path,paste0(data_type,"_aggregate_data.rds")))

# Predicting finer scar at layer
LabelledSignalData <- LabelledSignalData  %>% mutate(depth_label = case_when(
  endocardium_scar == 0 & intramural_scar == 0 & epicardial_scar == 0 ~ "NoScar",
  endocardium_scar == 1 ~ "AtLeastEndo",
  endocardium_scar == 0 & intramural_scar == 1  ~ "AtLeastIntra",
  endocardium_scar == 0 & intramural_scar == 0 & epicardial_scar == 1 ~ "epiOnly",
  TRUE ~ "Otherwise"
))  %>% select(-c(endocardium_scar,intramural_scar,epicardial_scar))


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

get_mean <- function(list_containing_df) {
  data <- list_containing_df %>% unlist()
  result <- mean(data)
  return(result)
}

get_std <- function(list_containing_df) {
  data <- list_containing_df %>% unlist()
  result <- sd(data,na.rm = T)
  return(result)
}



count_crossings <- function(list_containing_df) {
  # number of times the signal crosses the x axis
  data <- list_containing_df %>% unlist()

  num_crossings <- 0
  prev_sign <- sign(data[1])
  for (i in 2:length(data)) {
    current_sign <- sign(data[i])
    if (current_sign != prev_sign) {
      num_crossings <- num_crossings + 1
      prev_sign <- current_sign
    }
  }
  return(num_crossings)
}

count_slope_changes <- function(list_containing_df) {
  # number of slope changes. i.e. goes from increasing to decreasing
  data <- list_containing_df %>% unlist()
  # Calculate the differences between consecutive numbers
  diffs <- diff(data)

  # Calculate the signs of the differences
  signs <- sign(diffs)

  # Count the number of times the sign changes
  num_changes <- sum(signs[-1] != signs[-length(signs)])

  return(num_changes)
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
  mutate(mean = get_mean(signal_unipolar),
         standard_deviation = sd(signal_unipolar %>% unlist(),na.rm = T),
         positivesum = sum_positive_values(signal_unipolar),
         positivemean = mean_positive_values(signal_unipolar),
         duration = length((signal_unipolar %>% unlist())),
         positivesumcheck = sapply(signal_unipolar, function(x) sum(x[x > 0])),
         positivemeancheck = sapply(signal_unipolar, function(x) mean(x[x > 0])),
         fourier_features = lapply(signal_unipolar,compute_fourier_transform),
         count_slope_changes = lapply(signal_unipolar,count_slope_changes) %>% unlist(),
         count_crossings = lapply(signal_unipolar,count_crossings) %>% unlist())


#extract circular mean and variance from fourier features to aggregate info into a feature.



LabelledSignalData <- LabelledSignalData %>% rowwise() %>%
  mutate(phase_mean = circular(fourier_features$magnitude) %>% mean(),
         phase_var = circular(fourier_features$phase) %>% var.circular(),
         magnitude_mean = mean(fourier_features$magnitude))


#prepare for a model.

#saveRDS(LabelledSignalData, file = here::here(generated_data_path,"model_data.rds"))

model_data <- LabelledSignalData

# doesnt make sense to bring in percentage of healthy numbers for imputed info - will be blank
if (data_type == "imputed"){
  model_data <- model_data %>% select(depth_label, # to determin labels
                                      mean,standard_deviation,positivesum,positivemean,duration, #aggregate features of signal
                                      phase_mean,phase_var,magnitude_mean, # aggregate features of fft
                                      count_slope_changes,count_crossings)
} else {
  model_data <- model_data %>% select(depth_label, # to determin labels
                                      mean,standard_deviation,positivesum,positivemean,duration, #aggregate features of signal
                                      phase_mean,phase_var,magnitude_mean, # aggregate features of fft
                                      count_slope_changes,count_crossings,
                                      healthy_perc_endo,healthy_perc_intra,healthy_perc_epi)
}



# Note positional data, sheep info are not be used as features.


#Saving Model data for Orange exploration
write_csv(model_data,here::here(generated_data_path,paste0("model_data",data_type,"unipolar.csv")))
saveRDS(model_data,file = here::here(generated_data_path,paste0("model_data",data_type,"unipolar.rds")))



