
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

#saveRDS(LabelledSignalData, file = here::here(generated_data_path,"model_data.rds"))

model_data <- LabelledSignalData

# Include certain features that should be used for prediction
model_data <- model_data %>% select(unipolar_voltage,bipolar_voltage,LAT, #signal settings
                                    Categorical_Label,endocardium_scar,intramural_scar, epicardial_scar, #labels will be excluded later
                                    mean,standard_deviation,sum,positivesum,positivemean,duration, #aggregate features of signal
                                    phase_mean,phase_var,magnitude_mean # aggregate features of fft
)

# Note positional data, sheep info are not be used as features.

# Predicting Scar or NoScar only at this stage and not depth.
model_data <- model_data %>% select(-c(endocardium_scar,intramural_scar,epicardial_scar)) %>%
  mutate(Categorical_Label = as.factor(Categorical_Label))

#Saving Model data for Orange exploration
write_csv(model_data,here::here(generated_data_path,paste0("model_data",data_type,".csv")))
saveRDS(model_data,file = here::here(generated_data_path,paste0("model_data",data_type,".rds")))


