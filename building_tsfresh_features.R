
library(reticulate)
library(tidyverse)
library(tsibble)
library(feasts)
library(tsibbledata)
library(reticulate)
library(tsfeatures)
use_python("/Users/kris/miniconda3/bin/python")
import("tsfresh")
source("paths.R")
get_paths()



#data_type <- "filtered"
data_type <- "imputed"

LabelledSignalData <- readRDS(file = here::here(generated_data_path,paste0(data_type,"_aggregate_data.rds")))



long_data <- LabelledSignalData %>% select(Point_Number,Catheter_Type,WaveFront,signal,sheep) %>% unnest(signal)

long_data <- long_data %>% mutate(id = paste0("ID",sheep,Point_Number,WaveFront,Catheter_Type)) %>%
  select(id,signal_data) %>% group_by(id) %>% mutate(Row = row_number()) %>% ungroup()


tsb <- as_tsibble(long_data, key = id, index = Row)

# takes long time to run.
data_features <- tsb %>% features(features = tsfresh_features, .var = signal_data)

# to run again and join assuming works





