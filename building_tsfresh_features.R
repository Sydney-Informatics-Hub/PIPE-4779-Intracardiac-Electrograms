library(renv)
renv::init()
library(reticulate)
library(tidyverse)
library(tsibble)
library(feasts)
library(tsibbledata)
library(reticulate)
library(tsfeatures)
library(feasts.tsfresh) # for tsfresh_features
renv::snapshot() #handle reticlate seperately

use_python("/Users/kris/miniconda3/bin/python")
import("tsfresh")
source("paths.R")
get_paths()



data_type <- "filtered"
data_type <- "imputed"

LabelledSignalData <- readRDS(file = here::here(generated_data_path,paste0(data_type,"_aggregate_data.rds"))) %>%
  mutate(Point_Number = as.factor(Point_Number))

labels <- LabelledSignalData %>% select(Point_Number,Catheter_Type,WaveFront,signal,sheep,endocardium_scar,intramural_scar,epicardial_scar)
#testing
LabelledSignalData <- LabelledSignalData

long_data <- LabelledSignalData %>% select(Point_Number,Catheter_Type,WaveFront,signal,sheep) %>% unnest(signal)

long_data <- long_data %>% mutate(id = paste0("ID","_",sheep,"_",Point_Number,"_",WaveFront,"_",Catheter_Type)) %>%
  select(id,signal_data) %>% group_by(id) %>% mutate(Row = row_number()) %>% ungroup()

tsb <- as_tsibble(long_data, key = id, index = Row)

# takes long time to run.
# run time of 14 sec for 10 point observations 1.4 sec per observation

data_features <- tsb %>% features(features = tsfresh_features, .var = signal_data)

data_features  <- data_features %>%
  separate(id, into = c("ID", "sheep", "Point_Number","WaveFront","Catheter_Type"), sep = "_", remove = FALSE) %>%
  select(-ID) %>% mutate(Point_Number = as.factor(Point_Number))


data_features <- data_features %>% left_join(.,labels,by = c("sheep", "Point_Number","WaveFront","Catheter_Type"))

saveRDS(data_features,file = here::here(generated_data_path,paste0("ts_features","_",data_type,".rds")))


# - Post run
data_features <- readRDS(file = here::here(generated_data_path,paste0("ts_features","_",data_type,".rds")))


#labels reflect how deep removal of scar tissue is likely to be
# aligns category with the action during procedure.
data_features <- data_features  %>% mutate(depth_label = case_when(
  endocardium_scar == 0 & intramural_scar == 0 & epicardial_scar == 0 ~ "NoScar",
  endocardium_scar == 1 ~ "AtLeastEndo",
  endocardium_scar == 0 & intramural_scar == 1  ~ "AtLeastIntra",
  endocardium_scar == 0 & intramural_scar == 0 & epicardial_scar == 1 ~ "epiOnly",
  TRUE ~ "Otherwise"
)) %>% select(-signal) %>% select(-c(endocardium_scar,intramural_scar,epicardial_scar))



write_csv(data_features, file = here::here(generated_data_path,paste0("ts_features","_",data_type,".csv")))

# run time of 14 sec for 10 point observations 1.4 sec per observation

# to run again and join assuming works





