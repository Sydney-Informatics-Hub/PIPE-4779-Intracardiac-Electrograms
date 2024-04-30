# collection of writing data scripts not entirely neccessary for model pipeline
# but for checking and on request.
library(here)
library(tidyverse)
library(arrow)
source("paths.R")
get_paths()

#output of combine_data.py versus output of post_aggregation
df <- read_parquet(here::here("results","NestedDataAll_rawsignal_clean.parquet"))
df2 <- read_parquet(here::here("data","generated","publishable_model_data_TSAI.parquet"))

data_type <- "filtered"


#check stuff
sheep_name <- "S18"
NestedData <- readRDS(file = here::here(generated_data_path,paste0("NestedData",sheep_name,".rds")))
NestedData_csv <- read_csv(file = here::here(generated_data_path,paste0("NestedData",sheep_name,".csv")))
NestedData_csv <- read_csv(file = here::here(generated_data_path,paste0("NestedData",sheep_name,".csv")))
d1 <- read_csv(file = here::here(generated_data_path,"signal_woi_unipolar_long.csv"))
d2 <- read_csv(file = here::here(generated_data_path,"signal_raw_bipolar_long.csv"))
d3 <- read_csv(file = here::here(generated_data_path,"signal_woi_bipolar_long.csv"))

data_tsfresh <- read_csv(file = here::here(generated_data_path,paste0("ts_features","_",data_type,".csv"))) %>%
  select(depth_label, everything())


write_csv(data_tsfresh, file = here::here(generated_data_path,paste0("ts_all_features","_",data_type,"_and_label.csv")))


# Publishable data without any assumption on features.

data_type <- "filtered"
#data_type <- "imputed"

model_data <- readRDS(file = here::here(generated_data_path,paste0(data_type,"_aggregate_data.rds")))

# Predicting finer scar at layer
model_data<- model_data  %>% mutate(depth_label = case_when(
  endocardium_scar == 0 & intramural_scar == 0 & epicardial_scar == 0 ~ "NoScar",
  endocardium_scar == 1 ~ "AtLeastEndo",
  endocardium_scar == 0 & intramural_scar == 1  ~ "AtLeastIntra",
  endocardium_scar == 0 & intramural_scar == 0 & epicardial_scar == 1 ~ "epiOnly",
  TRUE ~ "Otherwise"
))  %>% select(-c(Animal,`unipolar voltage`,`bipolar voltage`,AnyScar))


saveRDS(model_data,here::here(generated_data_path,paste0("publishable_model_data",data_type,".rds")))


#writing long format to excel
data <- readRDS(file = here::here(generated_data_path,"filtered_aggregate_data.rds"))
data <- data %>% select(-c(Animal,`bipolar voltage`,`unipolar voltage`,From,To))

signal_woi_bipolar_long <- data %>% unnest(signal) %>% select(-c(rawsignal, signal_unipolar))
signal_raw_bipolar_long <- data %>% unnest(rawsignal) %>% select(-c(signal, signal_unipolar))
signal_woi_unipolar_long <- data %>% unnest(signal_unipolar) %>% select(-c(signal, rawsignal))

write_csv(signal_woi_bipolar_long,here::here(generated_data_path,"signal_woi_bipolar_long.csv"))
write_csv(signal_raw_bipolar_long,here::here(generated_data_path,"signal_raw_bipolar_long.csv"))
write_csv(signal_woi_unipolar_long,here::here(generated_data_path,"signal_woi_unipolar_long.csv"))

