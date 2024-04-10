

library(here)
library(tidyverse)
library(stringr)
library(skimr)
library(plotly)
library(RColorBrewer)
source("paths.R")
get_paths()

aggregate_data <- list.files(path = generated_data_path,
                             pattern = "NestedDataS[0-9]+\\.rds",
                             full.names = TRUE) %>% map(readRDS) %>% bind_rows()


imputed_aggregate_data <- aggregate_data %>% filter(!is.null(signal)) %>%
  mutate(endocardium_scar = ifelse(is.na(endocardium_scar),0,endocardium_scar),
         intramural_scar = ifelse(is.na(intramural_scar),0,intramural_scar),
         epicardial_scar = ifelse(is.na(epicardial_scar),0,epicardial_scar))


aggregate_data <- aggregate_data %>% mutate(AnyScar = sum(endocardium_scar + intramural_scar + epicardial_scar)) %>%
  mutate(Categorical_Label = case_when(
    is.na(AnyScar) ~ NA_character_,
    AnyScar >= 1 ~ "Scar",
    AnyScar == 0 ~ "NoScar"
  ))

imputed_aggregate_data <- imputed_aggregate_data %>% mutate(AnyScar = sum(endocardium_scar + intramural_scar + epicardial_scar)) %>%
  mutate(Categorical_Label = case_when(
    is.na(AnyScar) ~ NA_character_,
    AnyScar >= 1 ~ "Scar",
    AnyScar == 0 ~ "NoScar"
  ))


cleaned_aggregate_data <- aggregate_data %>% filter(!is.null(signal)) %>%
  filter(!is.na(endocardium_scar))

#

saveRDS(cleaned_aggregate_data,file = here::here(generated_data_path,"filtered_aggregate_data.rds"))

saveRDS(imputed_aggregate_data,file = here::here(generated_data_path,"imputed_aggregate_data.rds"))

saveRDS(aggregate_data,file = here::here(generated_data_path,"aggregate_data.rds"))

cleaned_aggregate_long_data <- cleaned_aggregate_data %>% select(-rawsignal, -From,-To,-Animal) %>% unnest(signal)
cleaned_aggregate_long_data <- cleaned_aggregate_long_data %>% group_by(sheep,Point_Number,WaveFront, Catheter_Type) %>% mutate(Row = row_number()) %>% ungroup()

saveRDS(cleaned_aggregate_long_data,file = here::here(generated_data_path,"cleaned_aggregate_long_data.rds"))


#writing long format to excel
data <- readRDS(file = here::here(generated_data_path,"filtered_aggregate_data.rds"))
data <- data %>% select(-c(Animal,`bipolar voltage`,`unipolar voltage`,From,To))

signal_woi_bipolar_long <- data %>% unnest(signal) %>% select(-c(rawsignal, signal_unipolar))
signal_raw_bipolar_long <- data %>% unnest(rawsignal) %>% select(-c(signal, signal_unipolar))
signal_woi_unipolar_long <- data %>% unnest(signal_unipolar) %>% select(-c(signal, rawsignal))

write_csv(signal_woi_bipolar_long,here::here(generated_data_path,"signal_woi_bipolar_long.csv"))
write_csv(signal_raw_bipolar_long,here::here(generated_data_path,"signal_raw_bipolar_long.csv"))
write_csv(signal_woi_unipolar_long,here::here(generated_data_path,"signal_woi_unipolar_long.csv"))

