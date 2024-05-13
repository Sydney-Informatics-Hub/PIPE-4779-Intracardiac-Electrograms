

library(here)
library(tidyverse)
library(stringr)
library(skimr)
library(plotly)
library(RColorBrewer)
library(arrow)
source(here::here("r_scripts","paths.R"))
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



# Publishable data without any assumption on features..... filtered for labels only

data_type <- "filtered"
model_data <- readRDS(file = here::here(generated_data_path,paste0(data_type,"_aggregate_data.rds")))

# Predicting finer scar at layer
model_data <- model_data  %>% mutate(depth_label = case_when(
  endocardium_scar == 0 & intramural_scar == 0 & epicardial_scar == 0 ~ "NoScar",
  endocardium_scar == 1 ~ "AtLeastEndo",
  endocardium_scar == 0 & intramural_scar == 1  ~ "AtLeastIntra",
  endocardium_scar == 0 & intramural_scar == 0 & epicardial_scar == 1 ~ "epiOnly",
  TRUE ~ "Otherwise"
))  %>% select(-c(Animal,`unipolar voltage`,`bipolar voltage`,AnyScar))


saveRDS(model_data,here::here(generated_data_path,paste0("publishable_model_data",data_type,".rds")))

parquet_file <- here::here(generated_data_path,paste0("publishable_model_data.parquet"))
write_parquet(model_data, parquet_file)

# for completing the loop on publishable data to the model we are
# renaming signal_data which is the specific signal tsai modelling uses
model_data <- model_data %>% rename(signal_data = raw_unipolar)
parquet_file <- here::here(generated_data_path,paste0("publishable_model_data_TSAI.parquet"))
write_parquet(model_data, parquet_file)



# Publishable data without any assumption on features.....
# imputed represents all co-ordinates and assumes signals that were not labelled was
# no scar

data_type <- "imputed"
model_data <- readRDS(file = here::here(generated_data_path,paste0(data_type,"_aggregate_data.rds")))

# Predicting finer scar at layer
model_data <- model_data  %>% mutate(depth_label = case_when(
  endocardium_scar == 0 & intramural_scar == 0 & epicardial_scar == 0 ~ "NoScar",
  endocardium_scar == 1 ~ "AtLeastEndo",
  endocardium_scar == 0 & intramural_scar == 1  ~ "AtLeastIntra",
  endocardium_scar == 0 & intramural_scar == 0 & epicardial_scar == 1 ~ "epiOnly",
  TRUE ~ "Otherwise"
))  %>% select(-c(Animal,`unipolar voltage`,`bipolar voltage`,AnyScar))


saveRDS(model_data,here::here(generated_data_path,paste0("publishable_model_data",data_type,".rds")))

# for completing the loop on publishable data to the model we are
# renaming signal_data which is the specific signal tsai modelling uses
model_data <- model_data %>% rename(signal_data = raw_unipolar)
parquet_file <- here::here(generated_data_path,paste0("publishable_model_data_TSAI",data_type,".parquet"))
write_parquet(model_data, parquet_file)


