# summary of labels and sheep - is 10%
library(here)
library(arrow)
library(tidyverse)
source(here::here("r_scripts","paths.R"))
get_paths()
path_models <- here::here("python","models")
deploy_data_path <- here::here("deploy","data")

parquet_file <- here::here(generated_data_path,paste0("publishable_model_data_TSAI","imputed",".parquet"))
all_truth <- read_parquet(parquet_file) %>% select(sheep,Point_Number,WaveFront,X,Y,Z,depth_label,signal_data)

# total observations by sheep
all_truth %>% group_by(sheep) %>% summarise(count = n())

#totals by sheep, wavefront and label
totals <- all_truth %>% group_by(sheep,WaveFront,depth_label) %>% summarise(count = n())

write_csv(totals,file = here::here(deploy_data_path,"totals.csv"))
