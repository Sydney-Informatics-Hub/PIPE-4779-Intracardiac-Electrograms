library(here)
library(arrow)
library(tidyverse)

deploy_data_path <- here::here("deploy","data")

df_python <- read_parquet(here(deploy_data_path,"preprocessed_rawsignal_unipolar_penta.parquet")) %>%
  arrange(WaveFront,Point_Number) %>% mutate(WaveFront = as_factor(WaveFront)) %>% as_tibble()

high_level <- df_python %>% unnest(signal) %>% group_by(WaveFront) %>% summarise(total = sum(signal))

df_r <- read_parquet(here(deploy_data_path,"data_injest_fromR.parquet")) %>%
        mutate(WaveFront = as_factor(WaveFront)) %>% as_tibble() %>%
        arrange(WaveFront,Point_Number)



high_level2 <-df_r %>% unnest(signal_data) %>% group_by(WaveFront) %>% summarise(total = sum(signal_data))

high_level == high_level2 #true


df_r_long <- df_r %>% unnest(signal_data)

df_python_long <- df_python %>% rename(signal_data = signal ) %>% unnest(signal_data) %>%
  rename(paths = files)

result <- all.equal(df_python_long, df_r_long) #TRUE

print(result == TRUE)#TRUE


