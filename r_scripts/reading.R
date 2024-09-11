
library(here)
library(tidyverse)
library(arrow)

library(here)
path <- here::here("deploy","output")
data <- read_parquet(here::here(path,"preprocessed_rawsignal_unipolar_penta.parquet"))

data <- data |> select(Point_Number,X,Y,Z,Catheter_Type,WaveFront,signal)
data2 <- data %>% unnest(signal)

write_csv(data2,file = "output.csv")



