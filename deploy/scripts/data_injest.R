# run data_injest() function in Rstudio
# or Rscript data_injest.R from command line

# Will save the parquet file in /deploy/data/data_injest.parquet


library(here)
library(tidyverse)
library(arrow)
deploy_data_path <- here::here("deploy","data")


data_injest <- function() {
  data <- collect_data("Penta")
}
collect_data <- function(Catheter_Type) {
  # Catheter_Type <- "Penta"
  template <- get_template(Catheter_Type)

  template <- template %>% rowwise() %>%
    mutate(signal_data = list(get_raw_signal_unipolar_data(WaveFront, Catheter_Type, Point_Number)))
  template <- template %>% select(paths, everything()) %>% mutate(Point_Number = as.integer(Point_Number))
  print("finished reading signals...")

  wavefronts_collected <- template %>% distinct(WaveFront) #usually on LVp RVp and SR but not always

  # for each wavefront collect and rowbind geometry
  all_geometries <- map_df(wavefronts_collected$WaveFront, ~ get_geometry(.x,Catheter_Type))

  #merge geometry info with signal data
  signals <- right_join(x = all_geometries,y = template,by = join_by(Catheter_Type,WaveFront,Point_Number))
  parquet_file <- here::here(deploy_data_path,paste0("data_injest.parquet"))
  write_parquet(signals, parquet_file)
  print("finished combing geometries...")
  #df <- arrow::read_parquet(here::here(deploy_data_path,paste0("data_injest.parquet")))
  return(template)

}

# produces dataframe with WaveFront, Catheter_Type, Point_Number combinations given the
# signals that need to be injested based on Export analysis folder in deploy_data_path
get_template <- function(Catheter_Type) {
  df <- tibble(files = list.files(path = here::here("deploy","data","Export_Analysis"),
             pattern = "_ECG_Export.txt",
             full.names = FALSE)) %>% mutate(paths = files)

  df$Point_Number <- gsub(".*P(\\d+).*", "\\1", df$files) #%>% as.integer() #look for digit after P


  df <- df %>% separate(files, sep = " ",into = c("A","WaveFront")) %>%
    select(-A) %>%
    mutate(Catheter_Type = Catheter_Type)

  return(df)

}

# read the signal information
# Assumes rows where the data starts is constant.
get_raw_signal_unipolar_data <- function(WaveFront, Catheter_Type, Point_Number) {
  txt_file <- find_signal_file(WaveFront,Catheter_Type,Point_Number)
  tabular_content <-  read_table(txt_file, skip = 3)
  raw_ecg_gain <- str_extract(read_lines(txt_file,n_max = 4), "^Raw ECG to MV \\(gain\\) = ([0-9.]+)$") %>%
    parse_number()
  raw_ecg_gain <- raw_ecg_gain[!is.na(raw_ecg_gain)]

  #double check gain and channels are extracted from txt file

  if (!is.numeric(raw_ecg_gain)) {
    return() #return null
  }

  lines <- read_lines(txt_file, n_max = 4)
  pattern <- "Unipolar Mapping Channel=\\w+_\\w+"
  channel <- str_extract(lines, pattern)
  channel <- str_match(channel[3], "Unipolar Mapping Channel=(\\w+_\\w+)")[2]


  if (is.null(channel)) {
    return() #return null
  }

  signal <- tabular_content %>% select(contains(channel)) %>% select(-contains("-")) %>% select(1) * raw_ecg_gain

  #standardising column name irrespective of channel name
  signal <- signal %>% rename(.,"signal_data" = names(signal))
  return(signal)
}


find_signal_file <- function(WaveFront, Catheter_Type, Point_Number) {
  pattern <- sprintf(".*%s %s_P%s_ECG_Export\\.txt$", WaveFront, Catheter_Type, Point_Number)
  matching_file <- list.files(path = here::here(deploy_data_path,"Export_Analysis"),
                              pattern = pattern, full.names = TRUE)

  return(matching_file)

}


get_geometry <- function(WaveFront,Catheter_Type) {

  pattern <- sprintf(".*%s %s_car\\.txt$", WaveFront, Catheter_Type)
  matching_file <- list.files(path = here::here(deploy_data_path,"Export_Analysis"),
                              pattern = pattern, full.names = TRUE)

  tabular_content <- read_table(matching_file, skip = 1,col_names = FALSE) %>%
    select(X3,X5,X6,X7) %>% rename(Point_Number = X3, X = X5 , Y = X6, Z = X7) %>%
    mutate(WaveFront = WaveFront, Catheter_Type = Catheter_Type)

}

data_injest()
