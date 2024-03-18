library(here)
library(tidyverse)
library(readxl) # excel files
library(xml2) #xml files
library(plotly)
library(arrow)
source("paths.R")
get_paths()
# run_all("S15") #done again "S9" "S17" "S18" "S12" "S20" "S15"


# The run_all is made up of the following components:
# run retrieve_all_signals("S17") to aggregate the data for animal S18. Will Save rds file.
# run incorporate_histology("S17") to merge with histology and create NestedData rds file
# run write_as_long_format("S17") to save as a csv in long format (for possible python stuff)
# run write_as_parquet("S17") to save nested data as parquet file

post_introduction_perc_health("S15")

#sheep_name <- "S20"
#get_paths(sheep_name)

run_all <- function(sheep_name) {
  retrieve_all_signals(sheep_name)
  incorporate_histology(sheep_name)
  write_as_long_format(sheep_name)
  write_as_parquet(sheep_name)

}
load_sheep <- function(sheep_name) {
  get_sheep_path(sheep_name)
  # Aggregating Labelled Data
  LVpLabelledData <- read_excel(here::here(labelled_data_path,"cleaned_LVp Penta_car_labelled.xlsx")) %>%
    mutate(Categorical_Label = ifelse(Categorical_Label == -1,"Scar","NoScar"),
                                      Catheter_Type = "Penta") %>%
    select(c(-1)) %>% mutate(WaveFront = "LVp")


  RVpLabelledData <- read_excel(here::here(labelled_data_path,"cleaned_RVp Penta_car_labelled.xlsx"))  %>%
    mutate(Categorical_Label = ifelse(Categorical_Label == -1,"Scar","NoScar"),
           Catheter_Type = "Penta") %>%
    select(c(-1)) %>% mutate(WaveFront = "RVp")

  SRLabelledData <- read_excel(here::here(labelled_data_path,"cleaned_SR Penta_car_labelled.xlsx")) %>%
    mutate(Categorical_Label = ifelse(Categorical_Label == -1,"Scar","NoScar"),
           Catheter_Type = "Penta") %>%
    select(c(-1)) %>% mutate(WaveFront = "SR")

  LabelledData <- bind_rows(LVpLabelledData,RVpLabelledData,SRLabelledData)
  LabelledData <- LabelledData %>% mutate(sheep = sheep_name)

  rm(LVpLabelledData,RVpLabelledData,SRLabelledData)
  return(LabelledData)
}

find_from_woi <- function(WaveFront, Catheter_Type, Point_Number) {
  file_pattern <- paste0(".*", WaveFront, "\\s", Catheter_Type, "_P", Point_Number, "_Point_Export\\.xml$")
  matching_file <- list.files(path = main_data_path,
                              pattern = file_pattern, full.names = TRUE)
  #expect one file
  if (length(matching_file) == 1 ) {
    xml_data <- read_xml(matching_file)
    # Define XPath expressions to extract key elements from xml file
    reference_annotation <- xml_find_all(xml_data, "//Annotations/@Reference_Annotation") %>%
      xml_text() %>% as.numeric()
    woi_from <- xml_find_all(xml_data, "//WOI/@From") %>%
      xml_text() %>% as.numeric()
    woi_to <- xml_find_all(xml_data, "//WOI/@To") %>%
      xml_text() %>% as.numeric()
    return(reference_annotation + woi_from)
  } else {
    return() # returns NULL and can use is.null() to check
  }
}

find_to_woi <- function(WaveFront, Catheter_Type, Point_Number) {
  file_pattern <- paste0(".*", WaveFront, "\\s", Catheter_Type, "_P", Point_Number, "_Point_Export\\.xml$")
  matching_file <- list.files(path = main_data_path,
                              pattern = file_pattern, full.names = TRUE)
  #expect one file
  if (length(matching_file) == 1 ) {
    xml_data <- read_xml(matching_file)
    # Define XPath expressions to extract key elements from xml file
    reference_annotation <- xml_find_all(xml_data, "//Annotations/@Reference_Annotation") %>%
      xml_text() %>% as.numeric()
    woi_from <- xml_find_all(xml_data, "//WOI/@From") %>%
      xml_text() %>% as.numeric()
    woi_to <- xml_find_all(xml_data, "//WOI/@To") %>%
      xml_text() %>% as.numeric()
    return(reference_annotation + woi_to)
  } else {
    return() # returns NULL and can use is.null() to check
  }
}

find_window <- function(WaveFront, Catheter_Type, Point_Number) {
  # Extracts key elements from the Point Export Files.
  # Find the window of interest in the signal as specified by Point Export files.
  # Example: find_window("SR","Penta",4815) and find_window("RVp","Penta",4815) has files and windows of interest
  # find_window("LVp","Penta",4815) does not exist and NULL returned

  file_pattern <- paste0(".*", WaveFront, "\\s", Catheter_Type, "_P", Point_Number, "_Point_Export\\.xml$")

  matching_file <- list.files(path = main_data_path,
                               pattern = file_pattern, full.names = TRUE)
  #expect one file
  if (length(matching_file) == 1 ) {
    xml_data <- read_xml(matching_file)
    # Define XPath expressions to extract key elements from xml file
    reference_annotation <- xml_find_all(xml_data, "//Annotations/@Reference_Annotation") %>%
      xml_text() %>% as.numeric()
    woi_from <- xml_find_all(xml_data, "//WOI/@From") %>%
      xml_text() %>% as.numeric()
    woi_to <- xml_find_all(xml_data, "//WOI/@To") %>%
      xml_text() %>% as.numeric()

    woi <- list(From = reference_annotation + woi_from, To = reference_annotation + woi_to)
    return(woi)
  } else {
    return() # returns NULL and can use is.null() to check
  }

}

find_signal_file <- function(WaveFront, Catheter_Type, Point_Number) {
  #Given you have a woi, get signal information
  pattern <- sprintf(".*%s %s_P%s_ECG_Export\\.txt$", WaveFront, Catheter_Type, Point_Number)

  print(pattern)
  print(main_data_path)
  matching_file <- list.files(path = main_data_path,
                              pattern = pattern, full.names = TRUE)


  return(matching_file)

}


get_signal_data <- function(WaveFront, Catheter_Type, Point_Number) {
  # read the signal information
  # Assumes rows where the data starts is constant.

  woi <- find_window(WaveFront,Catheter_Type,Point_Number)
  if (is.null(woi)) {
    return() #return null - could find signal info
  }
  txt_file <- find_signal_file(WaveFront,Catheter_Type,Point_Number)
  tabular_content <-  read_table(txt_file, skip = 3)
  raw_ecg_gain <- str_extract(read_lines(txt_file,n_max = 4), "^Raw ECG to MV \\(gain\\) = ([0-9.]+)$") %>%
    parse_number()
  raw_ecg_gain <- raw_ecg_gain[!is.na(raw_ecg_gain)]

  #double check gain and channels are extracted from txt file

  if (!is.numeric(raw_ecg_gain)) {
    return() #return null
  }

  channel <- str_match(read_lines(txt_file,n_max = 4), "Bipolar Mapping Channel=(\\w+-\\w+)")[3 , 2]
  #find appropriate column to extract bipolar recordings given channel and window of interest

  if (is.null(channel)) {
    return() #return null
  }

  signal <- tabular_content %>% select(matches(channel)) %>%
            slice(.,woi$From:woi$To ) * raw_ecg_gain

  #standardising column name irrespective of channel name
  signal <- signal %>% rename(.,"signal_data" = names(signal))
  return(signal)
}


get_raw_signal_data <- function(WaveFront, Catheter_Type, Point_Number) {

  woi <- find_window(WaveFront,Catheter_Type,Point_Number)
  if (is.null(woi)) {
    return() #return null - could find signal info
  }
  txt_file <- find_signal_file(WaveFront,Catheter_Type,Point_Number)
  tabular_content <-  read_table(txt_file, skip = 3)
  raw_ecg_gain <- str_extract(read_lines(txt_file,n_max = 4), "^Raw ECG to MV \\(gain\\) = ([0-9.]+)$") %>%
    parse_number()
  raw_ecg_gain <- raw_ecg_gain[!is.na(raw_ecg_gain)]

  #double check gain and channels are extracted from txt file

  if (!is.numeric(raw_ecg_gain)) {
    return() #return null
  }

  channel <- str_match(read_lines(txt_file,n_max = 4), "Bipolar Mapping Channel=(\\w+-\\w+)")[3 , 2]
  #find appropriate column to extract bipolar recordings given channel and window of interest

  if (is.null(channel)) {
    return() #return null
  }

  signal <- tabular_content %>% select(matches(channel)) * raw_ecg_gain

  #standardising column name irrespective of channel name
  raw_signal <- signal %>% rename(.,"signal_data" = names(signal))
  return(raw_signal)
}



retrieve_all_signals <- function(sheep_name) {
  LabelledSignalData <- load_sheep(sheep_name)

  LabelledSignalData <- LabelledSignalData %>% rowwise() %>%
   mutate(signal = list(get_signal_data(WaveFront, Catheter_Type, Point_Number)),
           rawsignal = list(get_raw_signal_data(WaveFront, Catheter_Type, Point_Number)))


  # bring extra woi data only if there is a signal
  no_signals <- LabelledSignalData %>% filter(is.null(signal)) %>% mutate(From = 0, To = 0)

  with_signals <- LabelledSignalData %>% filter(!is.null(signal)) %>% rowwise() %>% mutate(From = list(find_from_woi(WaveFront, Catheter_Type, Point_Number)) %>% unlist() %>% as.integer(),
                                              To = list(find_to_woi(WaveFront, Catheter_Type, Point_Number)) %>% unlist() %>% as.integer())

  LabelledSignalData <- bind_rows(with_signals,no_signals)

  #format and sort appropriately
  LabelledSignalData <- LabelledSignalData %>%
    mutate(across(where(is.character), as.factor))

  LabelledSignalData <- LabelledSignalData %>% arrange(sheep,Catheter_Type,WaveFront,Point_Number)

  saveRDS(LabelledSignalData,file = here::here(generated_data_path,paste0("LabelledSignalData",sheep_name,".rds")))
  }

load_histology <- function() {
  histology_labels <- read_csv(file = here::here("data","cleaned_histology_all.csv")) %>%
    select(Animal,Specimen_ID,Endo3_anyscar,IM3_anyscar,Epi3_anyscar,Endo3__VM, IM3_VM,Epi3_VM) %>%
    rename(Histology_Biopsy_Label = Specimen_ID, endocardium_scar = Endo3_anyscar,intramural_scar = IM3_anyscar, epicardial_scar = Epi3_anyscar,
           healthy_perc_endo = Endo3__VM, healthy_perc_intra = IM3_VM,healthy_perc_epi = Epi3_VM) %>%
    na.omit()
  return(histology_labels)

}

incorporate_histology <- function(sheep_name) {
  #histology information regarding if scar is present at different depths
  LabelledSignalData <- readRDS(file =here::here(generated_data_path,paste0("LabelledSignalData",sheep_name,".rds")))
  histology <- load_histology()
  result <- left_join(LabelledSignalData, histology, by = "Histology_Biopsy_Label")
  saveRDS(result,here::here(generated_data_path,paste0("NestedData",sheep_name,".rds")))

}

post_introduction_perc_health <- function(sheep_name){
#post hack to introduce percentage health data - not entire workflow so to avoid signal aggregation
  incorporate_histology(sheep_name)
  write_as_long_format(sheep_name)
  write_as_parquet(sheep_name)

}
write_as_long_format <- function(sheep_name){
  NestedData <- readRDS(file = here::here(generated_data_path,paste0("NestedData",sheep_name,".rds")))
  LongData <- NestedData %>% unnest(signal)
  write_csv(LongData, file = here::here(generated_data_path,paste0("NestedData",sheep_name,".csv")))
}

write_as_parquet <- function(sheep_name){
  NestedData <- readRDS(file = here::here(generated_data_path,paste0("NestedData",sheep_name,".rds")))
  parquet_file <- here::here(generated_data_path,paste0("NestedData",sheep_name,".parquet"))
  write_parquet(NestedData, parquet_file)

}



