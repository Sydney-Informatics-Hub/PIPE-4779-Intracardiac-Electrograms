library(here)
library(tidyverse)
library(readxl) # excel files
library(xml2) #xml files
library(plotly)

# run retrieve_all_signals() to aggregate the data for animal S18
# checks are done to make sure the extracted signal aligns with the data dictionary given


labelled_data_path <- here::here("data","S18","labelled")

main_data_path <- here::here("data","S18","Export_Analysis-01_05_2024-17-02-34")

# Aggregating Labelled Data
LVpLabelledData <- read_excel(here::here(labelled_data_path,"cleaned_9-1-1-ReLV LVp Penta_car_labelled.xlsx")) %>%
  mutate(Categorical_Label = ifelse(Categorical_Label == -1,"Scar","NoScar"),
                                    Catheter_Type = "Penta") %>%
  select(c(-1)) %>% mutate(WaveFront = "LVp")


RVpLabelledData <- read_excel(here::here(labelled_data_path,"cleaned_9-1-ReLV RVp Penta_car labelled.xlsx"))  %>%
  mutate(Categorical_Label = ifelse(Categorical_Label == -1,"Scar","NoScar"),
         Catheter_Type = "Penta") %>%
  select(c(-1)) %>% mutate(WaveFront = "RVp")

SRLabelledData <- read_excel(here::here(labelled_data_path,"cleaned_9-LV SR Penta_car_labelled.xlsx")) %>%
  mutate(Categorical_Label = ifelse(Categorical_Label == -1,"Scar","NoScar"),
         Catheter_Type = "Penta") %>%
  select(c(-1)) %>% mutate(WaveFront = "SR")

LabelledData <- bind_rows(LVpLabelledData,RVpLabelledData,SRLabelledData)
rm(LVpLabelledData,RVpLabelledData,SRLabelledData)


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
            slice(.,(woi$From + 5):(woi$To +5) ) * raw_ecg_gain

  return(signal)
}

retrieve_all_signals <- function() {
  LabelledSignalData <- LabelledData %>% rowwise()  %>% mutate(signal = list(get_signal_data(WaveFront, Catheter_Type, Point_Number)))
  saveRDS(LabelledSignalData,file = here::here(labelled_data_path,"LabelledSignalData.rds"))
  }

# this confirms the graph in the data dictionary
data <- get_signal_data("SR","Penta",4815) %>% as_tibble()
plotme <- plot_ly(data, y = ~`20A_17-18(129)`, type = 'scatter', mode = 'lines')
plotme

#Double check this is the same thing in the aggregate data.
LabelledSignalData <- readRDS(file = here::here(labelled_data_path,"LabelledSignalData.rds"))

LabelledSignalData %>%
  filter(WaveFront == "SR",Catheter_Type == "Penta", Point_Number == 4815) %>%
  select(signal) %>%
  unlist() %>%
  plot(type = "l")



