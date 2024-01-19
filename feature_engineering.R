library(here)
library(tidyverse)
library(readxl) # excel files
library(xml2) #xml files

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
  #Find the window of interest in the signal as specified by Point Export files.
  # For example: 9-LV SR Penta_P4815_Point_Export.xml
  # find_window("SR","Penta",4815) and find_window("RVp","Penta",4815) has files
  # find_window("LVp","Penta",4815) does not exist and NULL returned

  file_pattern <- paste0(".*", WaveFront, "\\s", Catheter_Type, "_P", Point_Number, "_Point_Export\\.xml$")
  matching_files <- list.files(path = main_data_path,
                               pattern = file_pattern, full.names = TRUE)

  if (length(matching_files) > 0 ) {
    return(matching_files)
  } else {
    return() # returns NULL and can use is.null() to check
  }

}

# to be continued..
lookslike <- read_xml(find_window("SR","Penta",4815))




