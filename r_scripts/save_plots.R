# Collection of files to save charts to pdf

library(here)
library(tidyverse)
library(stringr)
library(skimr)
library(plotly)
library(RColorBrewer)
#just for saving plots
library(reticulate)
use_python("/Users/kris/miniconda3/bin/python")
import("kaleido")
import("plotly")
source("paths.R")
get_paths()

aggregate_data <- list.files(path = generated_data_path,
                             pattern = "NestedDataS[0-9]+\\.rds",
                             full.names = TRUE) %>% map(readRDS) %>% bind_rows()


cleaned_aggregate_data <- readRDS(file = here::here(generated_data_path,"filtered_aggregate_data.rds"))
cleaned_aggregate_long_data <- readRDS(file = here::here(generated_data_path,"cleaned_aggregate_long_data.rds"))




save_plot <- function(name,p){
  # Save the plot to a PDF file
  path <- here::here(file = here::here(generated_data_path,name))
  save_image(p, file = path)
  Sys.sleep(5)  # sleep to make sure it renders to file
}


get_individual_plot <- function(plot_name,select_sheep,select_wavefront ) {
  colors <- brewer.pal(n = 4, name = "Set1")
  p <- cleaned_aggregate_long_data %>% select(Categorical_Label,signal_data,Row,sheep,WaveFront, Catheter_Type, Point_Number) %>%
    filter(sheep == select_sheep,WaveFront == select_wavefront) %>%
    group_by(sheep,WaveFront, Catheter_Type, Point_Number) %>%
    plot_ly(., x = ~Row,y = ~signal_data, type = 'scatter', mode = 'lines',
            color = ~Categorical_Label, line = list(width = 0.4),
            colors = colors)  %>%
    layout(
      xaxis = list(title = "Row"),
      yaxis = list(title = "signal"),
      title = paste0(select_sheep," ",select_wavefront," signals"))

  save_plot(plot_name,p)

  return(p)

}

#"S9" "S17" "S18" "S12" "S20" "S15"
#"SR" ("Ap" for S9) "LVp" "RVp"
#look at individual case
select_sheep <- "S15"
select_wavefront <- "LVp"
p <- get_individual_plot(paste0(select_sheep,"_",select_wavefront,".pdf"),select_sheep,select_wavefront )


# run all

all_wavefronts <- c("SR","LVp","RVp")
all_sheep <- c("S17","S18","S12","S20","S15")
#saves them all as pdf
for (s in all_sheep) {
  for (w in all_wavefronts){
    p <- get_individual_plot(paste0(s,"_",w,".pdf"),s,w )

  }
}
all_sheep <- c("S9")
all_wavefronts <- c("Ap","LVp","RVp")
#saves them all as pdf
for (s in all_sheep) {
  for (w in all_wavefronts){
    p <- get_individual_plot(paste0(s,"_",w,".pdf"),s,w )

  }
}

#All Data - Scar versus No Scar

# Define a color palette
colors <- brewer.pal(n = 2, name = "Set1")

p <- cleaned_aggregate_long_data %>% select(Categorical_Label,signal_data,Row,sheep,WaveFront, Catheter_Type, Point_Number) %>%
  group_by(sheep,WaveFront, Catheter_Type, Point_Number) %>%
  plot_ly(., x = ~Row,y = ~signal_data, type = 'scatter', mode = 'lines',
          color = ~Categorical_Label, colors = colors, line = list(width = 0.5),opacity = 0.8)

save_plot("scar_v_noscar.pdf",p)


#All Data - WaveFronts

colors <- brewer.pal(n = 3, name = "Set3")

p <- cleaned_aggregate_long_data %>% select(Categorical_Label,signal_data,Row,sheep,WaveFront, Catheter_Type, Point_Number) %>%
  group_by(sheep,WaveFront, Catheter_Type, Point_Number) %>%
  plot_ly(., x = ~Row,y = ~signal_data, type = 'scatter', mode = 'lines',
          color = ~WaveFront, line = list(width = 0.5),
          colors = colors)

save_plot("scar_by_wavefront.pdf",p)





