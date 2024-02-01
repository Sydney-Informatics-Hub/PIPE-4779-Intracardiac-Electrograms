
get_paths <- function(sheep_name) {
  labelled_data_path <- here::here("data",sheep_name,"labelled")
  generated_data_path <-  here::here("data","generated")
  main_data_path <- here::here("data",sheep_name,"Export_Analysis")
  assign("generated_data_path", generated_data_path, envir = .GlobalEnv)
  assign("main_data_path", main_data_path, envir = .GlobalEnv)
  assign("labelled_data_path", labelled_data_path, envir = .GlobalEnv)

}

