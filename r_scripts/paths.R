
get_paths <- function() {
  generated_data_path <-  here::here("data","generated")
  assign("generated_data_path", generated_data_path, envir = .GlobalEnv)

}


get_sheep_path <- function(sheep) {
  generated_data_path <-  here::here("data","generated")
  assign("generated_data_path", generated_data_path, envir = .GlobalEnv)
  labelled_data_path <- here::here("data",sheep,"labelled")
  assign("labelled_data_path", labelled_data_path, envir = .GlobalEnv)
  main_data_path <- here::here("data",sheep,"Export_Analysis")
  assign("main_data_path", main_data_path, envir = .GlobalEnv)

}

