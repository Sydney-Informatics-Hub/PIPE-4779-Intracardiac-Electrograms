
get_paths <- function() {
  generated_data_path <-  here::here("data","generated")
  assign("generated_data_path", generated_data_path, envir = .GlobalEnv)

}

