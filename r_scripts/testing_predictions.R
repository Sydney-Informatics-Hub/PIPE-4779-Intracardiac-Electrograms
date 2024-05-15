library(here)
library(arrow)
library(tidyverse)
source(here::here("r_scripts","paths.R"))
get_paths()
path_models <- here::here("python","models")
deploy_data_path <- here::here("deploy","data")

# truth is from publishable data saved by running rscripts data_injest.R + post_aggregation.R during testing
# df_python dataframe is the result of data_injest.py
# prediction collects the results of running classify_ecg.py

# this file checks why prediction is not close to truth. Represents only S18 and Penta cases.

parquet_file <- here::here(generated_data_path,paste0("publishable_model_data_TSAI","imputed",".parquet"))

# 6909 rows
truth <- read_parquet(parquet_file) %>% filter(sheep == "S18") %>%
  select(Point_Number,WaveFront,X,Y,Z,depth_label,signal_data)


# 6909 rows
df_python <- read_parquet(here(deploy_data_path,"preprocessed_rawsignal_unipolar_penta.parquet")) %>%
  arrange(WaveFront,Point_Number) %>% mutate(WaveFront = as_factor(WaveFront)) %>% as_tibble()

nrow(truth) == nrow(df_python) #TRUE truth and the data_injest are accurate


# List all Parquet file predictions that have been run
# SR, LVp RVp combinations by
#with prediction = -1 or 1 and target = NoScar
parquet_files <- list.files(path = here::here(path_models), pattern = "\\.parquet$", full.names = TRUE)

# Read Parquet files and row-bind them into one dataframe
predictions <- purrr::map_dfr(parquet_files, arrow::read_parquet)

nrow(predictions) == nrow(truth)  #TRUE if only 1 target for SR, LVp RVp  is picked up


predictions <- predictions %>% mutate(forecast = ifelse(prediction == 1,target,0))

predictions <- predictions %>% filter(forecast != 0) # rows 9025 should be 6909. Possibly different wavelength forecasts give different results
#so you end up with more

# redo above but specify groups - something wrong

preds_scar <-  purrr::map_dfr(list(
    here::here(path_models,"predictions_clf_NoScar_LVp_120epochs.parquet"),
    here::here(path_models,"predictions_clf_NoScar_RVp_120epochs.parquet"),
    here::here(path_models,"predictions_clf_NoScar_SR_120epochs.parquet")
), arrow::read_parquet)

nrow(preds_scar) == 6909 #TRUE


preds_endo <-  purrr::map_dfr(list(
  here::here(path_models,"predictions_clf_AtLeastEndo_LVp_120epochs.parquet"),
  here::here(path_models,"predictions_clf_AtLeastEndo_RVp_120epochs.parquet"),
  here::here(path_models,"predictions_clf_AtLeastEndo_SR_120epochs.parquet")
), arrow::read_parquet)

nrow(preds_endo) == 6909 #TRUE


preds_intra <-  purrr::map_dfr(list(
  here::here(path_models,"predictions_clf_AtLeastIntra_LVp_120epochs.parquet"),
  here::here(path_models,"predictions_clf_AtLeastIntra_RVp_120epochs.parquet"),
  here::here(path_models,"predictions_clf_AtLeastIntra_SR_120epochs.parquet")
), arrow::read_parquet)

nrow(preds_intra) == 6909 #TRUE


preds_epi <-  purrr::map_dfr(list(
  here::here(path_models,"predictions_clf_epiOnly_LVp_120epochs.parquet"),
  here::here(path_models,"predictions_clf_epiOnly_RVp_120epochs.parquet"),
  here::here(path_models,"predictions_clf_epiOnly_SR_120epochs.parquet")
), arrow::read_parquet)

nrow(preds_epi) == 6909 #TRUE

# so conclusion would be pick one model say RVP and apply and it should be consistent.
# Or alter the classify_ecg.py model to create different instances of the TSAI class
# that is based on wavefront and apply the specific class to what the row in the dataframe is
# so you dont run them all and aggregate


# testing for exlusivity - do the models predict more than 2 classifications for the same signal

preds_scar <- preds_scar %>% mutate(outcome = ifelse(prediction == 1,target,"NotClassified"))

preds_endo <- preds_endo %>% mutate(outcome = ifelse(prediction == 1,target,"NotClassified"))


preds_intra <- preds_intra %>% mutate(outcome = ifelse(prediction == 1,target,"NotClassified"))

preds_epi<- preds_epi %>% mutate(outcome = ifelse(prediction == 1,target,"NotClassified"))

nrow(truth) == nrow(preds_scar) + nrow(preds_endo) + nrow(preds_intra) + nrow(preds_epi)
# 6909 NOT EQUAL TO 5436 + 1546 + 1405 + 638
total_predictions = nrow(preds_scar) + nrow(preds_endo) + nrow(preds_intra) + nrow(preds_epi)
# 2116 overlapping predictions
total_predictions - nrow(truth)

#where are the overlapping outcomes.
truth_to_outcome <- truth %>% select(-signal_data) %>%
  left_join(.,preds_scar,by = c("Point_Number","WaveFront","X","Y","Z")) %>%
  rename(outcome_scar = outcome) %>% rename(probability_scar = probability) %>% select(-c(prediction,target))

truth_to_outcome <- truth_to_outcome %>%
  left_join(.,preds_endo,by = c("Point_Number","WaveFront","X","Y","Z")) %>%
  rename(outcome_endo = outcome) %>% rename(probability_endo = probability) %>% select(-c(prediction,target))


truth_to_outcome <- truth_to_outcome %>%
  left_join(.,preds_intra,by = c("Point_Number","WaveFront","X","Y","Z")) %>%
  rename(outcome_intra = outcome) %>% rename(probability_intra = probability) %>% select(-c(prediction,target))

truth_to_outcome <- truth_to_outcome %>%
  left_join(.,preds_epi,by = c("Point_Number","WaveFront","X","Y","Z")) %>%
  rename(outcome_epi = outcome) %>% rename(probability_epi = probability) %>% select(-c(prediction,target))

#there will be different overlapping combinations - this one holds 284 overlaping prediction outcomes.
overlapping <- truth_to_outcome %>%
  filter(outcome_scar != "NotClassified" & outcome_endo != "NotClassified" ) %>%
  select(Point_Number,WaveFront,X,Y,Z,outcome_scar,outcome_endo,probability_scar,probability_endo)


