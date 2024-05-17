library(here)
library(arrow)
library(tidyverse)
source(here::here("r_scripts","paths.R"))
get_paths()
path_models <- here::here("python","models")
deploy_data_path <- here::here("deploy","data")
# this file checks why prediction is not close to truth. Represents only S18 and Penta cases and All Wavefronts.


# truth is from publishable data saved by running rscripts data_injest.R + post_aggregation.R during testing
# df_python dataframe is the result of data_injest.py
# prediction collects the results of running classify_ecg.py



parquet_file <- here::here(generated_data_path,paste0("publishable_model_data_TSAI","imputed",".parquet"))

# 6909 rows
truth <- read_parquet(parquet_file) %>% filter(sheep == "S18") %>%
  select(Point_Number,WaveFront,X,Y,Z,depth_label,signal_data)


# 6909 rows
df_python <- read_parquet(here(deploy_data_path,"preprocessed_rawsignal_unipolar_penta.parquet")) %>%
  arrange(WaveFront,Point_Number) %>% mutate(WaveFront = as_factor(WaveFront)) %>% as_tibble()

nrow(truth) == nrow(df_python) #TRUE truth and the data_injest are accurate


#Conclusion: deploy data injest behaves as expected. Was tested more thoroughly in testing.R


# List all Parquet file predictions that have been run
# SR, LVp RVp combinations by
#with prediction = -1 or 1 and target = NoScar
parquet_files <- list.files(path = here::here(deploy_data_path), pattern = "\\.parquet$", full.names = TRUE)

# Read Parquet files and row-bind them into one dataframe
predictions <- purrr::map_dfr(parquet_files, arrow::read_parquet)

nrow(predictions) == nrow(truth) #false

did_not_make_prediction <- predictions %>% filter(is.na(prediction)) # 13818 rows are neither 1 or -1 ?

predictions <- predictions %>% filter(!is.na(prediction))

predictions %>% distinct(WaveFront) # All 3

predictions <- predictions %>% mutate(forecast = ifelse(prediction == 1,target,0))

predictions <- predictions %>% filter(forecast != 0) #6462 rows


# redo above but specify groups - something wrong

preds_scar <-  purrr::map_dfr(list(
    here::here(deploy_data_path,"predictions_clf_NoScar_LVp_120epochs.parquet"),
    here::here(deploy_data_path,"predictions_clf_NoScar_RVp_120epochs.parquet"),
    here::here(deploy_data_path,"predictions_clf_NoScar_SR_120epochs.parquet")
), arrow::read_parquet)

nrow(preds_scar) == 6909 #TRUE


preds_endo <-  purrr::map_dfr(list(
  here::here(deploy_data_path,"predictions_clf_AtLeastEndo_LVp_120epochs.parquet"),
  here::here(deploy_data_path,"predictions_clf_AtLeastEndo_RVp_120epochs.parquet"),
  here::here(deploy_data_path,"predictions_clf_AtLeastEndo_SR_120epochs.parquet")
), arrow::read_parquet)

nrow(preds_endo) == 6909 #TRUE


preds_intra <-  purrr::map_dfr(list(
  here::here(deploy_data_path,"predictions_clf_AtLeastIntra_LVp_120epochs.parquet"),
  here::here(deploy_data_path,"predictions_clf_AtLeastIntra_RVp_120epochs.parquet"),
  here::here(deploy_data_path,"predictions_clf_AtLeastIntra_SR_120epochs.parquet")
), arrow::read_parquet)

nrow(preds_intra) == 6909 #TRUE


preds_epi <-  purrr::map_dfr(list(
  here::here(deploy_data_path,"predictions_clf_epiOnly_LVp_120epochs.parquet"),
  here::here(deploy_data_path,"predictions_clf_epiOnly_RVp_120epochs.parquet"),
  here::here(deploy_data_path,"predictions_clf_epiOnly_SR_120epochs.parquet")
), arrow::read_parquet)

nrow(preds_epi) == 6909 #TRUE


# Conclusion: number of predictions by case match the deploy data ingest..

# testing for exlusivity - do the models predict more than 2 classifications for the same signal

preds_scar <- preds_scar %>% mutate(outcome = ifelse(prediction == 1,target,"NotClassified"))
preds_endo <- preds_endo %>% mutate(outcome = ifelse(prediction == 1,target,"NotClassified"))
preds_intra <- preds_intra %>% mutate(outcome = ifelse(prediction == 1,target,"NotClassified"))
preds_epi<- preds_epi %>% mutate(outcome = ifelse(prediction == 1,target,"NotClassified"))


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

#there will be different overlapping combinations - this one holds 117 overlaping prediction outcomes.
overlapping <- truth_to_outcome %>%
  filter(outcome_scar != "NotClassified" & outcome_endo != "NotClassified" ) %>%
  select(Point_Number,WaveFront,X,Y,Z,outcome_scar,outcome_endo,probability_scar,probability_endo)


# Assuming one Wavefront Model works lets compare truth with prediction outcome

accuracy_NoScar <- truth %>% left_join(.,preds_scar, by = c("Point_Number","X","Y","Z","WaveFront")) %>%
  select(-c(prediction,target)) %>% filter(outcome != "NotClassified") %>%
  mutate(correct = ifelse(depth_label == outcome,1,0))


accuracy_Endo <- truth %>% left_join(.,preds_endo, by = c("Point_Number","X","Y","Z","WaveFront")) %>%
  select(-c(prediction,target)) %>% filter(outcome != "NotClassified") %>%
  mutate(correct = ifelse(depth_label == outcome,1,0))

accuracy_Intra <- truth %>% left_join(.,preds_intra, by = c("Point_Number","X","Y","Z","WaveFront")) %>%
  select(-c(prediction,target)) %>% filter(outcome != "NotClassified") %>%
  mutate(correct = ifelse(depth_label == outcome,1,0))

accuracy_Epi <- truth %>% left_join(.,preds_epi, by = c("Point_Number","X","Y","Z","WaveFront")) %>%
  select(-c(prediction,target)) %>% filter(outcome != "NotClassified") %>%
  mutate(correct = ifelse(depth_label == outcome,1,0))

# Accuracy for ensumble
sum(accuracy_NoScar$correct) / nrow(accuracy_NoScar) # 98%

sum(accuracy_Endo$correct) / nrow(accuracy_Endo) #64%

sum(accuracy_Intra$correct) / nrow(accuracy_Intra) #45%

sum(accuracy_Epi$correct) / nrow(accuracy_Epi) #15%

# Conclusion: Accuracy varies significantly with each wavefront model. How much?


accuracy_NoScar %>% group_by(WaveFront) %>% summarise(accuracy = mean(correct))


dataframes <- list(accuracy_NoScar, accuracy_Endo, accuracy_Intra, accuracy_Epi)
summarise_accuracy <- function(df) {
  df %>%
    group_by(WaveFront) %>%
    summarise(accuracy = mean(correct))
}

# Apply the function to each data frame and collect results
results <- dataframes %>%
  map_df(summarise_accuracy, .id = "source") %>%
  mutate(source = c(rep("accuracy_NoScar",3),
                    rep("accuracy_Endo",3),
                    rep("accuracy_Intra",3),
                    rep("accuracy_Epi",3)))

print(results)

write_csv(results,file = here::here(deploy_data_path,"accuracy.csv"))
