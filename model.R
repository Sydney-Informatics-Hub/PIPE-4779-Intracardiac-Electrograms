library(tidyverse)
library(tidymodels)
source("paths.R")
get_paths()
model_data <- readRDS(file = here::here(generated_data_path,"model_data.rds"))

# Include certain features that should be used for prediction
model_data <- model_data %>% select(unipolar_voltage,bipolar_voltage,LAT, #signal settings
                      Categorical_Label,endocardium_scar,intramural_scar, epicardial_scar, #labels will be excluded later
                      mean,standard_deviation,sum,positivesum,positivemean,duration, #aggregate features of signal
                      phase_mean,phase_var,magnitude_mean # aggregate features of fft
                      )

# Note positional data, sheep info are not be used as features.

# Predicting Scar or NoScar only at this stage and not depth.
model_data <- model_data %>% select(-c(endocardium_scar,intramural_scar,epicardial_scar)) %>%
  mutate(Categorical_Label = as.factor(Categorical_Label))

#For orange exploration
write_csv(model_data,here::here(generated_data_path,"model_data1.csv"))

# Define your recipe to preprocess the data
data_recipe <- recipe(Categorical_Label ~ ., data = model_data) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())

# Split the data into training and testing sets
set.seed(123) # for reproducibility
data_split <- initial_split(model_data, prop = 0.8, strata = Categorical_Label)
data_train <- training(data_split)
data_test <- testing(data_split)

# Set up your model specification
model_spec <- multinom_reg() %>%
  set_engine("nnet") # Using nnet package for multinomial regression

# Train the model
trained_model <- workflow() %>%
  add_recipe(data_recipe) %>%
  add_model(model_spec) %>%
  fit(data = data_train)

# Make predictions on the test set
predictions <- predict(trained_model, data_test) %>%
  bind_cols(data_test)

#here!
# Evaluate model performance
conf_mat <- conf_mat(predictions, truth = Categorical_Label, estimate = .pred_class)
accuracy <- accuracy(conf_mat)

# Print the confusion matrix and accuracy
print(conf_mat)
print(accuracy)
