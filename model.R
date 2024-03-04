library(tidyverse)
library(tidymodels)
source("paths.R")
get_paths()
data_type <- "filtered"
data_path <- paste0("model_data",data_type,".rds")
#Havent pursued this path fully yet. Orange Data Mining Used instead for now
model_data <- readRDS(file = here::here(generated_data_path,data_path))

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
#conf_mat <- conf_mat(predictions, truth = Categorical_Label, estimate = .pred_class)
#accuracy <- accuracy(conf_mat)

# Print the confusion matrix and accuracy
#print(conf_mat)
#print(accuracy)
