# Libraries
library(tidyverse)
library(jsonlite)
library(tidymodels)
library(textrecipes)
library(vroom)
library(kernlab)
library(LiblineaR)

tidymodels::tidymodels_prefer()

#----------------------------------------------------------
# 1. Read Data
#----------------------------------------------------------

trainData <- read_file("GitHub/WhatsCooking/train.json") %>% fromJSON()
testData  <- read_file("GitHub/WhatsCooking/test.json") %>% fromJSON()

# Ingredients are lists → collapse into one string per recipe
trainData <- trainData %>%
  mutate(ingredients = map_chr(ingredients, ~ paste(.x, collapse = " ")))

testData <- testData %>%
  mutate(ingredients = map_chr(ingredients, ~ paste(.x, collapse = " ")))

#----------------------------------------------------------
# 2. Recipe for tokenization + TF–IDF
#----------------------------------------------------------

cook_recipe <- recipe(cuisine ~ ingredients, data = trainData) %>%
  step_tokenize(ingredients) %>%  
  step_tokenfilter(ingredients, max_tokens = 3000) %>%   # larger vocab → better accuracy
  step_tfidf(ingredients)

#----------------------------------------------------------
# 3. Linear SVM model (kernlab)
#----------------------------------------------------------

svm_model <- svm_linear() %>%
  set_engine("LiblineaR") %>%
  set_mode("classification")

#----------------------------------------------------------
# 4. Workflow
#----------------------------------------------------------

cook_wf <- workflow() %>%
  add_recipe(cook_recipe) %>%
  add_model(svm_model)

#----------------------------------------------------------
# 5. Fit model
#----------------------------------------------------------

cook_fit <- cook_wf %>% fit(trainData)

#----------------------------------------------------------
# 6. Predict test set
#    (the recipe transforms test data automatically)
#----------------------------------------------------------

preds <- predict(cook_fit, testData)

submission <- tibble(
  id = testData$id,
  cuisine = preds$.pred_class
)

#----------------------------------------------------------
# 7. Write submission file
#----------------------------------------------------------

vroom_write(submission,
            "GitHub/WhatsCooking/svm.csv",
            delim = ",")

beep(2)
