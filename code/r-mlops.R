# Load Packages and Set Options
setwd("D:/Data Science/tutorials/r-mlops")
library(tidyverse)        # data wrangling and cleaning
library(tidymodels)       # modeling and ml
library(palmerpenguins)   # penguin dataset
library(gt)               # creatiing table objects for data
library(ranger)           # random forest model engine
library(pins)             # sharing resources across sessions and users TODO: pins tutorial?
library(vetiver)          # model versioning and deploymenr
library(plumber)          # api creation
library(conflicted)       # handle common conflicts with tidymodels
tidymodels::tidymodels_prefer()
conflicted::conflict_prefer('penguins', 'palmerpenguins')

# EDA

penguins %>% 
  dplyr::filter(!is.na(sex)) %>% 
  ggplot(aes(x = flipper_length_mm
             , y = bill_length_mm
             , color = sex
             , size = body_mass_g))+
  geom_point(alpha = 0.5)+
  facet_wrap(~species)+
  theme_classic(12)
# For all species, males tend to have longer bill and flipper lengths and heavier body mass. Can predict sex using the other variables.

# remove rows with missing sex, exclude year and island.
penguins_df <- penguins %>% 
  tidyr::drop_na(sex) %>% 
  dplyr::select(-year, -island)

# Set the seed for reproducibility
set.seed(1234)

# Split the data into train and test sets stratified by sex.
penguin_split <- rsample::initial_split(penguins_df, strata = sex)

#######################################################################
# First time using this, inspect the result:
# Produces list of length 4:
# First object is the original data:
penguin_split[[1]]
# Second objects contains the rows to be used in the training data:
penguin_split[[2]]
# Third object - out_id? Not sure...
penguin_split[[3]]
# Third object - id = Resample1?
penguin_split[[4]]
#########################################################################


penguin_train <- rsample::training(penguin_split)
penguin_test <- rsample::testing(penguin_split)

# Create folds for cross validation.
# The vfold_cv() function creates a v-fold cross-validation object, which is 
# used to evaluate the performance of a model on different subsets of the data.
penguin_folds <- rsample::vfold_cv(penguin_train)

# Create recipe for preprocessing data
#########################################################################
# A recipe is a description of the steps to be applied to a data set in #
# order to prepare it for data analysis.                                #
# https://recipes.tidymodels.org/reference/recipe.html                  #
#########################################################################
penguin_rec <- recipes::recipe(sex ~ .
                               , data = penguins_df) %>% 
  # transformation to address skewness.
  recipes::step_YeoJohnson(recipes::all_numeric_predictors()) %>% 
  # transformation to scale and center.
  recipes::step_normalize(recipes::all_numeric_predictors()) %>%  
  # create dummies for categorical variable
  recipes::step_dummy(species)

# Specify three models to use with parsnip.
# 1. glm_spec         - logistic regression
# 2. tree_spec        - random forest
# 3. mlp_brulee_spec  - multilayer perceptron

glm_spec <- parsnip::logistic_reg() %>% 
  parsnip::set_engine('glm')

tree_spec <- parsnip::rand_forest(min_n = tune()) %>% 
  parsnip::set_engine('ranger') %>% 
  parsnip::set_mode('classification')

mlp_brulee_spec <- parsnip::mlp(hidden_units = tune::tune()
                                , epochs = tune::tune()
                                , penalty = tune::tune()
                                , learn_rate = tune::tune()) %>% 
  parsnip::set_engine('brulee') %>% 
  parsnip::set_mode('classification')

# Create the workflow set and fit models
# 
# The control_bayes() creates an object to store the settings for Bayesian 
# optimization. Bayesian optimization is a method for finding the optimal 
# set of hyperparameters for a machine learning model. The no_improve argument 
# specifies the number of consecutive iterations with no improvement in 
# the objective function before the optimization process is terminated. The 
# time_limit argument specifies the maximum amount of time that the 
# optimization process can run in minutes. The save_pred argument specifies 
# whether to save the predictions made during the optimization process.

bayes_control <- tune::control_bayes(no_improve = 10L
                                     , time_limit = 2# 20
                                     , save_pred = T
                                     , verbose = T)

# A workflow set combines the recipes and models to fit to our training data.
# The workflow_set() function creates a workflow set object, which consists of 
# 1. a list of preprocessing recipes in the preproc argument and 
# 2. a list of modelling specifications in the models argument.

# NOTE: Having issues with torch.
# TODO: Debug torch and retry with the mlp included.
workflow_set <- workflowsets::workflow_set(preproc = list(penguin_rec)
                                           , models = list(glm = glm_spec
                                                           , tree = tree_spec
                                                           #, torch = mlp_brulee_spec
                                           )) %>% 
  workflowsets::workflow_map("tune_bayes"
                             , iter = 5L #50L
                             , resamples = penguin_folds
                             , control = bayes_control)

# Compare model results
workflowsets::rank_results(workflow_set
                           , rank_metric = 'roc_auc'
                           , select_best = T) %>% 
  gt::gt()

# Throughout many tidymodels packages, autoplot is a handy method to rapidly visualize steps in a model workflow.
workflow_set %>% workflowsets::autoplot()
# Logistic regression ranks first on all metrics. Use it going forward.
best_model_id <- "recipe_glm"
# Finalize Model
# Fit the model to the full dataset.
# In the code below, the best_fit object is extract the best model 
# from the workflow using the workflow ID we selected above. This 
# is done with workflowsets::extract_workflow_set_result() and 
# tune::select_best() to give us best_fit, a tibble of hyperparameters 
# for the best fit model.

best_fit <- workflow_set %>% 
  workflowsets::extract_workflow_set_result(best_model_id) %>% 
  tune::select_best(metric = "accuracy")
best_fit
# We can then use finalize_workflow() to take the hyperparameters from 
# best_fit and apply it to the final_workflow object. We can then update 
# the fit of the model to use the entire training set instead of folds 
# and evaluate the model on the test set.
final_workflow <- workflow_set %>% 
  tune::extract_workflow(best_model_id) %>% 
  tune::finalize_workflow(best_fit)
final_workflow

final_fit <- final_workflow %>% 
  tune::last_fit(penguin_split)
final_fit
# Check model performances.
final_fit %>% 
  collect_metrics() %>% 
  gt()

final_fit %>% 
  tune::collect_predictions()  %>% 
  yardstick::roc_curve(sex, .pred_female) %>% 
  autoplot()

# Model Deployment

# The {vetiver} package provides a set of tools for building, deploying, and 
# managing machine learning models in production. It allows users to easily 
# create, version, and deploy machine learning models to various hosting 
# platforms, such as Posit Connect or a cloud hosting service like Azure.

# The vetiver_model() function is used to create an object that stores a machine 
# learning model and its associated metadata, such as the model’s name, type, 
# and parameters. vetiver_pin_write() and vetiver_pin_read() functions are 
# used to write and read vetiver_model objects to and from a server.

# To deploy our model with {vetiver}, we start with our final_fit from above, 
# we first need to extract the trained workflow. We can do that with tune::extract_workflow(). 
# The trained workflow is what we will deploy as a vetiver_model. That means 
# we need to convert it from a workflow to a vetiver model with vetiver_model().

# Create Vetiver Model
final_fit_to_deploy <- final_fit %>% tune::extract_workflow()
final_fit_to_deploy
v <- vetiver::vetiver_model(final_fit_to_deploy, model_name = 'penguins_model')
v

# Pin Model to Board

# The {pins} package is used for storing and managing data sets in a local or remote repository. 
# {pins} allows users to “pin” data sets to a “board”, allowing them to be easily accessed 
# and shared with others. Using the pins package, users can create a board, add data sets, and 
# access and retrieve data sets from the board. The board_rsconnect() function is used to 
# create a model_board or connect to an existing board on Posit Connect (formerly RStudio Connect), 
# which is a connection to a server where a vetiver_model can be stored and accessed. 
# We also specify versioned = TRUE so that we can version control our vetiver models.

# Once the model_board connection is made it’s as easy as vetiver_pin_write() to “pin” our model
# to the model board and vetiver_pin_read() to access it. In this case, we must specify the username 
# of the author of the pin, which in this case is cahc

model_board <- pins::board_local(versioned = T)
model_board
model_board %>% vetiver::vetiver_pin_write(v)