library(tidyverse)
library(tidymodels)
library(vroom)
library(dplyr)
library(skimr)
library(GGally)
library(DataExplorer)
library(ggplot2)
library(patchwork)
library(poissonreg)
library(lubridate)
library(rpart)
library(glmnet)
bikeshare <- vroom("train.csv")
biketest <- vroom("test.csv")

glimpse(bikeshare) #lists the variable type of each column
skim(bikeshare) #nice overview of the dataset
plot_intro(bikeshare) #visualization of glimpse()
corrplot <- plot_correlation(bikeshare) #correlation heat map between variables
barplot <- plot_bar(bikeshare) # bar charts of all discrete variables
histo <- plot_histrograms(bikeshare) # histograms of all numerical variables
plot_missing(bikeshare) # percent missing in each column
ggpairs(bikeshare) # 1/2 scatterplot and 1/2 correlation heat map





# Create bar plot for weather
p1 <- ggplot(bikeshare, aes(x = weather)) +
  geom_bar(fill = "skyblue") +
  theme_minimal() +
  labs(title = "Weather Distribution", x = "Weather", y = "Count")

# Scatter plot of temperature vs. registered users
p2 <- ggplot(bikeshare, aes(x = temp, y = count)) +
  geom_point() +
  geom_smooth(se=FALSE) +
  labs(title = "Temperature vs. Count", x = "Temperature", y = "Count")

# Scatter plot of humidity vs. registered users
p3 <- ggplot(bikeshare, aes(x = humidity, y = count)) +
  geom_point() +
  geom_smooth(se=FALSE) +
  labs(title = "Humidity vs. Count", x = "Humidity", y = "Count")

# Scatter plot of windspeed vs. registered users
p4 <- ggplot(bikeshare, aes(x = windspeed, y = count)) +
  geom_point() +
  geom_smooth(se=FALSE) +
  labs(title = "Windspeed vs. Count", x = "Windspeed", y = "Count")

#2x2 plot

(p1 + p2) / (p3 + p4)

#Data Cleaning

myCleanData <- bikeshare %>%
  select(datetime, season, holiday, workingday, weather, temp, atemp, humidity, windspeed, -casual, -registered, count) %>%
  mutate(
    hour = factor(hour(datetime), labels = paste0(1:12, c(" AM", " PM"))[rep(1:12, 2)]), # Extracts the hour and converts it to a factor with AM/PM
    weather = factor(weather, labels = c("Clear", "Mist", "Light Snow/Rain", "Heavy Rain/Snow")), # Convert weather to a factor with descriptive labels
    season = factor(season, labels = c("Spring", "Summer", "Fall", "Winter")), # Convert season to a factor with descriptive labels
    logcount = log(count + 1), # Create a new variable for log(count + 1)
    rush_hour = factor(
      ifelse(
        workingday == 1 & (hour %in% c("8 AM", "9 AM", "10 AM", "4 PM", "5 PM", "6 PM")),
        "Yes", 
        "No"
      )
    ) # Create rush_hour variable
  ) %>%
  select(-count, -datetime) # Remove the original count column


#Recipe making


# Create the recipe, keeping 'count' until after fitting the model
my_recipe <- recipe(count ~ season + holiday + workingday + weather + temp + atemp + humidity + windspeed + datetime, data = bikeshare) %>%
  step_time(datetime, features = "hour") %>%  # Extract hour from datetime
  step_date(datetime, features = "dow") %>%   # Extract day of week (dow) from datetime
  step_mutate(
    weather = factor(weather, levels = 1:4, labels = c("Clear", "Mist", "Light Snow/Rain", "Heavy Rain/Snow")), # Convert weather to factor
    season = factor(season, levels = 1:4, labels = c("Spring", "Summer", "Fall", "Winter"))) # Convert season to factor

# Define the linear regression model
lin_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

# Combine the recipe and model into a workflow
bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(lin_model)

# Fit the workflow with the 'logcount' outcome
bike_workflow_fit <- fit(bike_workflow, data = bikeshare)

# Apply the workflow to new (test) data
# Make sure 'biketest' has the same columns and transformations applied
lin_preds <- predict(bike_workflow_fit, new_data = biketest)

kaggle_submission <- lin_preds %>%
  bind_cols(., biketest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=kaggle_submission, file="./Featureeng.csv", delim=",")




###Linear Regression

## Setup and Fit the Linear Regression Model
bikelm <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression") %>%
  fit(formula = log(count) ~ datetime + season + holiday + workingday + weather + temp +
        atemp + humidity + windspeed, data = bikeshare)


## Generate Predictions Using Linear Model
bike_predictions <- predict(bikelm,
                            new_data= biketest) 
bike_predictions


## Format the Predictions for Submission to Kaggle
kaggle_submission <- bike_predictions %>%
bind_cols(., biketest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")


###Poisson Regression

poissonlm <- poisson_reg() %>%
  set_engine("glm") %>%
  set_mode("regression") %>%
  fit(formula = count ~ DOW + season + holiday + workingday + weather + temp +
        atemp + humidity + windspeed, data = bikeshare)

poisson_predictions <- predict(poissonlm,
                            new_data= biketest) 
glimpse(poisson_predictions)


#Kaggle submission
kaggle_poisson_submission <- poisson_predictions %>%
  bind_cols(., biketest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_poisson_submission, file="./PoissonPreds.csv", delim=",")


### Penalized regression
library(poissonreg) #if you want to do penalized, poisson regression

## Create a recipe
my_recipe <- recipe(count ~ season + holiday + workingday + weather + temp + atemp + humidity + windspeed + datetime, data = bikeshare) %>%
  step_time(datetime, features = "hour") %>%  # Extract hour from datetime
  step_date(datetime, features = "dow") %>%   # Extract day of week (dow) from datetime
  step_mutate(
    datetime_hour = as.factor(datetime_hour),
    datetime_dow = as.factor(datetime_dow),
    weather = ifelse(weather == 4, 3, weather),
    weather = factor(weather, levels = 1:3, labels = c("Clear", "Mist", "Snow/Rain")), # Convert weather to factor
    season = factor(season, levels = 1:4, labels = c("Spring", "Summer", "Fall", "Winter")))%>%
    #count = log(count))%>%
  step_dummy(all_nominal_predictors()) %>% #make dummy variables
  step_normalize(all_numeric_predictors())%>% # Make mean 0, sd=1
  step_rm(datetime)
  

## Penalized regression model
preg_model <- linear_reg(penalty=7, mixture=1) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R
preg_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(preg_model) %>%
fit(data=bikeshare)
penalized_predictions <- predict(preg_wf, new_data= biketest)

kaggle_penalized_submission <- penalized_predictions %>%
  bind_cols(., biketest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count))%>%
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_penalized_submission, file="./PenalizedPreds.csv", delim=",")
