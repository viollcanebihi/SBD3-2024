library(readr)
library(dlookr)
library(ggplot2)
library(lattice)
library(dplyr)    # for data manipulation
library(caret)    # machine learning workflow
library(rsample) # for re-sampling
library(rpart)       # performing regression trees
library(rpart.plot)  # plotting regression trees
library(randomForest) # implementation of the random forest algorithm
library(finalfit)
library(ROSE)
library(FactoMineR)

options(scipen=999) #scientific notation: off


#################################################
######Step 1: Data – Import and Data Check ######
#################################################
## Import data
# We set the working directory and we load the data that we are going to use. As the data is from a database called "data" we copy this data to the dataframe called "df"
rm(list = ls())

place <- "Nicolas"
#place <- "Viollca"
#place <- "Simon"


{
  # working directory
  if (place == "Nicolas") {
    setwd("/Users/nicolashaldemann/Library/Mobile Documents/com~apple~CloudDocs/Studium/SBD3/Group Project 1")
  } else if (place == "Viollca") {
    setwd("")
  }
  else if (place == "Simon") {
    setwd("")
  }
}
  getwd()

load("data_wage.RData")
df <- data

#At first we will need an understanding of the data by inspecting it. We did this with the following commands 
head(df)
tail(df)
summary(df)
str(df)
names(df)
colnames(df)
dim(df)
str(df)
# Examine if there are missing values
diagnose(df)
missing_pattern(df) #As a (visual) alternative
hist(df$wage) # check the distribution of the target variable
#--> Left skewed: The higher the salary the fewer people who get that much
boxplot_wage <- boxplot(df$wage)
#--> There are  outliers in the upper range above 200'000. Most salaries are in the range of 20'000 to 100'000
diagnose_numeric(df, 1:78) 
# With the diagnose_numeric() function we see that the numeric variables all have many to very many outliers
diagnose_category(df, 1:78)
#For the categorical variables, using the diagnose_category() function, we see that there are no outliers.




data_num <- df %>%
  select_if(is.numeric)
data_num <- as.data.frame(data_num)

boxplot(scale(data_num), xaxt = "n") 
text(x = 1:length(data_num),
     y = par("usr")[3] - 0.8,
     labels = names(data_num),
     xpd = NA,
     srt = 35,
     cex = 0.8,
     adj = 1)
# Through this boxplot see at the top the outliers. 

diagnose_outlier(data_num)
# Here we see an overview of all numerical variables to the outliers.


data_num %>%
  plot_outlier(diagnose_outlier(data_num) %>%
                 filter(outliers_ratio >= 0.5) %>%
                 select(variables) %>%
                 unlist())
# Each variable is visualized with and without outliers.
# Here we see each variable visualized once with and once without outliers. This gives a good overall view.
# We noted that there are outlines which are in reality not. For categories such as coding skills it is not recognized as a categorical value.
# The data is right skewed.
# With the knowledge gained, we now have to restructure the dataset to get more useful data.Therefore the following this have to be done:
# - Reshape the dataset do get categories right
# - Reshape dataset to get even distributed data



#Feature Engineering
#We will use this method to categorize the coding laguage to reduce the number of variables while keep the meaning of the data
# Kopiere df zu df_2
df_2 <- df


# Definiere die Kategorien
objektorientiert <- c("Programming_Java", "Programming_C.C..", "Programming_Python")
funktional <- c("Programming_Scala", "Programming_Julia")
skript <- c("Programming_Python", "Programming_R", "Programming_Bash", "Programming_Javascript.Typescript", "Programming_Visual.Basic.VBA", "Programming_SAS.STATA"   )
Notebooks <- c("Notebooks_Kaggle.Kernels", "Notebooks_Google.Colab", "Notebooks_Azure.Notebook", "Notebooks_Google.Cloud.Datalab", "Notebooks_JupyterHub.Binder")
cloud_plattforms <- c("cloud_Google.Cloud.Platform..GCP.", "cloud_Amazon.Web.Services..AWS.", "cloud_Microsoft.Azure", "cloud_IBM.Cloud", "cloud_Alibaba.Cloud")
data_a <- c("data_Categorical.Data", "data_Genetic.Data", "data_Geospatial.Data", "data_Image.Data", "data_Numerical.Data", "data_Sensor.Data", "data_Video.Data")
data_ex <- c("Programming_Scala", "Programming_Julia")




# Zähle die beherrschten objektorientierten Sprachen
df_2$Count_Objektorientiert <- rowSums(df_2[, objektorientiert] == 1, na.rm = TRUE)

# Zähle die beherrschten funktionalen Sprachen
df_2$Count_Funktional <- rowSums(df_2[, funktional] == 1, na.rm = TRUE)

# Zähle die beherrschten Skriptsprachen
df_2$Count_Skript <- rowSums(df_2[, skript] == 1, na.rm = TRUE)

# Deep Learning Frameworks
df_2$Uses_Deep_Learning <- as.numeric(rowSums(df_2[, c("ML_framework_TensorFlow", "ML_framework_Keras", "ML_framework_PyTorch")] == 1) > 0)

# Traditionelle ML Frameworks
df_2$Uses_Traditional_ML <- as.numeric(rowSums(df_2[, c("ML_framework_Scikit.Learn", "ML_framework_Caret", "ML_framework_Spark.MLlib", "ML_framework_H20")] == 1) > 0)

# Ensemble Learning Frameworks
df_2$Uses_Ensemble_Learning <- as.numeric(rowSums(df_2[, c("ML_framework_Xgboost", "ML_framework_randomForest")] == 1) > 0)

# Zähle die Anzahl der genutzten Notebook-Plattformen (ausschließen der "None"-Kategorie)
df_2$Count_notebooks_used <- rowSums(df_2[, Notebooks] == 1, na.rm = TRUE)


# Zähle die Anzahl der genutzten cloud-Plattformen (ausschließen der "None"-Kategorie)
df_2$Count_Clouds_Used <- rowSums(df_2[, cloud_plattforms] == 1, na.rm = TRUE)

# Ensemble Vizualization
df_2$Uses_Ensemble_Learning <- as.numeric(rowSums(df_2[, c("Visualization_ggplot2", "Visualization_Matplotlib", "Visualization_Altair", "Visualization_Shiny", "Visualization_Plotly")] == 1) > 0)

# Zähle die Anzahl der genutzten daten-Plattformen (ausschließen der "None"-Kategorie)
df_2$data_analized <- rowSums(df_2[, data_a] == 1, na.rm = TRUE)

# Ensemble Vizualization
df_2$explainability_model <- as.numeric(rowSums(df_2[, c("explainability.model_Examine.individual.model.coefficients", "explainability.model_examine.feature.correlations", "explainability.model_Examine.feature.importances", "explainability.model_Create.partial.dependence.plots", "explainability.model_LIME.functions", "explainability.model_SHAP.functions")] == 1) > 0)



# Liste der Spalten, die entfernt werden sollen
columns_to_remove <- c("Programming_Java", "Programming_C.C..", "Programming_Python", "Programming_Scala", "Programming_Julia", 
                       "Programming_R", "Programming_Bash", "Programming_Javascript.Typescript", "Programming_Visual.Basic.VBA", "Programming_SAS.STATA",
                       "ML_framework_TensorFlow", "ML_framework_Keras", "ML_framework_PyTorch", "ML_framework_Scikit.Learn", "ML_framework_Caret", 
                       "ML_framework_Spark.MLlib", "ML_framework_H20", "ML_framework_Xgboost", "ML_framework_randomForest",
                       "Notebooks_Kaggle.Kernels", "Notebooks_Google.Colab", "Notebooks_Azure.Notebook", "Notebooks_Google.Cloud.Datalab", "Notebooks_JupyterHub.Binder", "Notebooks_None",
                       "cloud_Google.Cloud.Platform..GCP.", "cloud_Amazon.Web.Services..AWS.", "cloud_Microsoft.Azure", "cloud_IBM.Cloud", "cloud_Alibaba.Cloud", "cloud_I.have.not.used.any.cloud.providers",
                       "data_Categorical.Data", "data_Genetic.Data", "data_Geospatial.Data", "data_Image.Data", "data_Numerical.Data", "data_Sensor.Data", "data_Video.Data",
                       "Visualization_ggplot2", "Visualization_Matplotlib", "Visualization_Altair", "Visualization_Shiny", "Visualization_Plotly", "Visualization_None",
                       "explainability.model_Examine.individual.model.coefficients", "explainability.model_examine.feature.correlations", "explainability.model_Examine.feature.importances", "explainability.model_Create.partial.dependence.plots", "explainability.model_LIME.functions", "explainability.model_SHAP.functions", "explainability.model_None.I.do.not.use.these.model.explanation.techniques")

# Entferne die Spalten aus df_2
df_2[, columns_to_remove] <- list(NULL)


# Überprüfe das Ergebnis
head(df_2)
tail(df_2)
summary(df_2)
str(df_2)
names(df_2)
colnames(df_2)
dim(df_2)
str(df_2)

ncol(df)

df_2











#As we want a salary, so a numeric value, we want to build a logistic regression model and not a classifier




# The following types will be tried out and compared to each other. The most accurate model will be used for calculating the students expected future salary
#  - Linear Regression 
#  - Regression Tree
#  - Random Forest Regression





###############################################################
######## Step 2: Split data into train and test set  ##########
###############################################################

## Split the data into training and testing sets. Here, we use 70% of the observations for training and 30% for testing.
set.seed(7) # Set random seed to make results reproducible
split  <- initial_split(df_2, prop = 0.7)
data_train  <- training(split)
data_test   <- testing(split)

#roughly the same as in base R:
train.indices <- sample(nrow(df),0.7*nrow(df), replace = FALSE)
data.train.2 <- df[train.indices,]
data.test.2 <- df[-train.indices,]

hist(data_train$wage) # check the distribution of the target variable in train data
hist(data_test$wage) # check the distribution of the target variable in test data


####################################################################
## Step 3: Train and evaluate your model(s) [BASIC IMPLEMENTAION]##
##################################################################

##################################
### Linear Regression ############
##################################

#Train a model explaining price (i.e. Y), specify independent variables (i.e. Xs)
regression_model <- lm(wage ~ ., data = data_train)

# Print results of the model
summary(regression_model)

# Make predictions on the test data to evaluate your model's performance
predictions <-regression_model  %>% predict(data_test)

# Compute the prediction error, RMSE
cat("RMSE (Linear Regression):", RMSE(predictions, data_test$wage))

##################################
### Regression trees ############
##################################

### Train a model explaining price (i.e. Y), specify independent variables (i.e. Xs)
## Set parameters for tree growth: 
# maxdepth= maximum tree depth; cp= complexity parameter (stoping parameter)
tree_model <- rpart(wage ~ .,
                    data = data_train,
                    method = "anova",
                    control = rpart.control(minsplit = 2, minbucket = 1, cp = 0, maxdepth = 5))


# Plot the decision tree
rpart.plot(tree_model, type=5)

# Make predictions on the test data to evaluate your model's performance
predictions <-tree_model %>% predict(data_test)

# Compute the prediction error, RMSE
cat("RMSE (Regression tree):", RMSE(predictions, data_test$price))


##################################
### Random Forest Regression  ###
##################################

### Train a model explaining price (i.e. Y), specify independent variables (i.e. Xs)
## Set parameters specific to random forest: 
# ntree = number of trees to be generated; mtry = number of features used in the construction of each tree. Default value = sqrt(number of features)
randomForest_model <- randomForest(wage ~ .,
                                   data= data_train, 
                                   ntree=50, mtry= 3,
                                   importance=TRUE)

# Plot variable importance
varImpPlot(randomForest_model)

# Make predictions on the test data to evaluate your model's performance
predictions <-randomForest_model  %>% predict(data_test)

# Compute the prediction error, RMSE
cat("RMSE (Random Forest Regression):", RMSE(predictions, data_test$price))



###############################################
#### Your turn: Can you improve these models?
###############################################



