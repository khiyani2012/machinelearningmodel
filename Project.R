---
title : "Spotify Song ML Model Recommendation"
---
  
1) Data Acquisition
Loading the dataset
```{r}

case_study_data <- read_csv("Desktop/Data Mining - SEM2/case_study_data.csv")
View(case_study_data)
data <- case_study_data

```

2) Data Exploration

-Exploratory Data plot

str(data)
dim(data)

#the str function gives us the structure of the dataset
#the dim function gives us the dimensions of the dataset

3) Data Cleaning and Shaping

The columns X1, song_title, artist has no significance in datset, so we need to clean the data
by dropping the above columns. Column User_Choices i.e if the user likes the song 
then the value is 1 and if user dislike the song the value will be 0.
This dataset does not contain any NA values so there is no requirement 
of cleaning the dataset anymore so no imputation is needed. 

```{r}
data <- data[,2:14]
data$User_choices <- data$mode
cleaned_df <- data
cleaned_df
#we have now created a cleaned dataframe
```

4) Correlation Analysis

#corrplot function is used for correlation analysis

```{r}

CorAll<- corrplot(cor(cleaned_df[,2:14]), tl.cex = 0.7, col = col(200),  addCoef.col = "black",  tl.col="black"
        ,type = "upper",  insig = "blank", tl.srt=45, sig.level = 0.01)
CorAll

```

5) Normalizing the data using PCA
we use the prcomp function to normalize the data 

```{r}
pc <- prcomp(cleaned_df, center = TRUE, scale = TRUE)
> plot(pc, type="l", main = " ")
> box()
> grid(nx = 10, ny = 10)


data_norm <- function(x) { ((x - min(x))/(max(x) - min(x)))}
cleaned_df_norm <- as.data.frame(lapply(cleaned_df, data_norm))
summary(cleaned_df) # All the values here are not normalized
summary(cleaned_df_norm) # All the values are normalized i.e All values are "0" & "1"

```


6) Model building and Evaluation

we first have to divide the dataset into training and validation subsets for accuracy prediction
we create a 70-30 split of the dataset.

training set consists of 70% of the cleaned_df and validation set consist of 30% of cleaned_df

```{r}
library(caret)
set.seed(1)
cleaned_df_Partition <-createDataPartition(cleaned_df$User_choices,p=0.70,list=FALSE)
Train <- cleaned_df[cleaned_df_Partition,]
Test <- cleaned_df[-cleaned_df_Partition,]
Validate <- Test

Train_1 <- Train[,-14]
Test_1 <- Test[,-14]
cl <- Train[,14]
cl <- factor(as.data.frame(cl))

is.na(cl)
is.na(Train_1)
is.na(Test_1)
Knn_model1 <- knn(Train_1,Test_1,cl,k=5)

dim(Train_1)
dim(Test_1)
length(cl)

```

Model 1 - KNN Model

Before building KNN model we have to first create control parameters. Defining control 
parameters helps in evaluating the model. 

A 5 fold cross validation will be used to define control parameters

```{r}

ControlParam <- trainControl(method = "cv",
                            number = 5,
                            savePredictions = TRUE,
                            classProbs = TRUE)

```

#Use the Control parameter and train the model with training dataset

```{r}

KNN <- train(User_choices ~., data = Train, method = "knn",  preProcess= c('center', 'scale'),tuneLength= 10,
       trControl= ControlParam)
```

```{r}
#predicting the model using validation dataset

KNN_Pred <- predict(KNN , Validate)

```

```{r}


#Using Confusion Matrix to get accuracy of model

K_CM <- confusionMatrix(KNN_Pred, Validate$User_choices)
K_CM

```

2) Random Forest Model
- we use the same control parameters in this model as well

A 5 fold cross validation will be used to define control parameters

```{r}


ControlParam1 <- trainControl(method = "cv",
                              number = 5,
                              savePredictions = TRUE,
                              classProbs = TRUE)

```

#Use the Control parameter and train the model with training dataset

```{r}

RF_Model <- randomForest(User_choices~., data = Train)
                 
```

```{r}
#predicting the model using validation dataset

RF_Pred <- predict(RF_Model , Validate)

```

```{r}


#Using Confusion Matrix to get accuracy of model


R_CM <- confusionMatrix(RF_Pred, Validate$User_choices)
R_CM

```

If accuracy is considered then the random forest is better than the knn.

3) SVM Model 

```{r}

SVM_Model <- svm(User_choices ~., Train, kernel = "radial")
SVM_Predict <- predict(SVM_Model, Validate)
S_CM <- confusionMatrix(SVM_Predict, Validate$User_choices)
S_CM

```





```{r}



#The model with the high accuracy

Optimal_Model<- c(K_CM$overall[1], R_CM$overall[1], S_CM$overall[1])
names(Optimal_Model) <- c("KNN Model", "Random Forest Model", "SVM Model")
Best_Model <- subset(Optimal_Model, Optimal_Model==max(Optimal_Model))
Best_Model

```




























