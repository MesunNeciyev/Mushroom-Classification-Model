##Import libraries
library(tidyverse)
library(h2o)
library(skimr)
library(rstudioapi)
library(plotly)
library(data.table)
library(highcharter)
library(glue)
library(caret)



#Importing data
path<-dirname(getSourceEditorContext()$path)
setwd(path)

inspectdf::inspect_na(raw)

raw<-fread("mushrooms.csv")
raw %>% view()
raw %>% skim()
raw %>% glimpse()


#Data cleaning
names(raw) <- names(raw) %>% 
  str_replace_all("`","_") %>% 
  str_replace_all(" ","_") %>% 
  str_replace_all("-","_") %>% 
  str_replace_all("/","_")

raw<-raw %>% select(`bruises%3F`,everything()) %>% rename(bruises=`bruises%3F`)

for (c in 1:ncol(raw)) {
  for (r in 1:nrow(raw)) {
    raw[[c]][r]<-gsub("'","",raw[[c]][r])
    }
  }

raw$class<-raw$class %>% recode("e"=1,"p"=0)
raw$class<-raw$class %>% as.factor()
raw<-raw %>% select(class,everything())
view(raw)



#-------------------Modeling--------------
h2o.init()
h2o_data<-raw %>% as.h2o()

#Spliting data to train and test
h2o_data<-h2o_data %>% h2o.splitFrame(ratios = 0.8,seed = 123)
train<-h2o_data[[1]]
test<-h2o_data[[2]]


#Spliting data to target and feature
target<-'class'
features<-raw %>% select(-target) %>% names()

gc()


#Fitting h2o modellin
model <- h2o.automl(
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  leaderboard_frame = test,
  stopping_metric = "AUC",
  nfolds = 10, seed = 123,
  max_runtime_secs = 480)

tryCatch({
  h2o_data <- h2o.splitFrame(data = h2o_data, ratios = 0.8, seed = 123)
}, error = function(e) {
  print(paste("Error occurred during data splitting:", e))
  # Handle the error or adjust parameters as needed
})




model@leaderboard %>% as.data.frame() %>% view()
model@leader

# Predicting the Test set results ----
pred<-model@leader %>% h2o.predict(test) %>% as.data.frame()


# Threshold / Cutoff ----  
model@leader %>% 
  h2o.performance(test) %>% 
  h2o.find_threshold_by_max_metric('f1')->threshold


# ----------------------------- Model evaluation -----------------------------

# Confusion Matrix----
model@leader %>%
  h2o.confusionMatrix(test) %>% 
  as.tibble() %>% 
  select("0","1") %>% 
  .[1:2,] %>% t() %>% 
  fourfoldplot(conf.level = 0,color=c("red","darkgreen"),
               main=paste("Accuracy=",
                          round(sum(diag(.))/sum(.)*100,1),"%"))

# Area Under Curve (AUC) ----
# threshold - proqnozları o ve 1 e cevirmek ucun secilmmis optimal limit xetdir
# precision - tp/(tp+fp)
# recall    - tp/(tp+fn)
model@leader %>% 
  h2o.performance(test) %>% 
  h2o.metric() %>% 
  select(threshold,precision,recall,tpr,fpr) %>% 
  add_column(tpr_r=runif(nrow(.),min=0.001,max=1)) %>% 
  mutate(fpr_r=tpr_r) %>% 
  arrange(tpr_r,fpr_r) -> deep_metrics

model@leader %>% 
  h2o.performance(test) %>% 
  h2o.auc() %>% round(2) -> auc

highchart() %>% 
  hc_add_series(deep_metrics, "scatter", hcaes(y=tpr,x=fpr), color='green', name='TPR') %>%
  hc_add_series(deep_metrics, "line", hcaes(y=tpr_r,x=fpr_r), color='red', name='Random Guess') %>% 
  hc_add_annotation(
    labels = list(
      point = list(xAxis=0,yAxis=0,x=0.3,y=0.6),
      text = glue('AUC = {enexpr(auc)}'))
  ) %>%
  hc_title(text = "ROC Curve") %>% 
  hc_subtitle(text = "Model is performing much better than random guessing") 





