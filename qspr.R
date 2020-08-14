#######################################
#
#                by S. Kaji
#                Aug. 2017
#
#######################################

## load necessary packages
## install them using the following lines if they do not exist in your system
#install.packages(c("gbm","Boruta","clue","coefplot","devtools","e1071","FNN","fastcluster","FactoMineR","FSelector","ggplot2","glmnet","kernlab","pls","PredPsych","Rtsne","randomForest"))
#install.packages("stringi", type="source")
#devtools::install_github("hoxo-m/pforeach")

source("regression_lib.R")

## variables which are not used for prediction
nonvar <- c(
  "ID","SMILES","Phases","rac_en","Melting","Melting_type",
  "Cmtype","Cm","Cptype","Cp","Amtype","Am","Aptype","Ap","Bmtype","Bm","Bptype","Bp","Smtype","Sm","Sptype","Sp","Nmtype","Nm","Nptype","Np","Dhmtype","Dhm","Dhptype","Dhp",
  "Clearing_type","Clearing","num_C","num_H","num_N","prohibited"
)
kelv = -273.15
## target value bounds for graph drawing
ymin <- 0
ymax <- 350

## 1) ############ Start here ##################################
## Set the working directory
#setwd("data")

## data file loading
## load from compressed R data file: you can skip to 2)
dat <- readRDS("desc_CN12.rds")

#dat <- readcsv("desc_sample.csv")
dat <- readcsv("d:/ml_res/desc_all.csv")
## add more data
#dat <- rbind(dat,readcsv("desc107773.csv")) 

###############################
### data preprocessing

###### data selection
## select by ID
#dat <- dat[!(dat$ID %in% 121104:121111),]

## extract molecules with a specified phase transition
#matched <- grep('Cr[[:digit:][:space:]\\.\\?]+C[[:digit:][:space:]\\.\\?]+A[[:digit:][:space:]\\.\\?]+N[[:digit:][:space:]\\.\\?]+is',dat$Phases)
#matched <- grep('[CB][[:digit:][:space:]\\*\\.\\?]+A[[:digit:][:space:]\\*\\.\\?]+[Ni]',dat$Phases)
#matched <- grep('[CB][[:digit:][:space:]\\*\\.\\?]+A',dat$Phases)
#matched <- grep('A[[:digit:][:space:]\\*\\.\\?]+[Ni]',dat$Phases)
#dat <- dat[matched,]

## select only molecules with #C+#N >= 12 (remove small and non-organic)
dat <- dat[(dat$num_C+dat$num_N)>=12,]


## select thoes do not contain prohibited atoms
#dat <- dat[dat$prohibited == 0,]

## remove chiral
#dat <- dat[dat$rac.en=="",] 

### add some factor variables
#dat[,rac_en:=as.factor(dat$rac_en)]
#LCexist=as.factor(apply(data.frame(dat$Cexist,dat$Aexist,dat$Nexist),1,max))

########## missing value handling
## Impute missing values
#library(missForest)
#dat <- missForest(dat)
## zero fill
#dat[is.na(dat)] <- 0

## remove rows with NA
#dat <- na.omit(dat)
## remove columns with NA
dat <- dat[, !apply(dat, 2, function(x) any(is.na(x)) )]
## remove columns with constant value
dat <- dat[, !apply(dat, 2, function(x) length(unique(x)) == 1 )]
## remove columns with zero variance
dat_sd <- apply(dat, 2, sd) 
dat <- dat[, (!is.na(dat_sd) & dat_sd>1e-24) | (colnames(dat) %in% nonvar)]

## scale for regression
dat <- data.frame(dat[,which(names(dat) %in% nonvar)],
                  scale(dat[,-which(names(dat) %in% nonvar)]))
## scale for classification
#dat <- data.frame(Name=dat$Name,ID=dat$ID,LCexist=as.factor(LCexist),Cexist=as.factor(dat$Cexist),
#                  Aexist=as.factor(dat$Aexist),Nexist=as.factor(dat$Nexist),
#                  scale(dat[,-which(names(dat) %in% nonvar)]))


####
## save data into a native R data format for faster loading for next time
#saveRDS(dat,"desc_CN12.rds")
#saveRDS(dat,"desc_all.rds")


#### 2) #############################################

## choose one variable to be predicted
#target <- "Clearing"
#dat <- dat[dat$Clearing_type==1,]

#target <- "Melting"
#dat <- dat[dat$Melting_type==1,]   # remove glass (Tg)

## for Nm, remove monotropic
target <- "Np"
dat <- dat[dat$Nm < dat$Np,]
#dat <- dat[dat$Nptype==2,]

#target <- "Ap"
#dat <- dat[dat$Am < dat$Ap,]

target <- "Cp"
dat <- dat[dat$Cm < dat$Cp,]

#target <- "Bp"
#dat <- dat[dat$Bm < dat$Bp,]

#target <- "Dhp"
#dat <- dat[dat$Dhm < dat$Dhp,]

## setting variables
prepare_prediction(target)

## save the preprocessed data to a csv file
#write.csv(file="PhCN-processed.csv", na.omit(dat), row.names = FALSE)

##############################################################################

## XGBoost regression
library(xgboost)
bst <- list()
folds <- 5
# model training
for(i in 1:folds){  ## cross validation
  testidx <- which(1:length(dat[,1])%%folds == (i%%folds))
  dtrain = xgb.DMatrix(data =  as.matrix(dat[-testidx,varcol]), label = dat[-testidx,targetcol])
  dtest = xgb.DMatrix(data =  as.matrix(dat[testidx,varcol]), label = dat[testidx,targetcol])
  watchlist = list(train=dtrain, test=dtest)
  bst[[i]] = xgb.train(data = dtrain, booster="gbtree",
                  max.depth = 6, 
                  eta = 0.05,  # learning rate
                  alpha = 1.0, # L1
                  lambda = 1.0, # L2
#                  nthread = 8, # automatically set
                  nround = 10000, 
                  colsample_bytree = 0.4, min_child_weight = 1.5, subsample = 0.8, gamma = 2,
                  watchlist = watchlist, 
                  early_stopping_rounds = 80,
                  print_every_n = 500,
                  ## for regression
#                  objective = "reg:squarederror", 
                  )
}

# prediction using the learned model
#bst <- readRDS(paste0(target,".rds"))
prediction <- list()
for(i in 1:folds){
  testidx <- which(1:length(dat[,1])%%folds == (i%%folds))
  dtest = xgb.DMatrix(data =  as.matrix(dat[testidx,varcol]), label = dat[testidx,targetcol])
  p <- predict(bst[[i]], dtest)
  t <- dat[testidx,target]
  prediction <- rbind(prediction,
                      data.frame(
                      ID=dat[testidx,"ID"], SMILES=dat[testidx,"SMILES"], Phases=dat[testidx,"Phases"],
                      pred=p, truth=t, error=p-t,
                      ratio=abs(p-t)/(t-kelv)))
}
# plot prediction vs truth
plotpred(data.frame(prediction$pred,prediction$truth),paste0("XGBoost:",target),sort=T)
# error ratio
p <- prediction$ratio
plotsort( p )
quantile(p,c(0.05,0.1,0.3,0.5,0.7,0.9,0.95))
hist(pmax( 0, pmin( p, 0.2)), freq=F, main="Histogram", xlab="error ratio")

## prediction results will be written to a csv: open it with, e.g., excel to see the results
write.csv(file=paste0("pred_",target,".csv"), prediction, row.names = FALSE)
saveRDS(bst,paste0(target,".rds"))

### Variable importance
importance <- xgb.importance(model = bst[[1]])
head(importance,20)
xgb.plot.importance(head(importance,20))


###################################
### classification
dat <- readRDS("desc_CN12.rds")
target <- "Cptype"
prepare_prediction(target)

bst <- list()
folds <- 5
for(i in 1:folds){  ## cross validation
  testidx <- which(1:length(dat[,1])%%folds == (i%%folds))
  dtrain = xgb.DMatrix(data =  as.matrix(dat[-testidx,varcol]), label = dat[-testidx,targetcol])
  dtest = xgb.DMatrix(data =  as.matrix(dat[testidx,varcol]), label = dat[testidx,targetcol])
  watchlist = list(train=dtrain, test=dtest)
  bst[[i]] = xgb.train(data = dtrain, booster="gbtree",
                       max.depth = 6, 
                       eta = 0.05,
                       alpha = 1.0,
                       lambda = 1.0,
                       #                  nthread = 8, # automatically set
                       nround = 10000, 
                       colsample_bytree = 0.4, min_child_weight = 1.5, subsample = 0.8, gamma = 2,
                       watchlist = watchlist, 
                       early_stopping_rounds = 80,
                       print_every_n = 500,
                       ## for classification
                                         objective = "multi:softprob",
                                         eval_metric = "mlogloss",
                                         num_class = 3 # may need to be changed
  )
}
# prediction
prediction <- list()
for(i in 1:folds){
  testidx <- which(1:length(dat[,1])%%folds == (i%%folds))
  dtest = xgb.DMatrix(data =  as.matrix(dat[testidx,varcol]), label = dat[testidx,targetcol])
  p <- predict(bst[[i]], dtest,reshape=T)
  t <- dat[testidx,target]
  prediction <- rbind(prediction,
                      data.frame(
                        ID=dat[testidx,"ID"], SMILES=dat[testidx,"SMILES"], Phases=dat[testidx,"Phases"],
                        pred=max.col(p, ties.method = "last")-1,
                        prob0=p[,1],prob1=p[,2],prob2=p[,3],
                        truth=t))
}
# types: 0 means "not exist", 1 means "exist", 2 means "(*)"; e.g., N* means cholesteric
confusionMatrix(factor(prediction$pred),
                factor(prediction$truth),
                mode = "everything")
write.csv(file=paste0("clas_",target,".csv"), prediction, row.names = FALSE)
saveRDS(bst,paste0(target,".rds"))

### Variable importance
importance <- xgb.importance(model = bst[[1]])
head(importance,20)
xgb.plot.importance(head(importance,20))



################
##https://github.com/microsoft/LightGBM/tree/master/R-package
#PKG_URL <- "https://github.com/microsoft/LightGBM/releases/download/v3.0.0rc1/lightgbm-3.0.0-1-r-cran.tar.gz"
#remotes::install_url(PKG_URL, INSTALL_OPTS = "--no-multiarch")

## binary
#PKG_URL <- "https://github.com/microsoft/LightGBM/releases/download/v3.0.0rc1/lightgbm-3.0.0-1-r40-windows.zip"
#local_file <- paste0("lightgbm.", tools::file_ext(PKG_URL))
#download.file(  url = PKG_URL  , destfile = local_file)
#install.packages(  pkgs = local_file  , type = "binary"  , repos = NULL)

library(lightgbm)

dat <- readRDS("desc_CN12.rds")
target <- "Np"
prepare_prediction(target)
dat <- dat[dat$Nm < dat$Np,]

folds <- 5
bst <- list()
for(i in 1:folds){  ## cross validation
  testidx <- which(1:length(dat[,1])%%folds == (i%%folds))
  dtrain <- lgb.Dataset(as.matrix(dat[-testidx,varcol]), label = dat[-testidx,targetcol])
  dtest <- lgb.Dataset.create.valid(dtrain,as.matrix(dat[testidx,varcol]), label = dat[testidx,targetcol])
  params <- list(objective="regression", metric="l2",lambda_l1 = 1,lambda_l2 = 1,max_depth = 5)
  #bst[[i]] <- lgb.cv(params, dtrain, 10, nfold=5, min_data=1, learning_rate=1, early_stopping_rounds=80)
  bst[[i]] <- lgb.train(params, dtrain, 
#                   device_type="gpu", 
                   seed = 42,force_col_wise=T,
                   num_leaves=2^3,  # should be less than 2^max_depth
                   colsample_bytree = 0.4, min_child_weight = 1.5, subsample = 0.8,
                   nrounds=10000, eval_freq=500, min_data=1, learning_rate=0.1, 
                   early_stopping_rounds=80, valids=list(test=dtest))
}
prediction <- list()
for(i in 1:folds){
  testidx <- which(1:length(dat[,1])%%folds == (i%%folds))
  p <- predict(bst[[i]], as.matrix(dat[testidx,varcol]),reshape=T)
  t <- dat[testidx,target]
  prediction <- rbind(prediction,
                      data.frame(
                        ID=dat[testidx,"ID"], SMILES=dat[testidx,"SMILES"], Phases=dat[testidx,"Phases"],
                        pred=p, truth=t, error=p-t,
                        ratio=abs(p-t)/(t-kelv)))
}

# plot prediction vs truth
plotpred(data.frame(prediction$pred,prediction$truth),paste0("lighGBM:",target),sort=T)
# error ratio
p <- prediction$ratio
plotsort( p )
quantile(p,c(0.05,0.1,0.5,0.9,0.95))
hist(pmax( 0, pmin( p, 0.3)), freq=F, main="Histogram", xlab="error ratio")

## prediction results will be written to a csv: open it with, e.g., excel to see the results
write.csv(file=paste0("pred_lgb_",target,".csv"), prediction, row.names = FALSE)
importance <- lgb.importance(model = bst[[2]])
lgb.plot.importance(
  importance, top_n = 10L, measure = "Gain", left_margin = 10L, cex = NULL
)



######
library(data.table)
grid_search <- expand.grid(Depth = 2:8,L1 = 0:5,L2 = 0:5)
model <- list()
perf <- numeric(nrow(grid_search))
for (i in 1:nrow(grid_search)) {
  model[[i]] <- lgb.train(list(objective = "regression",
                               metric = "l2",
                               lambda_l1 = grid_search[i, "L1"],
                               lambda_l2 = grid_search[i, "L2"],
                               max_depth = grid_search[i, "Depth"]),
                          dtrain,
                          num_leaves=2^3,
#                          device_type="gpu", 
                          seed = 42,
                          colsample_bytree = 0.4, min_child_weight = 1.5, subsample = 0.8,
                          nrounds=10000, eval_freq=200, min_data=1, learning_rate=0.1, 
                          early_stopping_rounds=80, valids=list(test=dtest))
  perf[i] <- min(rbindlist(model[[i]]$record_evals$test$l2))
}
cat("Model ", which.min(perf), " is lowest loss: ", min(perf), sep = "","\n")
print(grid_search[which.min(perf), ])


#################
# I have only fixed up to here.
# TODO: fix below
#################

#######################################################################
testidx <- grep('B[[:digit:][:space:]\\*\\.\\?]+A',dat$Phases)
testidx <- grep('A[[:digit:][:space:]\\*\\.\\?]+i',dat$Phases)

#############
## hyper-parameter search
library(doParallel)
registerDoParallel(makePSOCKcluster(4))
set.seed(0)

modelLookup("xgbTree")
xgb_grid_1 = expand.grid(
  nrounds = 2000, 
#  lambda = c(0.0,0.1,0.01),
#  alpha = c(2.0,0.2,0.02),
  eta = c(0.05,0.01),
  max_depth = c(5, 10, 15), 
  gamma = c(1, 2, 3), 
  colsample_bytree = c(0.4, 0.7, 1.0), 
  min_child_weight = c(0.5, 1, 1.5),
  subsample = c(.8,1)
  )
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 3,
  returnResamp = "all",
  allowParallel = T,
)

model <- train(f,data=dat,
  #x=as.matrix(apply(dat[,varcol],2,as.double)),y=as.double(dat[,targetcol]), 
               trControl=xgb_trcontrol_1, 
               tuneGrid = xgb_grid_1, 
               #               tuneLength = 4,
#               preProcess = c('center', 'scale'),
               #  objective = "reg:linear", 
               #  nthread = 4,
               #  early_stopping_rounds = 50,
               #  print_every_n = 500,
               method = 'xgbTree', verbose=T)
print(model)

#model <- readRDS("carret_tuning.rds")
#write.csv(file="tuning.csv", model$results)


##############################################
## Random Forest
library(randomForest)
prediction <- list()
for(i in 1:5){
  testidx <- which(1:length(dat[,1])%%5 == (i%%5))
  #model <- randomForest(f, data=dat[-testidx,], ntree=500)
  model <- pforeach(.cores=4, ntree = rep(200, .cores), .c = randomForest::combine)({
    randomForest(f, data=dat[-testidx,], ntree=ntree) #,importance=T)
  })
  prediction <- rbind(prediction,data.frame(pred=predict(model, dat[testidx,]),true=dat[testidx,targetcol]))
}
sortlist <- order(abs(prediction$true-prediction$pred),decreasing=T)
prediction <- prediction[sortlist,]
#write.csv(file=paste0("pred-PhCN1_",target,".csv"), bad, row.names = FALSE)
ymin=0
ymax=2
plotpred(prediction,"randomForest",sort=F)

print(model)
prediction <- predict(model, dat[testidx,])
## for regression
plotpred(data.frame(dat[testidx,targetcol],prediction),"RandomForest",sort=T)
## for classification
#plottab(prediction,testdat[,targetcol])
#cm <- caret::confusionMatrix(prediction,testdat[,targetcol])
#cm$byClass
## importance
#importance(model)
varImpPlot(model, main=paste0("Var importance for ",target))

## error
ymin <- -4
ymax <- 4
diff=data.frame(dat[testidx,targetcol]-kelv)/(prediction-kelv)-1
plotpred(diff)
library(vioplot)
summary(diff)
vioplot(diff)
library(fitdistrplus)
hist(diff,breaks=100)
plot(ecdf(diff))

fit<-fitdist(diff,"norm")
plot(fit,breaks=100)
summary(fit)


## Gaussian Process: TOO SLOW USE scikit
library(GauPro)
gp <- GauPro(as.matrix(dat[-testidx,varcol]), dat[-testidx,targetcol], parallel=T)
prediction <- gp$prediction(dat[testidx,varcol])


## PLS
library(pls)
library(plsRglm)
ncomp=30
bst <- list()
prediction <- list()
folds <- 9
for(i in 1:folds){
  testidx <- which(1:length(dat[,1])%%folds == (i%%folds))
  bst[[i]] <- plsr(f, ncomp=ncomp, data=dat[-testidx,])
#plsmodel <- plsRglm(f, data=dat[-testidx,],modele="pls-glm-gaussian") 
#prediction <- predict(plsmodel, newdata=dat[testidx,],type="response")
#summary(plsmodel)
#plot(plsmodel, ncomp=ncomp, asp=1, line=TRUE)
#plot(plsmodel, "loadings", comps=1:2, legendpos = "topleft")
#plotpred(data.frame(dat[testidx,targetcol],prediction),"PLS Regression",sort=T)
#write.csv(file="rac-bad.csv", bad, row.names = FALSE)
  p <- predict(bst[[i]], newdata=dat[testidx,])[,1,bst[[i]]$ncomp]
  t <- dat[testidx,target]
  prediction <- rbind(prediction,
                      data.frame(
                        ID=dat[testidx,"ID"], SMILES=dat[testidx,"SMILES"], Phases=dat[testidx,"Phases"],
                        pred=p, truth=t, error=p-t,
                        ratio=abs(p-t)/(t-kelv)))
}
write.csv(file=paste0("pred-pls-cv9_",target,".csv"), prediction, row.names = FALSE)


## SVM
## grid search for parameters
#tun=tune.svm(f, data=dat, gamma=5^(seq(-10, -2, 2)), cost=3^(seq(0, 3, 1)),tunecontrol=tune.control(sampling="cross", cross=5))
#plot(tun, transform.x=log10, transform.y=log10)
#prediction <- predict(tun$best.model, testdat)
#summary(tun)
## model fitting
model <- svm(f, data=dat[-testidx,], kernel = "radial", cost=9.0, gamma = 0.001)
# cross validation
model <- svm(f, data=dat, cost=9.0, gamma = 0.001, cross=5)
#plot(model, dat, formula=Q1~A19)
summary(model)
prediction <- predict(model, testdat)
plotpred(prediction,testdat[,targetcol],"SVR",sort=F)
plottab(prediction,testdat[,targetcol])

## kNN regression
library(FNN)
k = 5
prediction <- knn.reg(dat[-testidx,varcol], test = dat[testidx,varcol], y=dat[-testidx,targetcol], k = k, algorithm=c("kd_tree", "cover_tree", "brute"))
plotpred(data.frame(prediction$pred,dat[testidx,targetcol]),"KNN",sort=F)

## kNN classification
library(class)
prediction <- class::knn(train=dat[-testidx,varcol], 
                         test=dat[testidx,varcol], 
                         cl=dat[-testidx,targetcol], k=3)
plottab(prediction,testdat[,targetcol])


## (generalised) linear regression
require(coefplot)
model <- lm(f, data=dat[-testidx,], na.action = na.exclude)
model <- glm(f, data=dat[-testidx,], family = gaussian)
summary(model)
anova(model)
prediction <- predict(model, newdata=dat[testidx,])
coefplot(model)
plotpred(prediction,dat[testidx,targetcol], "Linear Regression",sort=T)
library(semPlot)
semPaths(model,what="std",layout="circle")

## regularised regression
library(glmnet)
model <- cv.glmnet(as.matrix(dat[-testidx,varcol]), as.vector(dat[-testidx,targetcol]), family="gaussian")
prediction <- predict(model, newx=as.matrix(dat[testidx,varcol]))
#plot(gmodel)
#coefplot(gmodel, lambda='lambda.1se')
plotpred(prediction,dat[testidx,targetcol],"Regularized Regression",sort=T)

## Baysian classification
model <- naiveBayes(f, data=traindat, laplace=0)
prediction <- predict(model, testdat, type="class")
model
plottab(prediction,testdat[,targetcol])



#############################################################
## PCA visualisation
#dat <- dat[1:15,]
library(ggrepel)
require(fastcluster) 
library(FactoMineR)
pca <- PCA(dat[,varcol],ncp=10,graph = FALSE,scale.unit=T)
pcap <- data.frame(pca$ind$coord)
col = as.factor(dat[,targetcol])
col = dat[,targetcol]
col = rep(F,length(dat[,1])); col[matched]=T
l = sqrt(sapply(dat$SMILES,nchar))
g <- ggplot(data=pcap, aes(x=pcap[,1],y=pcap[,2],color=col))+geom_point(size=I(2))
#xlim=c(-100,300),ylim=c(-50,200),  #l
g <- g+labs(main=target,xlab="primary component",ylab="sqrt of len(SMILES)") 
g <- g + geom_text_repel(aes(label=dat$SMILES),size=2)
#g <- g +scale_color_gradient(low="blue", high="red")
#g <- g + geom_point(data=data.frame(x=pcap[matched,1],y=pcap[matched,2]),aes(x,y,color=col[matched]),size=I(0.8))
print(g)



## visualisation
x="AMID"
y="piPC9"
qplot(dat[[x]],dat[[y]], 
      main=target,xlab=x,ylab=y,
      color=col,size=I(0.8)) +scale_color_gradient(name=target,low="blue", high="red")

# umap
summary(crd$layout)
crd <- umap(dat[,varcol])
qplot(crd$layout[,1],crd$layout[,2], 
      color=col,size=I(0.8))  # +scale_color_gradient(name=target,low="blue", high="red")


# t-SNE
library(Rtsne)
pplex <- 3
tsne <- Rtsne(as.matrix(dat[,varcol]), pca=TRUE, check_duplicates = FALSE, verbose=TRUE, max_iter = 1000, perplexity = pplex)
qplot(tsne$Y[,1], tsne$Y[,2], color=dat[,targetcol],size=I(0.8))+scale_color_gradient(low="blue", high="red")

## clustering
library(cluster)
library(clue)
#ward
cluster = hclust(dist(dat[,varcol]), method="ward.D")
plot(cluster)
rect.hclust(cluster, k=5, border="red")
#kmeans
cluster = kmeans(dist(dat[,varcol]), centers = 5)
clusplot(dat[,varcol], cluster$cluster, col.p=dat[,targetcol], shade=TRUE, labels=0, lines=1)
#cmeans
cluster = cmeans(dist(dat[,varcol]), centers = 5)
clusplot(dat[,varcol], cluster$cluster, col.p=dat[,targetcol], shade=TRUE, labels=0, lines=1)

## plot from the output of chainer code
res <- read.csv("result_PH1.txt", header=F, sep = ",", stringsAsFactors = FALSE)
target=0
ymin = 0
ymax = 0.4
plotpred(data.frame(res[2],res[3]),"H1",sort=F)
ymin = 1.2
ymax = 2.5
plotpred(data.frame(res[2],res[3]),"H0",sort=T)

## feature selection
library(Boruta)
set.seed(111)
boruta.bank <- Boruta(f, data = dat, doTrace = 2)
#saveRDS(boruta.bank, "boruta10000.rds")
boruta.bank <- readRDS("boruta10000.rds")
print(boruta.bank)

plot(boruta.bank, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.bank$ImpHistory),function(i)
  boruta.bank$ImpHistory[is.finite(boruta.bank$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.bank$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta.bank$ImpHistory), cex.axis = 0.7)

## select important variables
selected <- c(rep(T,length(nonvar)),(attStats(boruta.bank)[[1]]>6))
sdat <- dat[,selected]
dat <- sdat

## PredPsych
#dyn.load("/Library/Java/JavaVirtualMachines/jdk1.8.0_31.jdk/Contents/Home/jre/lib/server/libjvm.dylib")
library(FSelector)
library(PredPsych)
weights <- chi.squared(f,dat)
weights <- information.gain(f, dat)
print(weights)
#fscore(dat,classCol = 1, featureCol = c(2:83))
# select 5 most important vars
selected_features <- cutoff.k(weights, 20)
f <- as.simple.formula(selected_features, target)

## t-test
ptemp <- readcsv("pred-CAN-Nptemp.csv")
mtemp <- readcsv("pred-CAN-Nmtemp.csv")
#write.csv(file=paste0("pred-PhCN1_",target,".csv"), bad, row.names = FALSE)
errtarget="ratio"
err<-data.frame(ptemp=abs(ptemp[[errtarget]]),mtemp=abs(mtemp[[errtarget]]))
sortlist <- order(err$ptemp,decreasing=T)
err <- err[sortlist,]
ymax <- 0.5
plotpred(err,"error ratio",sort=F)
wilcox.test(err$ptemp,err$mtemp, paired = TRUE, alternative = "less")

plotpred(data.frame(ptemp$truth,ptemp$pred),"C",sort=T)
plotpred(data.frame(mtemp$truth,mtemp$pred),"C",sort=T)

res<-read.csv("pred-BAN_Atemp.csv")
head(res)
x=res$pred_p-res$truth_p
y=res$pred_m-res$truth_m
x=abs(res$ratio_p)
y=abs(res$ratio_m)
x=res$truth_p
y=res$truth_m
x=res$pred_p
y=res$pred_m
plot(x=x,y=y)
ymin=-0
ymax=0.3
plotpred(data.frame(x,y),"C",sort=T)

plotpred(data.frame(x),"C",sort=T)

#
ptemp <- readcsv("result.csv",header=F)
ymin=0
ymax=10
target=""
summary(ptemp$V2)
plotpred(data.frame(true=ptemp$V2,pred=ptemp$V3),title="H_0",sort=F)

summary(ptemp$V4)
ymin=0
ymax=2
plotpred(data.frame(true=ptemp$V4,pred=ptemp$V5),"H_1",sort=F)


library(vioplot)
vioplot(ptemp[[errtarget]],mtemp[[errtarget]],names=c("above","below"))


## rcdk
library(rcdk)
mols <- load.molecules( c('test.smi') )
view.molecule.2d(mols[[1]])
descNames <- unique(unlist(sapply(get.desc.categories(), get.desc.names)))
descs <- eval.desc(mols, descNames)
## save file
#write.molecules(mols, filename='mymols.sdf')
