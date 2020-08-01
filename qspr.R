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

############## Start here ##################################
## Set the working directory
#setwd("data")

#### data preprocessing

## data file loading
dat <- readcsv("desc_sample.csv")
#dat <- readcsv("desc_010000.csv")
## add more data
#dat <- rbind(dat,readcsv("desc107773.csv")) 

## load from compressed R data file
#dat <- readRDS("formatted/chem_all_clean.rds")
## select a part of the dataset
#dat <- dat[!(dat$ID %in% 121104:121111),]

## extract molecules with a specified phase transition
#matched <- grep('Cr[[:digit:][:space:]\\.\\?]+C[[:digit:][:space:]\\.\\?]+A[[:digit:][:space:]\\.\\?]+N[[:digit:][:space:]\\.\\?]+is',dat$Phases)
#matched <- grep('[CB][[:digit:][:space:]\\*\\.\\?]+A[[:digit:][:space:]\\*\\.\\?]+[Ni]',dat$Phases)
#matched <- grep('[CB][[:digit:][:space:]\\*\\.\\?]+A',dat$Phases)
#matched <- grep('A[[:digit:][:space:]\\*\\.\\?]+[Ni]',dat$Phases)
#dat <- dat[matched,]

## select thoes do not contain prohibited atoms
#dat <- dat[dat$prohibited == 0,]

## Impute missing values
#library(missForest)
#dat <- missForest(dat)
## zero fill
#dat[is.na(dat)] <- 0

#dat[,rac_en:=as.factor(dat$rac_en)]
#LCexist=as.factor(apply(data.frame(dat$Cexist,dat$Aexist,dat$Nexist),1,max))

## save preprocessed data into a native R data format for faster loading for next time
#saveRDS(dat,"chem_all_20190508.rds")

#################################################

## variables which are not used for prediction
nonvar <- c(
  "ID","SMILES","Phases","rac_en","Melting","Cmtype","Cm","Cptype","Cp","Amtype","Am","Aptype","Ap","Smtype","Sm","Sptype","Sp","Nmtype","Nm","Nptype","Np","Clearing_type","Clearing","unknownX"
)
kelv = -273.15

### cleaning
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


## choose one variable to be predicted
target <- "Clearing"
#target <- "Melting"
#target <- "Np"
#target <- "Am"

## setting variables
prepare_prediction(target)


## remove rows where the target attribute is absent
dat <- dat[!is.na(dat[[target]]),]

## remove chiral
#dat <- dat[dat$rac.en=="",] 

# remove rows with target < kelv
dat <- dat[dat[[target]] >= kelv,] 
# remove rows with target = 0 as they are not reliable
dat <- dat[dat[[target]]!=0,] 

# select rows with a specified type
#dat <- dat[dat[["Cmtype"]]==1,] 

## save to a csv file
#write.csv(file="PhCN-processed.csv", na.omit(dat), row.names = FALSE)

##############################################################################

## target value bounds for graph drawing
ymin <- 0
ymax <- 350

## XGBoost regression
library(xgboost)
bst <- list()
folds <- 4
# model training
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
                  ## for regression
#                  objective = "reg:squarederror", 
                  ## for classification
#                  objective = "multi:softprob",
#                  eval_metric = "mlogloss",
#                  num_class = 3
                  )
}

# prediction using the learned model
#bst <- readRDS(paste0(target,".rds"))
prediction <- list()
for(i in 1:folds){
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
quantile(p,c(0.05,0.1,0.5,0.9,0.95))
hist(pmax( 0, pmin( p, 0.3)), freq=F, main="Histogram", xlab="error ratio")

## prediction results will be written to a csv: open it with, e.g., excel to see the results
write.csv(file=paste0("pred_",target,".csv"), prediction, row.names = FALSE)
saveRDS(bst,paste0(target,".rds"))

### Variable importance
importance <- xgb.importance(model = bst[[1]])
head(importance,20)
xgb.plot.importance(head(importance,20))


###################################
### classification
target <- "Nmtype"
prepare_prediction(target)

bst <- list()
folds <- 4
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
                                         num_class = 3
  )
}
# prediction
prediction <- list()
for(i in 1:folds){
  p <- predict(bst[[i]], dtest,reshape=T)
  t <- dat[testidx,target]
  prediction <- rbind(prediction,
                      data.frame(
                        ID=dat[testidx,"ID"], SMILES=dat[testidx,"SMILES"], Phases=dat[testidx,"Phases"],
                        pred=max.col(p, ties.method = "last"),
                        prob1=p[,1],prob2=p[,2],prob3=p[,3],
                        truth=t))
}
# types: 0 means "not exist", 1 means "exist", 2 means "(*)"
confusionMatrix(factor(prediction$pred),
                factor(prediction$truth+1),
                mode = "everything")
write.csv(file=paste0("clas_",target,".csv"), prediction, row.names = FALSE)


#################
# I have only fixed up to here.
# TODO: fix below
#################

#######################################################################
testidx <- grep('B[[:digit:][:space:]\\*\\.\\?]+A',dat$Phases)
testidx <- grep('A[[:digit:][:space:]\\*\\.\\?]+i',dat$Phases)
dtrain = xgb.DMatrix(data =  as.matrix(dat[-testidx,varcol]), label = dat[-testidx,targetcol])
dtest = xgb.DMatrix(data =  as.matrix(dat[testidx,varcol]), label = dat[testidx,targetcol])
watchlist = list(train=dtrain, test=dtest)
bst = xgb.train(data = dtrain, booster="gbtree",
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
                     ## for regression
                     objective = "reg:linear", 
                     ## for classification
                     #                  objective = "multi:softprob",
                     #                  eval_metric = "mlogloss",
                     #                  num_class = 3
)
p <- predict(bst, dtest)
t <- dat[testidx,target]
prediction <- data.frame(
                      ID=dat[testidx,"ID"], SMILES=dat[testidx,"SMILES"], Phases=dat[testidx,"Phases"],
                      pred=p, truth=t, error=p-t,
                      ratio=(p-t)/(t-kelv))
plotpred(data.frame(prediction$pred,prediction$truth),paste0("XGBoost:",target),sort=T)

#############
## CV
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

## Random Forest
library(randomForest)
model <- randomForest(f, data=dat, ntree=1000) #,importance=T)

##CV
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


## t-test
setwd("~/Dropbox/tex/Liquid Crystal/analysis/20181126/analysis")
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
ptemp <- readcsv("~/Downloads/result.csv",header=F)
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

## rcdk
library(rcdk)
mols <- load.molecules( c('test.smi') )
view.molecule.2d(mols[[1]])
descNames <- unique(unlist(sapply(get.desc.categories(), get.desc.names)))
descs <- eval.desc(mols, descNames)
## save file
#write.molecules(mols, filename='mymols.sdf')
