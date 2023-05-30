#######################################
#
#                by S. Kaji
#                Aug. 2017
#               (last updated 11 Sep. 2020)
#######################################

## load necessary packages

###########################
source("regression_lib.R")

## Set the working directory
#setwd("data")

## set the following appropriately
descriptor_csv <- "desc_sample.csv"
preprocessed_rds <- "d:/ml_res/desc_all.rds"

## 1) ############ preprocessing ##################################
## if you already performed preprocessing and 
## have the saved rds file, you can skip to 2)

## load descriptors
dat <- readcsv(descriptor_csv)
#dat <- readRDS("d:/ml_res/desc_all.rds")
## add more data
#dat$group=1
#dat <- rbind(dat,data.frame(readcsv("NR_en.csv"),group=2)) 

###### data selection
## select by ID
#dat <- dat[!(dat$ID %in% 121104:121111),]

## extract molecules with a specified phase transition
#matched <- grep('Cr[[:digit:][:space:]\\.\\?]+C[[:digit:][:space:]\\.\\?]+A[[:digit:][:space:]\\.\\?]+N[[:digit:][:space:]\\.\\?]+is',dat$Phases)
#matched <- grep('[CB][[:digit:][:space:]\\*\\.\\?]+A[[:digit:][:space:]\\*\\.\\?]+[Ni]',dat$Phases)
#matched <- grep('[CB][[:digit:][:space:]\\*\\.\\?]+A',dat$Phases)
#matched <- grep('A[[:digit:][:space:]\\*\\.\\?]+[Ni]',dat$Phases)
#dat <- dat[matched,]

## check SMILES and formula coincides
dat <- dat[(dat$num_C == dat$nC) & (dat$num_N == dat$nN) & (dat$num_H == dat$nH),]

## select only molecules with #C+#N >= 12 (remove small and non-organic)
dat <- dat[(dat$num_C+dat$num_N)>=12,]

## select those do not contain prohibited atoms
#dat <- dat[dat$prohibited == 0,]

### add some factor variables
#dat[,rac_en:=as.factor(dat$rac_en)]

########## missing value handling
## Impute missing values
#library(missForest)
#dat <- missForest(dat)
## zero fill
#dat[is.na(dat)] <- 0

## remove rows with NA
#dat <- na.omit(dat)
## remove columns with NA
#dat <- dat[, !apply(dat, 2, function(x) any(is.na(x)) )]
## remove columns with constant value
#dat <- dat[, !apply(dat, 2, function(x) length(unique(x)) == 1 )]
## remove columns with zero variance
#dat_sd <- apply(dat, 2, sd) 
#dat <- dat[, (!is.na(dat_sd) & dat_sd>1e-24) | (colnames(dat) %in% nonvar)]

## scaling data
dat <- data.frame(dat[,which(names(dat) %in% nonvar)],scale(dat[,-which(names(dat) %in% nonvar)]))

####
## save data into a native R data format for faster loading for next time
saveRDS(dat,preprocessed_rds)


#### 2) ###### prediction with lightGBM ###################

library(lightgbm)
packageVersion("lightgbm")

# load preprocessed data file
master_dat <- readRDS(preprocessed_rds)

## choose one variable to be predicted
#prepare_prediction("Clearing")

#dat <- prepare_prediction("Melting", only_type=1) # remove glass (Tg)

dat <- prepare_prediction("Cp",remove_monotropic = T, only_type=2)

dat <- prepare_prediction("Dm",remove_monotropic = T)

dat <- prepare_prediction("Dtype")


folds <- 5
#bst <- readRDS(paste0("result/lgb_",target,".rds"))

if(is_regression){
  params <- list(objective="regression", metric="l2",lambda_l1 = 1,lambda_l2 = 1,max_depth = 5)
}else{
  n_class <- length(unique(dat[[target]]))
  if(n_class==2){
    params <- list(objective="binary", is_unbalance=T,lambda_l1 = 1,lambda_l2 = 1,max_depth = 5)
  }else{
    params <- list(objective="multiclass", metric="multi_logloss",num_class=n_class,lambda_l1 = 1,lambda_l2 = 1,max_depth = 5)
  }
}
for(i in 1:folds){  ## cross validation
  if(i==1){
    bst <- list()
  }
  testidx <- which(1:length(dat[,1])%%folds == (i%%folds))
#  testidx <- which(dat$Ctype==1)  
  dtrain <- lgb.Dataset(as.matrix(dat[-testidx,varcol]), label = dat[-testidx,targetcol])
  dtest <- lgb.Dataset.create.valid(dtrain,as.matrix(dat[testidx,varcol]), label = dat[testidx,targetcol])
  #bst[[i]] <- lgb.cv(params, dtrain, 10, nfold=5, min_data=1, learning_rate=1, early_stopping_rounds=80)
  bst[[i]] <- lgb.train(params, dtrain, 
#                   device_type="gpu", 
                   seed = 42,force_col_wise=T,
                   num_leaves=2^4,  # should be less than 2^max_depth
                   colsample_bytree = 0.4, min_child_weight = 1.5, subsample = 0.8,
                   nrounds=20000, eval_freq=500, min_data=1, learning_rate=0.1, 
                   early_stopping_rounds=1000,   # set to 100 for quick test
                   valids=list(test=dtest))
}
saveRDS(bst,paste0("result/lgb_",target,".rds"))

#regression_summary(dat[testidx,target],predict(bst[[1]], as.matrix(dat[testidx,varcol]),reshape=T))

prediction <- prediction_result(dat,bst,fold)
prediction_plot(prediction, plot_value=TRUE, plot_ROC=TRUE, plot_hist=TRUE)
importance <- importance_csv(bst, file=paste0("importance_",target,".csv"))


########### interpretation of the model via variable importance
## correlation between importance of variables
cc <- c("Cm","Cp","Ctype2","Ctype3")
#cc <- c("Nm","Np","Ntype2","Ntype3")
idx <- list()
for(i in 1:length(cc)){
  q <- read.csv(paste0("importance_",cc[[i]],".csv"), header=TRUE, sep = ",", stringsAsFactors = FALSE)
  if(i==1){
    imp <- q
  }else{
    imp <- full_join(imp,q, by="Feature")
  }
  s = 15*(i-1)
  idx[[i]] <- c(s+2,s+5,s+8,s+11,s+14)
}

cors <- list()
for(k in 1:length(cc)){
  for(j in k:length(cc)){
    cs <- c()
    for(i in idx[[j]]){
      cs <- c(cs,apply(imp[,idx[[k]]],2,cor,imp[[i]],use="complete.obs"))
    }
    names(cs) <- NULL
    cors[[paste0(cc[[k]],cc[[j]])]] <- cs
  }
}
png(paste0("cor_importance_C.png"), width = 1200, height = 600)
boxplot(cors)
dev.off()


## correlation of mean variable importance
cc <- c("Np","Nm","Ntype2","Ntype3","Cp","Cm","Ctype2","Ctype3","Ap","Am","Atype","Bp","Bm","Btype","Clearing","Melting") # "Dp","Dm","Dtype",
rdir = "/Users/kaji/Dropbox/shared/Liquid Crystal/analysis/latest_analysis/importance/"
for(i in 1:length(cc)){
  q <- read.csv(paste0(rdir,"importance_lgb_",cc[[i]],".csv"), header=TRUE, sep = ",", stringsAsFactors = FALSE)
  if(i==1){
    imp <- q
  }else{
    imp <- full_join(imp,q, by="Feature")
  }
}

## intra models
cor_imp = list()
for(i in 1:length(cc)){
  j = 15*(i-1)
  u = c(j+2,j+5,j+8,j+11,j+14)
  ci = cor(imp[,u], use="pairwise.complete.obs")
  cor_imp = rbind(cor_imp, data.frame(x=cc[[i]], y=c(ci[1,2],ci[1,3],ci[1,4],ci[1,5],ci[2,3],ci[2,4],ci[2,5],ci[3,4],ci[3,5],ci[4,5])))
}
g = ggplot(cor_imp,aes(x=x,y=y))+geom_violin()
ggsave(paste0("importance_intra_cor.png"),g)
summary(cor_imp$y)
sd(cor_imp$y)

## inter models
mean_imp = list()
for(i in 1:length(cc)){
  j = 15*(i-1)
  u = c(j+2,j+5,j+8,j+11,j+14)
  mean_imp[[cc[[i]]]] = apply(imp[,u],1,mean)
}
mean_imp <- data.frame(mean_imp)

target <- 'Nm'
il <- data.table(Feature=imp$Feature, Gain=mean_imp[[target]])
il <- head(il[order(il$Gain, decreasing =TRUE),],15)
g <- ggplot(il, aes(x = reorder(Feature, Gain), y = Gain, fill = 0))
g <- g + geom_bar(stat = "identity")+ coord_flip() +theme(legend.position = 'none') + theme(axis.title.x = element_blank(), axis.title.y = element_blank()) 
plot(g)
ggsave(paste0("importance_lgb_",target,".png"))

library(corrplot)
library(reshape2)
cormat <- round(cor(mean_imp, use="pairwise.complete.obs"),3)
png(paste0("cor_imp_elip.png"), width = 1024, height = 600)
corrplot(cormat, method="ellipse",tl.col="black",order="original")
dev.off()
melted_cormat <- melt(cormat)
png(paste0("cor_imp_col.png"), width = 1024, height = 600)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value))+geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()+labs(x="",y="")
dev.off()

########### error analysis ####################
library(openxlsx)
library(exactRankTests)
preds <- read.xlsx("prediction.xlsx")
#preds <- fread("pred_Np_relgcn.csv")

pred1 <- fread("result_old/pred_lgb_Np.csv")
pred2 <- fread("result_old/pred_lgb_Nm.csv")
regression_summary(pred1$truth,pred1$pred)
regression_summary(pred2$truth,pred2$pred)

#preds$ratio_Np <- abs(preds$pred_lgb-preds$truth)/(preds$truth-kelv)

## ratio of two sets of prediction errors
preds <- data.frame(r=pred1$ratio/pred2$ratio)
ggplot(preds,aes(x=c(0),y=r))+geom_violin()+scale_y_continuous(breaks = c(0.1,0.5,1,2,10), trans = "log10", limits=c(0.1,10))

preds <- data.frame(r=abs(pred1$ratio)-abs(pred2$ratio))
ggplot(preds,aes(x=c(0),y=r))+geom_violin()+scale_y_continuous(limits=c(-0.1,0.1))

##
pred1 <- fread("result_old/pred_lgb_Melting.csv")
p <- abs(pred1$error)
print(quantile(p,c(0.05,0.1,0.3,0.5,0.6,0.7,0.8,0.9,0.95)))


## wilcoxon signed rank test
wilcox.exact(pred1$ratio, pred2$ratio, paired = TRUE, conf.int=T,conf.level=0.95)
wilcox.exact(abs(pred1$error), abs(pred2$error), paired = TRUE, conf.int=T,conf.level=0.95)


## histogram: provided by N. Nakano
df = na.omit(rbind(data.frame(ratio=pred1$ratio, group=1), 
                   data.frame(ratio=pred2$ratio, group=2)))
bins <- 30
#  g <- ggplot(df, aes(x = ratio,  fill=group,group=group)) + geom_histogram(bins = bins, aes(y=..density.. / 100), alpha=0.6, position='identity')+scale_y_continuous(labels = percent_format())
#  g <- g + stat_bin(aes(y = cumsum(cnt <<- ..count..) / sum(..count..), color=group,group=group), geom = "line", bins = bins)
#  g <- g + theme(axis.title.x = element_blank(), axis.title.y = element_blank()) + xlim(0,0.25) +theme(legend.position = 'none')

breaks = c(0, 0.25, 0.5, 0.75, 1.0)
labels = c('0', '10%', '20%', '30%', '40%')
g = ggplot(df, aes(x=ratio, fill=group, group=group)) +
  geom_histogram(bins=bins, aes(y=..density../100*2.5), alpha=0.4, position='identity') +
  scale_y_continuous(breaks=breaks, labels=labels, sec.axis=sec_axis(~., name='cumulative percentage', labels=percent_format())) +
  stat_ecdf(geom='step', pad=FALSE, position='identity', aes(color=group), size=1.5) +
  xlab('relative prediction error') +
  ylab('percentage') +
  #theme(axis.title.x=element_text(size=14), axis.title.y=element_text(size=14)) + 
  theme(axis.title.x=element_blank(), axis.title.y=element_text(size=14)) + 
  theme(axis.text.x=element_text(size=12), axis.text.y=element_text(size=12)) + 
  xlim(0, 0.25) +
#    theme(legend.position='none')
#  plot(g)
ggsave(paste0("hist_",target1,".png"))


###### hyper-parameter search
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



#########################################
## XGBoost
library(xgboost)

dat <- readRDS(preprocessed_rds)
prepare_prediction("Np",remove_monotropic = T)
prepare_prediction("Nptype",remove_monotropic = F)

bst <- list()
folds <- 5
if(is_regression){
  params <- list(objective = "reg:squarederror")
}else{
  params <- list(objective = "multi:softprob",eval_metric = "mlogloss",num_class = length(unique(dat[[target]])))
}
# model training
for(i in 1:folds){  ## cross validation
  testidx <- which(1:length(dat[,1])%%folds == (i%%folds))
  #  testidx <- which(dat$group==2)  
  dtrain = xgb.DMatrix(data =  as.matrix(dat[-testidx,varcol]), label = dat[-testidx,targetcol])
  dtest = xgb.DMatrix(data =  as.matrix(dat[testidx,varcol]), label = dat[testidx,targetcol])
  watchlist = list(train=dtrain, test=dtest)
  bst[[i]] = xgb.train(params=params, data = dtrain, booster="gbtree",
                       max.depth = 6, 
                       eta = 0.05,  # learning rate
                       alpha = 1.0, # L1
                       lambda = 1.0, # L2
                       #                  nthread = 8, # automatically set
                       nround = 10000, 
                       colsample_bytree = 0.4, min_child_weight = 1.5, subsample = 0.8, gamma = 2,
                       watchlist = watchlist, 
                       early_stopping_rounds = 1000,  # set to 100 for quick test
                       print_every_n = 500,
  )
}

# prediction using the learned model
#bst <- readRDS(paste0(target,".rds"))
prediction <- list()
for(i in 1:folds){
  testidx <- which(1:length(dat[,1])%%folds == (i%%folds))
  #  testidx <- which(dat$group==2)  
  dtest = xgb.DMatrix(data =  as.matrix(dat[testidx,varcol]), label = dat[testidx,targetcol])
  p <- predict(bst[[i]], dtest, reshape=T)
  t <- dat[testidx,target]
  if(is_regression){
    prediction <- rbind(prediction,
                        data.frame(
                          ID=dat[testidx,"ID"], SMILES=dat[testidx,"SMILES"], Phases=dat[testidx,"Phases"],
                          pred=p, truth=t, error=p-t,
                          ratio=abs(p-t)/(t-kelv)))
  }else{
    prediction <- rbind(prediction,
                        data.frame(
                          ID=dat[testidx,"ID"], SMILES=dat[testidx,"SMILES"], Phases=dat[testidx,"Phases"],
                          pred=max.col(p, ties.method = "last")-1,
                          prob0=p[,1],prob1=p[,2],prob2=p[,3],
                          truth=t))
  }
}

if(is_regression){
  # plot prediction vs truth
  plotpred(data.frame(prediction$pred,prediction$truth),paste0("XGBoost:",target),sort=T)
  # error ratio
  p <- prediction$ratio
  #plotsort( p )
  print(quantile(p,c(0.05,0.1,0.3,0.5,0.7,0.9,0.95)))
  png(paste0("hist_",target,".png"), width = 1024, height = 600)
  hist(pmax( 0, pmin( p, 0.2)), freq=F, main="Histogram", xlab="error ratio")
  dev.off()
  ## prediction results will be written to a csv: open it with, e.g., excel to see the results
  write.csv(file=paste0("pred_",target,".csv"), prediction, row.names = FALSE)
}else{
  # types: 0 means "not exist", 1 means "exist", 2 means "(*)"; e.g., N* means cholesteric
  print(confusionMatrix(factor(prediction$pred),
                        factor(prediction$truth),
                        mode = "everything"))
  write.csv(file=paste0("pred_",target,".csv"), prediction, row.names = FALSE)
}
saveRDS(bst,paste0(target,".rds"))

### Variable importance
importance <- xgb.importance(model = bst[[1]])
#head(importance,20)
png(paste0("importance_",target,".png"), width = 1024, height = 600)
xgb.plot.importance(head(importance,20))
dev.off()




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


## Gaussian Process: TOO SLOW USE scikit
library(GauPro)
gp <- GauPro(as.matrix(dat[-testidx,varcol]), dat[-testidx,targetcol], parallel=T)
prediction <- gp$prediction(dat[testidx,varcol])





