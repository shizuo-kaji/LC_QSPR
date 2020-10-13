#######################################
#
# Regression/Classification toolbox
#                by S. Kaji
#                Aug. 2017
#
#######################################

## load necessary packages
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(scales))
suppressPackageStartupMessages(library(plotROC))

## variables which are not used for prediction
nonvar <- c(
  "ID","SMILES","Phases","rac_en","Melting","Melting_type",
  "Ctype","Cm","Cp","Atype","Am","Ap","Btype","Bm","Bp",
  "Ntype","Nm","Np","Dtype","Dm","Dp",
  "Stype","Xtype","Ytype", ## unclear phases
  "Ctype2","Ntype2",
  "Clearing_type","Clearing","num_C","num_H","num_N","prohibited","group"
)
kelv = -273.15

############## function definition ####################
## load csv file
readcsv <- function(filename, header=TRUE){
  origdat <- fread(filename, stringsAsFactors = FALSE, check.names = TRUE)
#  origdat <- read.csv(filename, header=header, sep = ",", stringsAsFactors = FALSE)
  print(paste("# of rows",nrow(origdat)))
  #  sapply(origdat,is.numeric)
  #  origdat[] <- suppressWarnings(lapply(origdat,as.numeric))
  #  origdat <- na.omit(origdat)
  #  origdat[is.na(origdat)] <- 0
  return(origdat)
}

## Read a transposed (variables in rows) CSV
read.tcsv = function(file, header=TRUE, sep=",", isnumeric=TRUE, ...) {
  n = max(count.fields(file, sep=sep), na.rm=TRUE)
  x = readLines(file)
  .splitvar = function(x, sep, n) {
    var = unlist(strsplit(x, split=sep))
    length(var) = n
    return(var)
  }
  x = do.call(cbind, lapply(x, .splitvar, sep=sep, n=n))
  x = apply(x, 1, paste, collapse=sep) 
  ## empty strings are converted to NA
  out = read.csv(text=x, sep=sep, header=header, na.strings = "", ...)
  rownames(out) <- out$X
  out$X <- NULL
  if(isnumeric){
    out <- data.frame(sapply(out,as.numeric))
  }
  return(out)
}

## conver list to data.frame
Unzip <- function(...) rbind(data.frame(), ...)

## L^p norm
lpnorm <- function(x,p){
  return(signif(mean(abs(x)^p)^(1/p)))
}

regression_summary <- function(truth,pred){
  dif <- abs(pred - truth)
  corr <- signif(cor(pred,truth))
  mae <- signif(MAE(pred, truth))
  rmse <- signif(RMSE(pred, truth))
  r2 <- caret::R2(pred, truth, form = "traditional")
  rmse_ratio <- signif(sqrt(mean((dif/(truth-kelv))^2)))
  mae_ratio <- signif(mean(dif/(truth-kelv)))
  sub <- paste(target, "Corr",corr, "R2", r2, "RMSE", rmse, "MAE", mae,"R-RMSE", rmse_ratio, "R-MAE", mae_ratio, sep=",")
  return(sub)
}

## plot regression graph
plotpred <- function(vals,title="",sort=TRUE,ymin=0,ymax=350){
  # sort
  if(sort){
    sortlist <- order(vals[[1]])
    vals <- data.frame(vals[sortlist,])
  }
  # plot
  plot(vals[[1]], type="l", xlim=c(1,length(vals[[1]])), ylim=c(ymin,ymax), col="blue", ann=FALSE)
  if(dim(vals)[[2]]>1){
    sub <- regression_summary(vals[[1]],vals[[2]])
    par(new=T)
    plot(vals[[2]], type="l", xlim=c(1,length(vals[[1]])), ylim=c(ymin,ymax), col="red",main=title, sub=sub, xlab="", ylab="")
  }
  if(dim(vals)[[2]]>2){
    for(i in 3:length(vals)){
      par(new=T)
      plot(vals[[i]], type="l", xlim=c(1,length(vals[[1]])), ylim=c(ymin,ymax), col="green", ann=FALSE)
    }
  }
}

## 

## plot sorted
plotsort <- function(val){
    sortlist <- order(val)
    val <- data.frame(val[sortlist])
    rmse <- signif(sqrt(mean(val[[1]]^2)))
    me <- signif(mean(abs(val[[1]])))
    sub <- paste(target, "RMSE", rmse, "MAE", me, sep=",")
    print(sub)
    plot(val[[1]], type="l", xlim=c(1,length(val[[1]])), ylim=c(-0.2,0.2), col="red",
         sub=sub, xlab="", ylab="")
}

## plot classification table
plottab <- function(prediction,truth,title=""){
  tab <- table(Pred=prediction,Truth=truth)
  print(tab)
  print(classAgreement(tab))
}

## plot classification table
plottab2 <- function(prediction,truth,title=""){
  tab <- table(Pred=prediction,Truth=truth)
  print(tab)
  print(classAgreement(tab))
  tp <- tab[2,2]
  fp <- tab[2,1]
  fn <- tab[1,2]
  crr <- cor(as.numeric(prediction), truth, use="pairwise.complete.obs")
  print(paste("recall:",tp/(tp+fn),"precision:",
              tp/(tp+fp),"F1:",2*tp/(2*tp+fp+fn),"Cor:",crr))
}

## prepare for regression; set variables and formula
prepare_prediction <- function(targ,only_type=-1,remove_monotropic=FALSE,combine_chiral=FALSE){
  dat <- master_dat
  target <<- targ
  expvar <<- setdiff(colnames(dat), nonvar)
  varcol <<- colnames(dat) %in% expvar
  f <<- as.formula(paste(paste(target, collapse=" + "),paste(expvar, collapse=" + "), sep=" ~ "))
  n <- nchar(target)
  lt <- substr(target,1,1) # phase name
  mp <- substr(target,n,n) # p or m
  if( (target != "Clearing") &&  (target != "Melting")){
    ## remove ambiguous (X can be the target)
    dat <- dat[dat[[paste0(lt,"type")]]>0 | dat$Xtype==0,] 
    ## remove ambiguous (S can be A,B,C,C*)
  }
  if(lt=="A"|| lt=="B" || lt=="C"){
    dat <- dat[dat[[paste0(lt,"type")]]>0 | dat$Stype==0,] 
  }else if(lt=="D"){   ## remove ambiguous (Y can be Dh, Dr, ...)
    dat <- dat[dat[[paste0(lt,"type")]]>0 | dat$Ytype==0,] 
  }

  if((n<4 && (mp=="m" || mp=="p")) || (target=="Clearing") || (target=="Melting")){ ## transition temperature
    is_regression <<- T
    #dat <<- dat[!is.na(dat[[target]]),]
    # remove rows with target < kelv or target > 800 and with target = 0 as they are not reliable
    dat <- dat[dat[[target]] >= kelv & dat[[target]] <= 800 & dat[[target]]!=0,] 
  }else{  ## existence of phase
    is_regression <<- F
    if(mp=="2"){
      dat[[target]] <- 0
      dat[dat[[paste0(lt,"type")]] > 0, target] <- 1      
    }
  }

  if(remove_monotropic && is_regression){
    dat <- dat[dat[[paste0(lt,"m")]] < dat[[paste0(lt,"p")]],]
    if(mp=="m"){
      dat <- dat[dat[[paste0(lt,"p")]] >= kelv & dat[[paste0(lt,"p")]] <= 800 & dat[[paste0(lt,"p")]]!=0,] 
    }else if(mp=="p"){
      dat <- dat[dat[[paste0(lt,"m")]] >= kelv & dat[[paste0(lt,"m")]] <= 800 & dat[[paste0(lt,"m")]]!=0,] 
    }
  }
  if(only_type >= 0){
    if(target=="Melting"){
      dat <- dat[dat[["Melting_type"]]==only_type,]
    }else if(target=="Clearing"){
      dat <- dat[dat[["Clearing_type"]]==only_type,]
    }else{
      dat <- dat[dat[[paste0(lt,"type")]]==only_type,]
    }
  }
  
  targetcol <<- colnames(dat) %in% target
  return(as.data.frame(dat))
}

##  prediction
prediction_result <- function(dat,bst,folds){
  prediction <- list()
  for(i in 1:folds){
    testidx <- which(1:length(dat[,1])%%folds == (i%%folds))
    #  testidx <- which(dat$group==2)  
    p <- predict(bst[[i]], as.matrix(dat[testidx,varcol]),reshape=T)
    t <- dat[testidx,target]
    if(is_regression){
      prediction <- rbind(prediction,
                          data.frame(
                            ID=dat[testidx,"ID"], SMILES=dat[testidx,"SMILES"], Phases=dat[testidx,"Phases"],
                            pred=p, truth=t, error=p-t,
                            ratio=abs(p-t)/(t-kelv)))
    }else{
      ps <- data.frame(
        ID=dat[testidx,"ID"], SMILES=dat[testidx,"SMILES"], Phases=dat[testidx,"Phases"],
        truth=t
      )
      if(n_class==2){
        ps$pred <- 0
        ps[p>0.5, "pred"] <- 1
        ps$prob1=p      
      }else{
        ps$pred <- as.factor(max.col(p, ties.method = "last")-1)
        for(i in 1:n_class){
          ps[[paste0("prob",i-1)]] <- p[,i]
        }
      }
      prediction <- rbind(prediction,ps)
    }  
  }
  return(prediction)
}

prediction_plot <- function(prediction, plot_value=FALSE, plot_ROC=TRUE, plot_hist=TRUE)
if(is_regression){
  print(regression_summary(prediction$truth,prediction$pred))
  # plot prediction vs truth
  if(plot_value){
    plotpred(data.frame(prediction$truth,prediction$pred),paste0(target),sort=T)
  }
  if(plot_hist){
    # error ratio histogram
    #p <- prediction$ratio
    #  plotsort( p )
    #  print(quantile(p,c(0.05,0.1,0.3,0.5,0.7,0.9,0.95)))
    #hist(pmax( 0, pmin( p, 0.3)), freq=F, main="Histogram", xlab="error ratio")
    bins <- 30
    g <- ggplot(prediction, aes(x = ratio)) + geom_histogram(bins = bins, aes(y=..density.. / 100))+scale_y_continuous(labels = percent_format())
    g <- g + stat_bin(aes(y = cumsum(cnt <<- ..count..) / sum(..count..)), geom = "line", bins = bins)
    #    scale_y_continuous(sec.axis = sec_axis(~ . / max(cnt), name = "percentage[%]"))
    g <- g + theme(axis.title.x = element_text("error ratio"), axis.title.y = element_blank()) + xlim(0,0.25)
    suppressWarnings(plot(g))
    ggsave(paste0("hist_",target,".png"))
  }
  ## prediction results will be written to a csv: open it with, e.g., excel to see the results
  write.csv(file=paste0("predidction_",target,".csv"), prediction, row.names = FALSE)
}else{
  ## classification results
  # types: 0 means "not exist", 1 means "exist", 2 means "(*)"; e.g., N* means cholesteric
  print(confusionMatrix(factor(prediction$pred),
                        factor(prediction$truth),
                        mode = "everything", 
                        positive="1"))
  if(n_class==2 && plot_ROC){
    g <- ggplot(prediction, aes(m=prob1, d=truth))+ geom_roc(n.cuts=0) + coord_equal() + style_roc()
    g <- g  + annotate("text", x=0.75, y=0.25, label=paste("AUC =", round((calc_auc(g))$AUC, 4)))
    ggsave(paste0("roc_lgb_",target,".png"), plot=g)
  }
}

## variable importance
importance_csv <- function(bst,file){
  for(i in 1:folds){
    il <- lgb.importance(model = bst[[i]])
    if(i==1){
      importance <- data.frame(il)
    }else{
      importance <- full_join(importance, il, by="Feature")
    }
  }
  write.csv(file=file, importance, row.names = FALSE)
  return(importance)
}

############## End of function definition ####################

