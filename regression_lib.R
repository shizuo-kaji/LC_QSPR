#######################################
#
# Regression/Classification toolbox
#                by S. Kaji
#                Aug. 2017
#
#######################################

## load necessary packages
## install them using the following lines if they do not exist in your system
#install.packages(c("clue","coefplot","devtools","e1071","FNN","fastcluster","FactoMineR","FSelector","ggplot2","glmnet","kernlab","pls","PredPsych","Rtsne","randomForest"))
#install.packages("stringi", type="source")
#devtools::install_github("hoxo-m/pforeach")
library(ggplot2)
library(pforeach)
library(kernlab)
library(e1071)
library(gbm)
library(caret)
library(dplyr)  


## variables which are not used for prediction
nonvar <- c(
  "ID","SMILES","Phases","rac_en","Melting","Melting_type",
  "Ctype","Cm","Cp","Atype","Am","Ap","Btype","Bm","Bp",
  "Ntype","Nm","Np","Dtype","Dm","Dp","Stype","Sm","Sp",
  "Clearing_type","Clearing","num_C","num_H","num_N","prohibited","group"
)
kelv = -273.15

############## function definition ####################
## load csv file
readcsv <- function(filename, header=TRUE){
  origdat <- read.csv(filename, header=header, sep = ",", stringsAsFactors = FALSE)
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

## plot regression graph: truth,pred_mean,pred_high,pred_low
plotpred <- function(vals,title="",sort=TRUE,ymin=0,ymax=350){
  # sort
  if(sort){
    sortlist <- order(vals[[1]])
    vals <- data.frame(vals[sortlist,])
  }
  # plot
  plot(vals[[1]], type="l", xlim=c(1,length(vals[[1]])), ylim=c(ymin,ymax), col="blue", ann=FALSE)
  # compute correlation and RMSE
  if(dim(vals)[[2]]>1){
    dif = vals[[1]]-vals[[2]]
    corr <- signif(cor(vals[[2]], vals[[1]]))
    rmse <- signif(sqrt(mean(dif^2)))
    me <- signif(mean(abs(dif)))
    rmse_ratio <- signif(sqrt(mean((dif/(vals[[1]]+273.15))^2)))
    me_ratio <- signif(mean(abs(dif/(vals[[1]]+273.15))))
    sub <- paste(target, "Corr",corr, "RMSE", rmse, "MAE", me,"R-RMSE", rmse_ratio, "R-MAE", me_ratio, sep=",")
    print(sub)
    par(new=T)
    plot(vals[[2]], type="l", xlim=c(1,length(vals[[1]])), ylim=c(ymin,ymax), col="red",
         main=title, sub=sub, xlab="", ylab="")
  }
  if(dim(vals)[[2]]>2){
    for(i in 3:length(vals)){
      par(new=T)
      plot(vals[[i]], type="l", xlim=c(1,length(vals[[1]])), ylim=c(ymin,ymax), col="green", ann=FALSE)
    }
  }
}

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
plottab <- function(prediction,truth,title=""){
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
prepare_prediction <- function(target,remove_monotropic=FALSE){
  target <<- target
  expvar <<- setdiff(colnames(dat), nonvar)
  targetcol <<- colnames(dat) %in% target
  varcol <<- colnames(dat) %in% expvar
  f <<- as.formula(paste(paste(target, collapse=" + "),paste(expvar, collapse=" + "), sep=" ~ "))
  n <- nchar(target)
  lt <- substr(target,1,1) # phase name
  mp <- substr(target,n,n) # p or m

  ## remove ambiguous (X can be the target)
  dat <<- dat[dat[[paste0(lt,"type")]]>0 | dat$Xtype==0,] 
  ## remove ambiguous (S can be A,B,C,C*)
  if(lt=="A"|| lt=="B" || lt=="C"){
    dat <<- dat[dat[[paste0(lt,"type")]]>0 | dat$Stype==0,] 
  }else if(lt=="D"){   ## remove ambiguous (Y can be Dh, Dr, ...)
    dat <<- dat[dat[[paste0(lt,"type")]]>0 | dat$Ytype==0,] 
  }

  if((n<4 && (mp=="m" || mp=="p")) || (target=="Clearing") || (target=="Melting")){
    is_regression <<- T
    #dat <<- dat[!is.na(dat[[target]]),]
    # remove rows with target < kelv or target > 800 and with target = 0 as they are not reliable
    dat <<- dat[dat[[target]] >= kelv & dat[[target]] <= 800 & dat[[target]]!=0,] 
  }else{
    is_regression <<- F
  }

  if(remove_monotropic && is_regression){
    dat <<- dat[dat[[paste0(lt,"m")]] < dat[[paste0(lt,"p")]],]
    if(mp=="m"){
      dat <<- dat[dat[[paste0(lt,"p")]] >= kelv & dat[[paste0(lt,"p")]] <= 800 & dat[[paste0(lt,"p")]]!=0,] 
    }else if(mp=="p"){
      dat <<- dat[dat[[paste0(lt,"m")]] >= kelv & dat[[paste0(lt,"m")]] <= 800 & dat[[paste0(lt,"m")]]!=0,] 
    }
  }
}
############## End of function definition ####################

