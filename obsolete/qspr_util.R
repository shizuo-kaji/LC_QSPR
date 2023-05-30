## missing value handling
## Impute missing values
#library(missForest)
#dat <- missForest(dat)
## zero fill
#dat[is.na(dat)] <- 0

## rcdk
library(rcdk)
mols <- load.molecules( c('test.smi') )
view.molecule.2d(mols[[1]])
descNames <- unique(unlist(sapply(get.desc.categories(), get.desc.names)))
descs <- eval.desc(mols, descNames)
## save file
#write.molecules(mols, filename='mymols.sdf')


###########################################
## save csv for chainer-chemistry
folds=5
target ='Ntype2'
cols = c('SMILES','ID','Phases','Np','Nm','Ntype')
for(i in 1:folds){
  testidx <- which(1:length(dat[,1])%%folds == (i%%folds))
  write.csv(file=paste0(target,"_train_",i,".csv"), dat[-testidx,cols], row.names = FALSE)
  write.csv(file=paste0(target,"_test_",i,".csv"), dat[testidx,cols], row.names = FALSE)
}

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
dat <- dat[,selected]

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

# umap
crd <- umap(dat[,varcol])
summary(crd$layout)
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
