# This method will be used to plot the train/validation losses after a training of an RNN:
plotLosses <- function(trainFile = "misc/train_losses.csv", valFile = "misc/val_losses.csv")
{
  # Start with reading the datasets:
  traindata <- read.csv(trainFile, header = F)
  valdata <- read.csv(valFile, header = F)
 
  # Retrieve the number of rows:
  nrows <- dim(traindata)[1]
  nrows2 <- dim(valdata)[1]
  
  if(nrows2 != nrows) {
    stop(paste("Mismatch in number of rows :",nrows,"!=",nrows2))
  }
  
  plot(1:nrows,traindata$V1,type="l",col="blue")
  lines(1:nrows,valdata$V1,col="green", lwd=3)
}

