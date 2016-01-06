
plotCorrectSigns <- function(name)
{
  data <- read.csv(paste0("misc/correct_signs_",name,".csv"),header=F);
  plot(data$V1,type="l")
}

addCorrectSigns <- function(name, col="red")
{
  data <- read.csv(paste0("misc/correct_signs_",name,".csv"),header=F);
  lines(data$V1,type="l", col = col)
}

plotTrainLosses <- function(name)
{
  data <- read.csv(paste0("misc/train_losses_",name,".csv"),header=F);
  plot(data$V1,type="l")
}

addTrainLosses <- function(name, col="red")
{
  data <- read.csv(paste0("misc/train_losses_",name,".csv"),header=F);
  lines(data$V1,type="l", col = col)
}
