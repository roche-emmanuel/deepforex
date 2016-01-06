
plotCorrectSigns <- function(name)
{
  data <- read.csv(paste0("misc/correct_signs_",name,".csv"),header=F);
  plot(data$V1,type="l")
}
