
#Method to plot the forcasts value against a prediction:
plotPrediction <- function(input.dir="inputs/test_2007_01_to_2007_04",count=-1)
{
  # Start with reading the datasets:
  forcasts <- read.csv(paste0(input.dir,"/forcasts.csv"), header = T)
  pred <- read.table(paste0(input.dir,"/result_gen.txt"), header = F)
  
  # Retrieve the number of rows:
  nrows <- dim(forcasts)[1]
  nrows2 <- dim(pred)[1]
  
  if(nrows2 != nrows) {
    stop(paste("Mismatch in number of rows :",nrows,"!=",nrows2))
  }
  
  if(count > 0)
  {
    nrows = count
  }
  
  y1 <- forcasts[1:nrows,1]
  y2 <- pred[1:nrows,1]
  
  plot(1:nrows,y1,type="l",col="black")
  lines(1:nrows,y2,col="green", lwd=3)
}
