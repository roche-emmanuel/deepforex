# This script will provide helper functions to analyze the minute datasets
data_path <- "data"

# loade the data.table package:
library(data.table)

# Method used to load the raw M1 dataset for a given symbol:
loadM1 <- function(symbol)
{
  dpath <- paste0(data_path,"/",symbol,"_M1.csv")
  
  print(paste0("Loading dataset ",dpath,"..."))
  
  # load the dataset:
  fread(dpath)
}

# Method used to format the dataset as desired
prepareDataset <- function(dset)
{
  # Start with changing the col names:
  names(dset) <- c("date","time","open","high","low","close","volume")
  
  # Check that high is always bigger or equal to all prices:
  bad <- dset$high < dset$open || dset$high < dset$low || dset$high < dset$close
  if(any(bad)) {
    stop("Detected invalid high prices")
  }
  
  # Check that low is always smaller or equal to all prices:
  bad <- dset$low > dset$open || dset$low > dset$high || dset$low > dset$close
  if(any(bad)) {
    stop("Detected invalid low prices")
  }
  
  
}
