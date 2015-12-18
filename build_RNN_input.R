# This script can be used to build the RNN input data

source("analysis_tools.R")

# Core function to generate input dataset:
rnnGenerateInputDataset <- function(isyms = NULL, period = NULL, fsyms = NULL, offsets = NULL)
{
  if(is.null(isyms)) {
    print("Using default input symbol list.")
    isyms = c("EURUSD","AUDUSD")
  }
  
  if(is.null(period)) {
    print("Using default period.")
    period = "M5"
  }
  
  if(is.null(fsyms)) {
    print("Using default forcast symbol list.")
    fsyms = c("EURUSD")
  }
  
  if(is.null(offsets)) {
    print("Using default offset list.")
    offsets = 5:10
  }
  
  # first generate the raw dataset:
  data <- rnnGetRawDataset(isyms,period)
  data <- rnnComputeForcast(data,fsyms,offsets)
  data <- rnnNormalizeDataset(data)
}

# retrieve the raw data:
rnnGetRawDataset <- function(symbols, period)
{
  data <- generateDataset(symbols, period)
  
  # Here we should return a list, containing the input dataframe,
  # The labels dataframe, and the normalization dataframe:
  list(inputs=data,symbols=symbols)
}

rnnComputeForcast <- function(data,symbols,offsets)
{
  # select only the close prices:
  prices <- data$inputs[,paste0(tolower(symbols),"_close"),with=F]
  
  # Retrieve the number of rows in the dataframe:
  len <- dim(prices)[1]
  
  forcast <- NULL
  
  # compute the forcast for each offset value provided
  for(offset in offsets)
  {
    # retrieve the future values only:
    future <- prices[(offset+1):len,]
    
    # retrieve the present values only that can be used for the comparaison:
    present <- prices[1:(len-offset),]
    
    # substract the values:
    diff <- future - present
    
    #assign the names:
    names(diff) <- paste0(tolower(symbols),"_forcast_",offset)
    
    if(is.null(forcast))
    {
      # assign this as forcast data.table:
      forcast <- diff
    }
    else 
    {
      # compute the minimal length:
      flen <- min(dim(forcast)[1],dim(diff)[1])
      
      # concatenate the forcasts:
      forcast <- cbind(forcast[1:flen,],diff[1:flen,])
    }
  }

  # add those forcast values into the list:
  data$forcasts <- forcast

  # also reduce the size of the input dataframe using the actual forcast size:
  flen <- dim(forcast)[1]
  data$inputs <- data$inputs[1:flen,]
  
  # return the updated list:
  data
}

rnnNormalizeDataset <- function(data)
{
  # Compute the means and std devs:
  # data$fmeans <- colMeans(forcasts)
  # data$fdev <- apply(forcast,2,sd)
  
  # Normalize the forcast data:
  data$forcasts <- as.data.table(scale(data$forcasts))
  data$fmeans <- attributes(data$forcasts)[["scaled:center"]]
  data$fdev <- attributes(data$forcasts)[["scaled:scale"]]
  
  # Additionally we take the tanh of the forcast to get in the range (-1,1)
  data$forcasts <- tanh(data$forcasts)
  
  # Now we also need to rescale the inputs:
  # We should use all the available input symbols:
  allsymbols <- data$symbols
  cprices <- data$input[,paste0(tolower(allsymbols),"_close"),with=F]
  
  # first we just compute the mean/dev from the close prices in the input:
  means <- colMeans(cprices)
  devs <- apply(cprices,2,sd) 
    
  # Repeat the means/devs 4 times for each symbol:
  means <- rep(means,each=4)
  devs <- rep(devs,each=4)
  
  # get all the columns of interest:
  mpaste <- function(A,B) {
    paste0(B,"_",A)
  }
  
  cnames <- as.vector(outer(c("open","high","low","close"),tolower(allsymbols),FUN=mpaste))
  
  prices <- data$input[,cnames,with=F]
  
  # Assign the names for the means:
  names(means) <- cnames
  names(devs) <- cnames
  
  # store the means/devs in the dataset:
  data$imeans <- means
  data$idevs <- devs
  
  # Rescale the prices:
  prices <- scale(prices,center=means, scale=devs)
  
  # Reconstruct the input dataframe:
  # Note that in this process we discard the volume columns
  inputs <- data$input[,c("date","weektime","time"),with=F]
  
  data$inputs <- cbind(inputs,prices)
  
  # return the dataset:
  data
}
