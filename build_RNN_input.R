# This script can be used to build the RNN input data

source("analysis_tools.R")

# Core function to generate input dataset:
rnnGenerateInputDataset <- function(isyms = NULL, period = NULL, fsyms = NULL, 
                                    offsets = NULL, cov.range = NULL, norms = NULL)
{
  if(is.null(isyms)) {
    print("Using default input symbol list.")
    isyms = c("EURUSD","AUDUSD","GBPUSD","NZDUSD","USDCAD","USDCHF","USDJPY")
    #isyms = c("EURUSD","AUDUSD")
  }
  
  if(is.null(period)) {
    print("Using default period.")
    period = "M1"
    #period = "M5"
  }
  
  if(is.null(fsyms)) {
    print("Using default forcast symbol list.")
    fsyms = c("EURUSD")
  }
  
  if(is.null(offsets)) {
    print("Using default offset list.")
    offsets = 5:10
  }
  
  if(is.null(cov.range)) {
    cov.range <- c("2004-01-01","2007-01-01")
  }
  
  print(paste("Using coverage from",cov.range[1],"to",cov.range[2]))
        
  # first generate the raw dataset:
  print("Generating raw dataset...")
  data <- rnnGetRawDataset(isyms,period,cov.range)
  print("Computing forcasts...")
  data <- rnnComputeForcast(data,fsyms,offsets)
  print("Normalizing dataset...")
  data <- rnnNormalizeInputs(data, norms)
  data <- rnnNormalizeForcasts(data, norms)
}

# retrieve the raw data:
rnnGetRawDataset <- function(symbols, period, cov.range)
{
  data <- generateDataset(symbols, period, cov.range)
  
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

rnnNormalizeInputs <- function(data, norms = NULL)
{
  # Now we also need to rescale the inputs:
  # We should use all the available input symbols:
  allsymbols <- data$symbols
  
  # get all the columns of interest:
  mpaste <- function(A,B) {
    paste0(B,"_",A)
  }
  
  cnames <- as.vector(outer(c("open","high","low","close"),tolower(allsymbols),FUN=mpaste))
  prices <- data$inputs[,cnames,with=F]
  
  if(!is.null(norms))
  {
    print("Normalizing input data with refs...")
    data$imeans <- norms$imeans
    data$idevs <- norms$idevs
  }
  else
  {
    print("Computing input data normalization...")
    # first we just compute the mean/dev from the close prices in the input:
    cprices <- data$inputs[,paste0(tolower(allsymbols),"_close"),with=F]
    means <- colMeans(cprices)
    devs <- apply(cprices,2,sd) 
    
    # Repeat the means/devs 4 times for each symbol:
    means <- rep(means,each=4)
    devs <- rep(devs,each=4)
    
    # Assign the names for the means:
    names(means) <- cnames
    names(devs) <- cnames
    
    # store the means/devs in the dataset:
    data$imeans <- means
    data$idevs <- devs    
  }
  
  # Rescale the prices:
  prices <- scale(prices,center=data$imeans, scale=data$idevs)
  
  # Reconstruct the input dataframe:
  # Note that in this process we discard the volume columns
  #inputs <- data$inputs[,c("date","weektime","time"),with=F]
  
  # The weektime should also be normalized in the range [-1,1]:
  weektime <- (data$inputs$weektime/(5*24*60) - 0.5)*2.0
  
  # Also normalize the daytime:
  daytime <- (data$inputs$time/(24*60) - 0.5)*2.0
  
  data$inputs <- cbind(data$inputs[,"date",with=F],weektime,daytime,prices)
  
  # return the dataset:
  data  
}

rnnNormalizeForcasts <- function(data, norms = NULL)
{
  # Compute the means and std devs:
  # data$fmeans <- colMeans(forcasts)
  # data$fdev <- apply(forcast,2,sd)
  
  # Normalize the forcast data:
  if(!is.null(norms)) 
  {
    print("Normalizing forcast data with refs...")
    data$forcasts <- as.data.table(scale(data$forcasts,center=norms$fmeans, scale=norms$fdevs))
    data$fmeans <- norms$fmeans
    data$fdevs <- norms$fdevs
  }
  else 
  {
    print("Computing forcast data normalization...")
    scaled <- scale(data$forcasts)
    data$forcasts <- as.data.table(scaled)
    data$fmeans <- attributes(scaled)[["scaled:center"]]
    data$fdevs <- attributes(scaled)[["scaled:scale"]]
  }
  
  # Additionally we take the tanh of the forcast to get in the range (-1,1)
  data$forcasts <- tanh(data$forcasts)
  
  data
}

# Method used to write the input dataset into a given input folder
rnnWriteDataset <- function(data,folder)
{
  path <- paste0("inputs/",folder)
  
  # create the target folder:
  if(!dir.exists(path))
  {
    print(paste("Creating folder",path))
    dir.create(path,recursive = T)
  }
  else
  {
    # The folder already exist, we should clean it:
    print(paste("Cleaning folder",path))
    rmfile <- function(fname)
    {
      file.remove(paste0(path,"/",fname))
    }
    do.call(rmfile,list(list.files(path)))
  }
  
  # write the input dataset:
  # Note that we do not write the dates in this file:
  write.csv(data$inputs[,-c("date"),with=F],paste0(path,"/inputs.csv"),row.names=F)
  
  # write the forcasts:
  write.csv(data$forcasts,paste0(path,"/forcasts.csv"),row.names=F)
  
  # write the inputs means/devs:
  #write.csv(data$imeans,paste0(path,"/imeans.csv"),row.names=F)
  #write.csv(data$idevs,paste0(path,"/idevs.csv"),row.names=F)
  
  # write the forcasts means/devs:
  #write.csv(data$fmeans,paste0(path,"/fmeans.csv"),row.names=F)
  #write.csv(data$fdevs,paste0(path,"/fdevs.csv"),row.names=F)
  
  # for convinience, we also write the number of samples available:
  len <- dim(data$inputs)[1]
  if(len != dim(data$forcasts)[1])
  {
    stop("Mismatch in inputs and forcasts datasets")
  }
  
  cfgfile <- paste0(path,"/","dataset.lua") 
  cat("return {", file=cfgfile, sep="\n")
  cat(paste0("\tnum_samples = ",len,","), file=cfgfile, sep="\n", append=T)
  cat(paste0("\tnum_inputs = ",(dim(data$inputs)[2]-1),","), file=cfgfile, sep="\n", append=T)
  cat(paste0("\tnum_forcasts = ",(dim(data$forcasts)[2]),","), file=cfgfile, sep="\n", append=T)
  cat(paste0("\timeans = { ",paste(data$imeans,collapse = ", ")," },"), file=cfgfile, sep="\n", append=T)
  cat(paste0("\tidevs = { ",paste(data$idevs,collapse = ", ")," },"), file=cfgfile, sep="\n", append=T)
  cat(paste0("\tfmeans = { ",paste(data$fmeans,collapse = ", ")," },"), file=cfgfile, sep="\n", append=T)
  cat(paste0("\tfdevs = { ",paste(data$fdevs,collapse = ", ")," },"), file=cfgfile, sep="\n", append=T)
  cat("}", file=cfgfile, sep="\n", append=T)
}

# Method used to generate the features dataset:
rnnGenFeatures <- function(data,seq_len = 50)
{
  print(paste("Generating feature dataset with sequence length:",seq_len))
  
  # First we should prepare the final matrix:
  dims <- dim(data$inputs)
  
  # number of rows should be the same as the inputs, except that we remove the first (seq_len-1) rows:
  nrows <- dims[1] - (seq_len-1)
  
  # number of cols should be 2 + seq_len * num_symbols * 4:
  stride <- length(data$symbols) * 4
  ncols <- 2 + seq_len * stride
  
  # Prepare the matrix:
  features <- matrix(nrow = nrows, ncol = ncols)
  
  # first we copy the weektime, daytime:
  features[,c(1,2)] <- as.matrix(data$inputs[-(1:49),c("weektime","daytime"),with=F])
  
  # prepare the actual feature data as a matrix:
  fdat <- as.matrix(data$inputs[,-c("date","weektime","daytime"),with=F])
  
  # now we iterate on each sequence index to inject the fdat into the desired sub matrix of features:
  # start with the third column:
  offset <- 3
  
  for(i in 1:seq_len)
  {
    print(paste0("Processing sequence index ",i,"..."))
    features[,offset:(offset+stride-1)] <- fdat[i:(i+nrows-1),]
    offset <- offset+stride
  }
  
  # Now store the global features matrix into the data:
  data$features <- features
  data$seq_length <- seq_len
  
  # return the data:
  data
}


# Simple method to write the raw inputs to a raw dataset file:
rnnWriteRawInputs <- function(isyms = NULL, period = NULL, cov.range = NULL, 
                              out.dir = "raw_2004_01_to_2007_01", norms = NULL)
{
  path <- paste0("inputs/",out.dir)
  
  # create the target folder:
  if(!dir.exists(path))
  {
    print(paste("Creating folder",path))
    dir.create(path,recursive = T)
  }
  else
  {
    # The folder already exist, we should clean it:
    print(paste("Cleaning folder",path))
    rmfile <- function(fname)
    {
      file.remove(paste0(path,"/",fname))
    }
    do.call(rmfile,list(list.files(path)))
  }
  
  if(is.null(isyms)) {
    print("Using default input symbol list.")
    isyms = c("EURUSD","AUDUSD","GBPUSD","NZDUSD","USDCAD","USDCHF","USDJPY")
    #isyms = c("EURUSD","AUDUSD")
  }
  
  if(is.null(period)) {
    period = "M1"
    #period = "M5"
  }
  print(paste("Using period:",period))
  
  if(is.null(cov.range)) {
    cov.range <- c("2004-01-01","2007-01-01")
  }
  print(paste("Using coverage from",cov.range[1],"to",cov.range[2]))
  
  print("Reading raw inputs...")  
  data <- rnnGetRawDataset(isyms,period,cov.range)
  
  #print("Normalizing inputs...")  
  #data <- rnnNormalizeInputs(data, norms)
  
  print(paste("Writing outputs to folder",path))

  # write the input dataset:
  # Note that we do not write the dates in this file:
  write.csv(data$inputs[,-c("date"),with=F],paste0(path,"/raw_inputs.csv"),row.names=F)
  
  len <- dim(data$inputs)[1]
  
  cfgfile <- paste0(path,"/","dataset.lua") 
  cat("return {", file=cfgfile, sep="\n")
  cat(paste0("\tnum_samples = ",len,","), file=cfgfile, sep="\n", append=T)
  cat(paste0("\tnum_inputs = ",(dim(data$inputs)[2]-1),","), file=cfgfile, sep="\n", append=T)
  #cat(paste0("\timeans = { ",paste(data$imeans,collapse = ", ")," },"), file=cfgfile, sep="\n", append=T)
  #cat(paste0("\tidevs = { ",paste(data$idevs,collapse = ", ")," },"), file=cfgfile, sep="\n", append=T)
  cat("}", file=cfgfile, sep="\n", append=T)
  
  data
}
