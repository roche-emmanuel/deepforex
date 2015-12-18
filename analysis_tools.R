# This script will provide helper functions to analyze the minute datasets
data_path <- "data"

# "+" = function(x,y) {
#   if(is.character(x) || is.character(y)) {
#     return(paste(x , y, sep=""))
#   } else {
#     .Primitive("+")(x,y)
#   }
# }

# load the dependency packages:
library(data.table)
library(plyr)

# Method used to generate a complete dataset for a given list
# of symbols and a given periodicity:
generateDataset <- function(symbols,period)
{
  result <- NULL
  
  for(sym in symbols)
  {
    dpath <- paste0(data_path,"/",sym,"_",period,".csv")
    
    # load the dataset:
    print(paste0("Loading dataset ",dpath,"..."))
    data <- fread(dpath)
    
    print("Checking symbol dataset...")    
    checkSymbolDataset(data)
    
    # rename the columns of the dataset:
    sname <- tolower(sym)
    
    names(data) <- c("date","time",
                     paste0(sname,"_open"),
                     paste0(sname,"_high"),
                     paste0(sname,"_low"),
                     paste0(sname,"_close"),
                     paste0(sname,"_volume"))
    
    if(is.null(result)) 
    {
      result <- data
    }
    else
    {
      print("Merging datasets...")
      result <- merge(result,data,by=c("date","time"))
    }
  }
  
  print("Finalizing symbols dataset...")
  finalizeDataset(result)
}

# method used to check basic components on a single symbol dataset:
checkSymbolDataset <- function(data)
{
  # Start with changing the col names:
  names(data) <- c("date","time","open","high","low","close","volume")
  
  # Check that high is always bigger or equal to all prices:
  bad <- data$high < data$open || data$high < data$low || data$high < data$close
  if(any(bad)) {
    stop("Detected invalid high prices")
  }
  
  # Check that low is always smaller or equal to all prices:
  bad <- data$low > data$open || data$low > data$high || data$low > data$close
  if(any(bad)) {
    stop("Detected invalid low prices")
  }  
}

# Method used to format the final dataset as desired
finalizeDataset <- function(data)
{
  # Update the date format:
  print("Updating dates...")
  data$date <- as.Date(data$date,"%Y.%m.%d")
  
  # Update the time format:
  print("Updating times...")
  offset <- as.numeric(strptime("0", format="%S"))
  data$time <- (as.numeric(strptime(data$time, format="%H:%M")) - offset)/60.0
  
  # generate week day column from dates:
  # We offset by one so that monday is 0
  days <- as.POSIXlt(data$date)$wday - 1
  
  # Then we generate a new week time column:
  data$weektime <- days*24*60 + data$time
  nc <- ncol(data)
  
  setcolorder(data,c(1,nc,2:(nc-1)))
  #data <- data[c(1,nc,2:(nc-1))]
  
  #data <- cbind(data$date,weektime,data[,2:ncol(data)])
  
  data
}

# Method used to check the dates in the dataset
checkDates <- function(data)
{
  # Check the number of time steps on each day:
  # We start with generating a summary of the start / stop times for each day:
  
  # prepare a custom function for the parsing of the day:
  custom <- function(date) { as.POSIXlt(date[1])$wday - 1 }
  
  print("Checking day time steps range...")
  daytimes <- ddply(data,"date",summarize,
                    starttime=min(time), 
                    stoptime=max(time), 
                    nsteps = length(time), 
                    day = custom(date))
  
  # Now with the previous daytimes dataset, we should check if
  # we have the same range for each given day:
  print("Validating day time steps range...")
  sumtimes <- ddply(daytimes,"day",summarize, 
                    startstatus=max(starttime)==min(starttime),
                    stopstatus=max(stoptime)==min(stoptime),
                    stepstatus=max(nsteps)==min(nsteps))
  
  sumtimes
}

# Given a column of data, we can check how this data varies after a given number of time step
# for instance given close prices for EURUSD, we can compute how the close prices are distributed relative to the
# current close price in X minutes:
getFutureDistribution <- function(prices,nsteps)
{
  # Retrieve the number of rows in the dataframe:
  len <- dim(prices)[1]
  
  # retrieve the future values only:
  future <- prices[(nsteps+1):len,]
  
  # retrieve the present values only that can be used for the comparaison:
  present <- prices[1:(len-nsteps),]
  
  # substract the values:
  diff <- future - present
  
  # compute the mean and the variance of the differences:
  mrow <- sapply(diff,mean)
  sdrow <- sapply(diff,sd)
  
  sdrow
}

# Method used to compute the evolution of the variance on a given range of steps:
computeVarianceEvolution <- function(prices, steps)
{
  num <- length(steps)
  result <- NULL
  # result <- data.table(steps = integer(num))
  
#   names <- names(prices)
#   len <- length(names)
#   for(i in seq_along(names)) {
#     result <- cbind(result,data.table(x=numeric(num)))
#   }
  
  for(i in seq_along(steps)) {
    res <- getFutureDistribution(prices,steps[i])
    # result[i,] <- c(i,res)
    result <- rbind(result,res)
  }
  
  result <- cbind(steps,result)
  #names(result)[1] <- "num_steps"
  row.names(result) <- NULL
  
  result
}
