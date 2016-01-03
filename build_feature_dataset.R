
#method used to build a simple feature dataframe from a raw input
# and merge it with the corresponding generated predictions
build_feature_dataset <- function(fpath = "mt5_2015_12", symbol = "EURUSD", log.offset = 1, ppath = "v36")
{
  # Read the datafile:
  data <- read.csv(paste0("inputs/",fpath,"/raw_inputs.csv"))
  
  sym <- tolower(symbol)
  sub <- data[c("timetag",paste0(sym,"_close"))]
  
  nrows <- dim(sub)[1]
  
  # add the log return with offset column:
  logret <- sub[(1+log.offset):nrows,2] / sub[1:(nrows-log.offset),2]
  rets <- c(rep(1,log.offset),logret)
  
  # Take the log and compute the normalization for the log returns:
  normed <- scale(log(rets))
  cmean <- attributes(normed)["scaled:center"]
  csig <- attributes(normed)["scaled:scale"]
  print(paste0("Normalization mean=",cmean,", dev=",csig))
  
  sub$logret <- sigmoid(normed)
  
  # merge with the predictions:
  preds <- read.csv(paste0("misc/eval_results_",ppath,".csv"))
  
  sub <- merge(sub, preds, "timetag")
  
#   
#   # here we compute the log with one less offset, that will be used for the comparaisons:
#   loff2 <- log.offset-1
#   logret2 <- sub[(1+loff2):nrows,1] / sub[1:(nrows-loff2),1]
#   rets2 <- c(rep(1,loff2),logret2)
#   sub$state <- rets2
#   
#   # We just need to compare those returns with 1:
#   # if we are above 1, then the latest price is higher that the past price and we are in a sell condition
#   # if we are under 1, then the latest price is lower that the past price and we are in a buy condition:
#   
#   sub$buycond <- rets2<1
#   
#   # now apply the row offset:
#   sub <- sub[(pred.offset):nrows,]
#   
#   # append the prediction and label data:
#   pdata <- read.csv(paste0("misc/eval_results_",ppath,".csv"))
#   
#   nrows2 <- dim(pdata)[1]
#   
#   # only keep the data that we can use for evaluation:
#   sub <- sub[1:nrows2,]
  sub
}