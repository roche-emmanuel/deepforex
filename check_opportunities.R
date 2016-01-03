library(plyr)
library(dplyr)

sigmoid <- function(x)
{
  1.0/(1.0+exp(-x))  
}

# Method used to check how the logRet with offset predictions can be used
# to find proper opportunities
check_opportunities <- function(ppath = "v34", fpath = "raw_2015_01_to_2015_10", 
                                pred.offset = 2044, symbol = "EURUSD", log.offset = 3,
                                nsteps = 50)
{
  # Read the datafile:
  data <- read.csv(paste0("inputs/",fpath,"/raw_inputs.csv"))
  
  sym <- tolower(symbol)
  sub <- data[paste0(sym,"_close")]
  
  nrows <- dim(sub)[1]
  
  # add the log return with offset column:
  logret <- sub[(1+log.offset):nrows,] / sub[1:(nrows-log.offset),]
  rets <- c(rep(1,log.offset),logret)
  
  # Take the log and compute the normalization for the log returns:
  normed <- scale(log(rets))
  cmean <- attributes(normed)["scaled:center"]
  csig <- attributes(normed)["scaled:scale"]
  print(paste0("Normalization mean=",cmean,", dev=",csig))
  
  sub$logret <- sigmoid(normed)
  
  # here we compute the log with one less offset, that will be used for the comparaisons:
  loff2 <- log.offset-1
  logret2 <- sub[(1+loff2):nrows,1] / sub[1:(nrows-loff2),1]
  rets2 <- c(rep(1,loff2),logret2)
  sub$state <- rets2
  
  # We just need to compare those returns with 1:
  # if we are above 1, then the latest price is higher that the past price and we are in a sell condition
  # if we are under 1, then the latest price is lower that the past price and we are in a buy condition:
  
  sub$buycond <- rets2<1
  
  # now apply the row offset:
  sub <- sub[(pred.offset):nrows,]
  
  # append the prediction and label data:
  pdata <- read.csv(paste0("misc/eval_results_",ppath,".csv"))
  
  nrows2 <- dim(pdata)[1]
  
  # only keep the data that we can use for evaluation:
  sub <- sub[1:nrows2,]
  
  sub$eval_index <- pdata$eval_index
  sub$prediction <- pdata$prediction
  sub$label <- pdata$label
  
  # prepare the [-1,1] predictions/labels:  
  sub$pred <- (sub$prediction-0.5)*2
  sub$lbl <- (sub$label-0.5)*2
  
  # only keep the eval steps we want:
  sub <- filter(as.data.table(sub),eval_index <= nsteps)
  
  # prepare the evaluation:
  levels <- seq(from=0.0, to=0.9, by=0.1)
  evals <- data.frame(siglevel=levels)
  
  goodsign <- NULL
  corr <- NULL
  counts <- NULL
  
  # now filter by the threshold
  for(j in levels)
  {
    # filter the data by keeping only the "high level" predictions
    filt <- filter(sub,((pred>j) & buycond==T) | (pred<(-j) & buycond==F))
    
    # then compute the desired statistics on this subset:
    good <- filt$pred * filt$lbl > 0
    
    ratio <- mean(good)
    
    counts <- c(counts,dim(filt)[1])
    goodsign <- c(goodsign,ratio)
    corr <- c(corr, cor(filt$prediction,filt$label))
  }
  
  evals[paste0("obs_count")] <- counts
  evals[paste0("goodsign_mean")] <- goodsign
  evals[paste0("pred_corr")] <- corr
  
  evals
}
