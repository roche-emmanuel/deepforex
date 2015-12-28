
library(dplyr)

# method used to try to load the seq_length evaluation data
# and compute statistics from them
evaluate_seq_len <- function()
{
  means <- NULL
  devs <- NULL
  seqlens <- NULL
  corrs <- NULL
  
  stepcorrs <- data.frame(step=1:100)
  
  # first we try to load each evaluation data file:
  for(i in seq(from=5, to=100, by=5))
  {
    print(paste0("Loading data for seqlen=",i))
    
    # Check if the correct_signs file exists:
    path <- paste0("misc/correct_signs_seq_",i,".csv")
    if(!file.exists(path))
    {
      next
    }
    
    print(paste("Loading file ", path))
    
    # Load the data:
    data <- read.csv(path,header=F)
    
    print(quantile(data$V1))
    means <- c(means,mean(data$V1))
    devs <- c(devs, sd(data$V1))
    seqlens <- c(seqlens,i)

    # For each data seq we should also study the overall correlation of the prediction
    # with the labels:
    path <- paste0("misc/eval_results_seq_",i,".csv")
    if(file.exists(path))
    {
      print(paste("Loading file ", path))
      
      # Load the data:
      data <- read.csv(path,header=T)
      
      corrs <- c(corrs,cor(data$prediction,data$label))
      
      # we should also study the correlation per evaluation step
      pstp <- ddply(data,"eval_index",summarize,correlation=cor(prediction,label))
      
      stepcorrs[paste0("seq_",i)] <- pstp$correlation
    }
    else {
      corrs <- c(corrs,NA)
    }
    
  }
    
  data <- data.frame(seqlen=seqlens,means=means,devs=devs,correlations=corrs)
  list(stats=data,stepcorrs=stepcorrs)
}
