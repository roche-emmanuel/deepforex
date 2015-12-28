library(dplyr)

#method used to evaluate the prediction efficiency considering only a 
# certain level of prediction input:
evaluate_predictions <- function(nsteps = 50)
{
  levels <- seq(from=0.0, to=0.9, by=0.1)
  evals <- data.frame(siglevel=levels)
  
  # first we try to load each evaluation data file:
  for(i in seq(from=5, to=100, by=5))
  {
    print(paste0("Checking data for seqlen=",i))
    
    # For each data seq we should also study the overall correlation of the prediction
    # with the labels:
    path <- paste0("misc/eval_results_seq_",i,".csv")
    if(file.exists(path))
    {
      print(paste("Loading file ", path))
      
      # Load the data:
      data <- read.csv(path,header=T)
      
      # from the predictions and the labels we must substract 0.5 and mult by 2:
      data2 <- mutate(data, pred = (prediction-0.5)*2, lbl = (label-0.5)*2)
      
      # only keep the data under the step given by nsteps
      data2 <- filter(data2, eval_index <= nsteps)
      
      goodsign <- NULL
      corr <- NULL
      counts <- NULL
      
      # now filter by the threshold
      for(j in levels)
      {
        # filter the data by keeping only the "high level" predictions
        filt <- filter(data2,abs(pred)>j)
        
        # then compute the desired statistics on this subset:
        good <- filt$pred * filt$lbl > 0
        
        ratio <- mean(good)
        
        counts <- c(counts,dim(filt)[1])
        goodsign <- c(goodsign,ratio)
        corr <- c(corr, cor(filt$prediction,filt$label))
      }
      
      evals[paste0("seq_",i,"_count")] <- counts
      evals[paste0("seq_",i,"_mean")] <- goodsign
      evals[paste0("seq_",i,"_corr")] <- corr
    }
  }
  
  evals
}