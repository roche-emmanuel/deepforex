
# method used to evaluate a compound model

evaluate_compound_model <- function(files,nsteps=50)
{
  levels <- seq(from=0.0, to=0.9, by=0.1)
  evals <- data.frame(siglevel=levels)
  
  # prepare the resulting dataframe:
  df <- NULL
  
  prednames <- NULL
  
  # first we load all the dataframes:
  for(i in seq_along(files))
  {
    path <- paste0("misc/eval_results_",files[i],".csv")
    
    print(paste("Loading file ", path))

    # Load the data:
    data <- read.csv(path,header=T)
    
    if(is.null(df)) 
    {
      df <- data.frame(eval_index = data$eval_index, label = data$label)
    }

    pname <- paste0("pred_",i)
    df[pname] <- data$prediction
    prednames <- c(prednames,pname)
  }
  
  # Compute the mean prediction:
  df$meanPred <- rowMeans(df[prednames])
  
  # Now we compute the regular statistics:
  # only keep the data under the step given by nsteps
  df <- filter(df, eval_index <= nsteps)
  
  # from the predictions and the labels we must substract 0.5 and mult by 2:
  df <- mutate(df, pred = (meanPred-0.5)*2, lbl = (label-0.5)*2)
  
  goodsign <- NULL
  corr <- NULL
  counts <- NULL
  
  # now filter by the threshold
  for(j in levels)
  {
    # filter the data by keeping only the "high level" predictions
    filt <- filter(df,abs(pred)>j)
    
    # then compute the desired statistics on this subset:
    good <- filt$pred * filt$lbl > 0
    
    ratio <- mean(good)
    
    counts <- c(counts,dim(filt)[1])
    goodsign <- c(goodsign,ratio)
    corr <- c(corr, cor(filt$meanPred,filt$label))
  }
    
  evals[paste0("seq_25_count")] <- counts
  evals[paste0("seq_25_mean")] <- goodsign
  evals[paste0("seq_25_corr")] <- corr

  evals  
}
