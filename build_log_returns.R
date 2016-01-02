# method used to build the log returns of a price vector with a given offset:
build_log_returns <- function(cprices,offset)
{
  cprices <- as.numeric(cprices)
  len <- length(cprices)
  ret <- cprices[(1+offset):len] / cprices[1:(len-offset)]
  log(ret)
}