local Class = createClass{name="ForexRawHandler",bases={"rnn.FileReaderWriter"}};

--[[
Class: utils.ForexRawHandler

This class is used to read raw inputs from csv file, 
This class inherits from <base.Object>.
]]

--[[
Function: new

]]
function Class:new(options)
  CHECK(options.data_dir,"Invalid data dir")
  self._dataDir = options.data_dir
end

--[=[
--[[
Constructor: ForexRawHandler

Create a new instance of the class.
]]
function ForexRawHandler(options)
]=]
function Class:initialize(options)

end

--[[
Function: loadDataset

Method used to load the dataset
]]
function Class:loadDataset()
  -- Load the dataset config:
  self.dcfg = dofile(path.join(self._dataDir, 'dataset.lua'))
  -- self:debug("Loaded dataset config: ", self.dcfg)

  -- prepare the tensor files:
  self.inputs_file = path.join(self._dataDir, 'raw_inputs.csv')
  
  self.features_file = path.join(self._dataDir, 'raw_inputs.t7')

  -- preprocess the data if required:
  if self:isPreprocessingRequired() then
    self:preprocessDataset()
  end

  -- load the data:
  self:debug("Loading data...")
  local prices = torch.load(self.features_file)
  local features, labels

  local timer = torch.Timer()
  self:debug("Starting with ", prices:size(1), " samples.")

  -- if we use log returns we should convert the features here:
  if self._rcfg.use_log_returns then
    features, labels = self:generateLogReturnDataset(prices)
  else
    features, labels = self:generatePriceDataset(prices)
  end

  -- print("Features content: ", features:narrow(1,1,10))
  self:debug('Done processing features in ' .. timer:time().real .. ' seconds')

  return features, labels    
end

--[[
Function: generateLogReturnDataset

Method to generate log returns from the raw prices
]]
function Class:generateLogReturnDataset(prices)
  self:debug("Generating log return prices")

  -- Retrive the number of symbols:
  local nsym = self:getNumSymbols(prices)

  print("Initial prices: ", prices:narrow(1,1,10))

  -- retrive the raw number of samples:
  local nrows = prices:size(1)

  -- for each symbol we keep on the close prices,
  -- So we prepare a new tensor of the proper size:
  local features = torch.Tensor(nrows,2+nsym)

  -- populate this new tensor:
  -- copy the week and day times:
  features[{{},{1,2}}] = prices[{{},{1,2}}]

  -- copy the close prices:
  local offset = 2
  for i=1,nsym do
    features[{{},offset+i}] = prices[{{},offset+4*i}]
  end

  print("Initial features: ", features:narrow(1,1,10))

  -- Convert the prices to log returns:
  self:debug("Converting prices to returns...")
  local fprices = features:narrow(2,3,nsym)

  local rets = torch.cdiv(fprices[{{2,-1},{}}],fprices[{{1,-2},{}}])
  fprices[{{2,-1},{}}] = rets
  fprices[{1,{}}] = 1.0

  print("Initial returns: ", features:narrow(1,1,10))

  self:debug("Taking log of returns...")
  fprices:log()

  print("Log returns: ", features:narrow(1,1,10))

  -- remove the first line of the features:
  features = features:sub(2,-1)

  print("Removed the first line: ", features:narrow(1,1,10))

  self:debug("Normalizing log returns...")
  self._rcfg.price_means = self._rcfg.price_means or {}
  self._rcfg.price_sigmas = self._rcfg.price_sigmas or {}

  for i=1,nsym do
    local cprice = features:narrow(2,offset+i,1)
    
    local cmean = self._rcfg.price_means[i] or cprice:mean(1):storage()[1]
    local csig = self._rcfg.price_sigmas[i] or cprice:std(1):storage()[1]

    self:debug("Symbol ",i," : log return mean=",cmean,", sigma=",csig)

    self._rcfg.price_means[i] = cmean
    self._rcfg.price_sigmas[i] = csig

    cprice[{}] = (cprice-cmean)/csig
  end

  print("Normalized log returns: ", features:narrow(1,1,10))

  -- Apply sigmoid transformation:
  local cprice = features:narrow(2,offset+1,nsym)

  cprice[{}] = torch.pow(torch.exp(-cprice)+1.0,-1.0)

  print("Sigmoid transformed log returns: ", features:narrow(1,1,10))

  -- Now apply normalization:
  self:normalizeTimes(features)
  print("Normalized times: ", features:narrow(1,1,10))

  -- Now generate the desired labels:
  -- The label is just the next value of the sigmoid transformed log returns
  -- labels will be taken from a given symbol index:

  self:debug("Forcast symbol index: ", self._rcfg.forcast_symbol)

  local idx = offset+self._rcfg.forcast_symbol
  local labels = features:sub(2,-1,idx,idx)

  --  Should remove the last row from the features:
  features = features:sub(1,-2)

  print("Generated log return labels: ", labels:narrow(1,1,10))

  if self._rcfg.numClasses > 1 then
    labels = self:generateClasses(labels,0,1,self._rcfg.numClasses)
  end

  print("Labels classes: ", labels:narrow(1,1,10))
  print("Final features: ", features:narrow(1,1,10))

  return features, labels
end

--[[
Function: getNumSymbols

Retrieve the number of symbols available
]]
function Class:getNumSymbols(features)
  local np = features:size(2) - 2
  local nsym = np/4

  if self._rcfg.num_symbols then
    CHECK(self._rcfg.num_symbols == nsym,"Mismatch in number of symbols")
  end

  self._rcfg.num_symbols = nsym

  return nsym
end

--[[
Function: generateClasses

Method used to cut a given vector into classes
]]
function Class:generateClasses(labels,rmin,rmax,nclasses)
  -- We now have normalized labels,
  -- but what we are interested in is in classifying the possible outputs:
  -- First we clamp the data with the max range (as number of sigmas):
  -- we assume here that we can replace the content of the provided vector.
  local eps = 1e-6
  labels = labels:clamp(rmin,rmax - eps)

  -- Now we cluster the labels in the different classes:
  self:debug("Number of classes: ", nclasses)
  local range = rmax - rmin
  self:debug("Labels range: ", range)
  local classSize = range/nclasses
  self:debug("Label class size: ", classSize)

  -- The labels classes should be 1 based, so we add 1 below:
  labels = torch.floor((labels - rmin)/classSize) + 1  

  return labels
end

--[[
Function: generatePriceDataset

Simple method used to generate a first version of the features/labels tensors
]]
function Class:generatePriceDataset(features)
  self:debug("Generating simple price features")
  -- From this feature tensor we must extract the labels using the forcastOffset:
  local cprices = features:narrow(2,2+self._rcfg.forcast_symbol*4,1)
  local labels = self:generateLabels{prices=cprices}

  -- Also only keep the valid raw samples from the features:
  features = features:sub(1,labels:size(1),1,-1)

  -- Now apply normalization:
  self:normalizeTimes(features)

  self:normalizePrices(features)

  return features, labels
end

--[[
Function: normalizePrices

Method called to normalize the prices
]]
function Class:normalizePrices(features)
  self:debug("Normalizing prices...")

  -- to normalize the prices we should only consider the close prices:
  local nsym = self:getNumSymbols(features)

  --  Store the labels mean/sig in the checkpoint data:
  self._rcfg.price_means = self._rcfg.price_means or {}
  self._rcfg.price_sigmas = self._rcfg.price_sigmas or {}


  self:debug("Processing with ",nsym," symbols...")
  local offset = 2
  local c

  for i=1,nsym do
    -- local cprice = features[{{},offset+4*i}]
    local cprice = features:narrow(2,offset+4*i,1)
    local cmean = self._rcfg.price_means[i] or cprice:mean(1):storage()[1]
    local csig = self._rcfg.price_sigmas[i] or cprice:std(1):storage()[1]

    self._rcfg.price_means[i] = cmean
    self._rcfg.price_sigmas[i] = csig

    local y = features:narrow(2,offset+1+4*(i-1),4)
    y[{}] = (y-cmean)/csig

    -- for j=1,4 do
    --   c = offset+j+4*(i-1)
    --   features[{{},c}] = (features[{{},c}]-cmean)/csig
    -- end
  end  
end

--[[
Function: normalizeTimes

Method called to normalize the times
]]
function Class:normalizeTimes(features)
  self:debug("Normalizing times...")
  local daylen = 24*60
  local weeklen = 5*daylen

  features[{{},1}] = (features[{{},1}]/weeklen - 0.5)*2.0
  features[{{},2}] = (features[{{},2}]/daylen - 0.5)*2.0  
end

--[[
Function: generateLabels

Method used to generate the labels from a given tensor of prices
and a forcast offset

Parameters:
]]
function Class:generateLabels(options)
  local prices = options.prices
  CHECK(prices,"Invalid prices")
  local offset = options.offset or self._rcfg.offset
  CHECK(offset,"Invalid offset")
  local sigRange = options.sigmaRange or self._rcfg.sigmaRange
  CHECK(sigRange,"Invalid sigmaRange")
  local numClasses = options.numClasses or self._rcfg.numClasses
  CHECK(numClasses,"Invalid numClasses")

  local lmean = options.mean or self._rcfg.lmean
  local lsig = options.sigma or self._rcfg.lsig

  CHECK(prices:nDimension()==1 or prices:size(2)==1, "Invalid prices dimensions.")
  
  -- remove the first (self.offset-1) elements:
  self:debug("Generating labels with offset of ", offset)
  local labels = prices:sub(offset+1,-1) - prices:sub(1,-offset-1)

  -- do the same for the features:
  -- except that we remove the end of the tensor
  local len = labels:size(1)
  self:debug("Keeping ", len, " valid samples.")

  lmean = lmean or labels:mean(1):storage()[1]
  lsig = lsig or labels:std(1):storage()[1]

  self._rcfg.lmean = lmean
  self._rcfg.lsig = lsig

  self:debug("Normalising labels: mean=",lmean,", sigma=",lsig)
  labels = (labels - lmean)/lsig

  -- from this point we have to consider that lsig=1.0
  lsig = 1.0

  labels = self:generateClasses(labels,-lsig*sigRange,lsig*sigRange,numClasses)

  -- print("labels content: ", labels:narrow(1,1,100))
  -- self:writeTensor("labels.csv", labels)
  return labels, lmean, lsig
end

--[[
Function: isPreprocessingRequired

Method used to check if preprocessing is required
to build the tensor files from the raw inputs
]]
function Class:isPreprocessingRequired()
  -- fetch file attributes to determine if we need to rerun preprocessing
  if not path.exists(self.features_file) then
    -- prepro files do not exist, generate them
    self:debug('Tensor files do not exist. Running preprocessing...')
    return true
  else
    -- check if the input file was modified since last time we 
    -- ran the prepro. if so, we have to rerun the preprocessing
    if self:isUpdateRequired(self.inputs_file,self.features_file) then
      self:debug(self.features_file,' detected as stale. Re-running preprocessing...')
      return true
    end
  end

  return false  
end

--[[
Function: preprocessDataset

Method used to convert from raw input data to tensors
]]
function Class:preprocessDataset()
  local timer = torch.Timer()

  self:debug('Preprocessing inputs...')
  local data = self:readCSV(self.inputs_file,self.dcfg.num_samples,self.dcfg.num_inputs)
  -- save the tensor to file:
  torch.save(self.features_file, data)

  self:debug('Preprocessing completed in ' .. timer:time().real .. ' seconds')
end

return Class


