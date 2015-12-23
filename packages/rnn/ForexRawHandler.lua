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

  CHECK(options.forcast_offset,"Invalid forcast offset")
  self._forcastOffset = options.forcast_offset

  CHECK(options.num_forcast_classes,"Invalid forcast classes")
  self._numClasses = options.num_forcast_classes

  CHECK(options.max_forcast_range,"Invalid max forcast range")
  self._maxRange = options.max_forcast_range

  -- Index of the column that should be forcasted from the raw inputs dataset:
  -- EURUSD is the first symbol, and we have the weektime and daytime columns
  -- before that, so by default we want to forcast the column 6
  self._forcastIndex = 6
end

--[=[
--[[
Constructor: ForexRawHandler

Create a new instance of the class.
]]
function ForexRawHandler(options)
]=]
function Class:initialize(options)
  self._checkData = {}
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
  local features = torch.load(self.features_file)
  
  local timer = torch.Timer()
  self:debug("Starting with ", features:size(1), " samples.")

  -- From this feature tensor we must extract the labels using the forcastOffset:
  local cprices = features:narrow(2,self._forcastIndex,1)
  local labels, lmean, lsig = self:generateLabels{
    prices=cprices,
    offset=self._forcastOffset,
    sigmaRange=self._maxRange,
    numClasses=self._numClasses,
  }

  -- Also only keep the valid raw samples from the features:
  features = features:sub(1,labels:size(1),1,-1)

  -- Now apply normalization:
  self:debug("Normalizing times...")
  local daylen = 24*60
  local weeklen = 5*daylen

  features[{{},1}] = (features[{{},1}]/weeklen - 0.5)*2.0
  features[{{},2}] = (features[{{},2}]/daylen - 0.5)*2.0

  self:debug("Normalizing prices...")

  -- to normalize the prices we should only consider the close prices:
  local np = features:size(2) - 2
  local nsym = np/4

  --  Store the labels mean/sig in the checkpoint data:
  self._checkData.labels_mean = lmean
  self._checkData.labels_sigma = lsig
  self._checkData.num_classes = self._numClasses
  self._checkData.max_range = self._maxRange
  self._checkData.forcast_offset = self._forcastOffset
  self._checkData.forcast_index = self._forcastIndex
  self._checkData.num_symbols = nsym
  self._checkData.price_means = {}
  self._checkData.price_sigmas = {}
  

  self:debug("Processing with ",nsym," symbols...")
  local offset = 2
  local c

  for i=1,nsym do
    -- local cprice = features[{{},offset+4*i}]
    local cprice = features:narrow(2,offset+4*i,1)
    local cmean = cprice:mean(1):storage()[1]
    local csig = cprice:std(1):storage()[1]

    table.insert(self._checkData.price_means,cmean)
    table.insert(self._checkData.price_sigmas,csig)

    local y = features:narrow(2,offset+1+4*(i-1),4)
    y[{}] = (y-cmean)/csig

    -- for j=1,4 do
    --   c = offset+j+4*(i-1)
    --   features[{{},c}] = (features[{{},c}]-cmean)/csig
    -- end
  end

  -- print("Features content: ", features:narrow(1,1,10))

  self:debug('Done processing features in ' .. timer:time().real .. ' seconds')

  return features, labels    
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
  local offset = options.offset
  CHECK(offset,"Invalid offset")
  local sigRange = options.sigmaRange
  CHECK(sigRange,"Invalid sigmaRange")
  local nunClasses = options.numClasses
  CHECK(numClasses,"Invalid numClasses")

  local lmean = options.mean
  local lsig = options.sigma

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

  self:debug("Normalising labels: mean=",lmean,", sigma=",lsig)
  labels = (labels - lmean)/lsig

  -- from this point we have to consider that lsig=1.0
  lsig = 1.0

  -- We now have normalized labels,
  -- but what we are interested in is in classifying the possible outputs:
  -- First we clamp the data with the max range (as number of sigmas):
  local eps = 1e-6
  labels = labels:clamp(-lsig*sigRange,lsig*sigRange - eps)

  -- Now we cluster the labels in the different classes:
  self:debug("Number of classes: ", numClasses)
  local range = (2*lsig*sigRange)
  self:debug("Labels range: ", range)
  local classSize = range/numClasses
  self:debug("Label class size: ", classSize)

  -- The labels classes should be 1 based, so we add 1 below:
  labels = torch.floor((labels - (-lsig * sigRange))/classSize) + 1

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


