local Class = createClass{name="ForexProvider",bases={"rnn.ProviderBase"}};

--[[
Class: utils.ForexProvider

Helper class used to create a Forex Loader

This class inherits from <rnn.ProviderBase>.
]]

--[=[
--[[
Constructor: ForexProvider

Create a new instance of the class.

Parameters:
  data_dir - directory where the raw input data is found
  batch_size - Size of the mini batch to use
  split_fractions - repartition of the data between training/validation/test
]]
function ForexProvider(options)
]=]
function Class:initialize(options)
  self:debug("Creating a Forex Loader instance.")
end

--[[
Function: setup

Re-implementation of setup method
]]
function Class:setup(options)
  -- Load the dataset config:
  self.dcfg = dofile(path.join(self.data_dir, 'dataset.lua'))
  -- self:debug("Loaded dataset config: ", self.dcfg)

  -- prepre the tensor files:
  self.inputs_file = path.join(self.data_dir, 'inputs.csv')
  self.forcasts_file = path.join(self.data_dir, 'forcasts.csv')
  
  self.features_file = path.join(self.data_dir, 'features.t7')
  self.labels_file = path.join(self.data_dir, 'labels.t7')

  -- preprocess the data if required:
  if self:isPreprocessingRequired() then
    self:preprocessDataset()
  end

  -- load the data:
  self:debug("Loading data...")
  local features = torch.load(self.features_file)
  local labels = torch.load(self.labels_file)

  local bsize = self.batch_size
  local seq_len = self.seq_length

  self._inputSize = features:size(2)
  self._outputSize = 1

  -- cut off the end so that it divides evenly
  local len = features:size(1)
  CHECK(len== labels:size(1),"Mismatch in features and labels sizes")
  
  if len % (bsize * seq_len) ~= 0 then
    self:debug('Cutting off end of features so that the batches/sequences divide evenly')
    local len = bsize * seq_len * math.floor(len / (bsize * seq_len))
    features = features:sub(1, len)
    labels = labels:sub(1, len)
  end

  self:debug("Features dimensions: ",features:size(1),"x",features:size(2))

  -- select only the first column for the labels:
  -- labels = labels[{{},1}]
  labels = labels:narrow(2,1,1)

  self:prepareBatches(features,labels)  
end

--[[
Function: isPreprocessingRequired

Method used to check if preprocessing is required
to build the tensor files from the raw inputs
]]
function Class:isPreprocessingRequired()
  -- fetch file attributes to determine if we need to rerun preprocessing
  if not (path.exists(self.labels_file) and path.exists(self.features_file)) then
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

    if self:isUpdateRequired(self.forcasts_file,self.labels_file) then
      self:debug(self.labels_file,' detected as stale. Re-running preprocessing...')
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

  self:debug('Preprocessing forcasts...')
  local data = self:readCSV(self.forcasts_file,self.dcfg.num_samples,self.dcfg.num_forcasts)
  -- save the features file:
  torch.save(self.labels_file, data)

  self:debug('Preprocessing completed in ' .. timer:time().real .. ' seconds')
end

return Class


