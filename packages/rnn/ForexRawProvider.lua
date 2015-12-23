local Class = createClass{name="ForexRawProvider",bases={"rnn.ProviderBase","rnn.ForexRawHandler"}};

--[[
Class: utils.ForexRawProvider

Helper class used to create a Forex Loader

This class inherits from <rnn.ProviderBase>.
]]

--[[
Function: new

New method.
]]
function Class:new(options)
  CHECK(options.forcast_offset,"Invalid forcast offset")
  CHECK(options.num_forcast_classes,"Invalid forcast classes")
  CHECK(options.max_forcast_range,"Invalid max forcast range")
  CHECK(options.data_dir,"Invalid data dir")
  
  self._dataDir = options.data_dir

  -- Build the raw config table:
  self._rcfg = {
    offset = options.forcast_offset,
    numClasses = options.num_forcast_classes,
    sigmaRange = options.max_forcast_range,

  -- Index of the column that should be forcasted from the raw inputs dataset:
  -- EURUSD is the first symbol, and we have the weektime and daytime columns
  -- before that, so by default we want to forcast the column 6
    forcastIndex = 6,
  }

  -- self:debug("Rcfg=", self._rcfg)
end

--[=[
--[[
Constructor: ForexRawProvider

Create a new instance of the class.
]]
function ForexRawProvider(options)
]=]
function Class:initialize(options)
  self:debug("Creating a Forex Loader instance.")
end

--[[
Function: setup

Re-implementation of setup method
]]
function Class:setup(options)
  local features, labels = self:loadDataset()

  local bsize = self.batch_size
  local seq_len = self.seq_length

  self._inputSize = features:size(2)
  self._outputSize = self._rcfg.numClasses

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
  labels = labels:narrow(2,1,1)

  self:prepareBatches(features,labels)  
end


--[[
Function: addCheckpointData

Add data to checkpoint before saving it
]]
function Class:addCheckpointData(checkpoint)
  checkpoint.config = self._rcfg
end

return Class


