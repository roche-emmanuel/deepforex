local Class = createClass{name="ForexRawProvider",bases={"rnn.ProviderBase","rnn.ForexRawHandler"}};

--[[
Class: utils.ForexRawProvider

Helper class used to create a Forex Loader

This class inherits from <rnn.ProviderBase>.
]]

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
  self._outputSize = self._numClasses

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
  checkpoint.feature_data = self._checkData
end

return Class


