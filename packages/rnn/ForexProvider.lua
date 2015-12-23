local Class = createClass{name="ForexProvider",bases={"rnn.ProviderBase","rnn.ForexDatasetHandler"}};

--[[
Class: utils.ForexProvider

Helper class used to create a Forex Loader

This class inherits from <rnn.ProviderBase>.
]]

--[=[
--[[
Constructor: ForexProvider

Create a new instance of the class.
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
  local features, labels = self:loadDataset()

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

return Class


