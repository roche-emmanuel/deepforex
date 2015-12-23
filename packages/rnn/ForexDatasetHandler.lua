local Class = createClass{name="ForexDatasetHandler",bases={"base.Object"}};

--[[
Class: utils.ForexDatasetHandler

Helper class used to create a Forex Loader

This class inherits from <rnn.ProviderBase>.
]]

--[[
Function: new

]]
function Class:new(options)
  CHECK(options.data_dir,"Invalid data dir")
  self.data_dir = options.data_dir
end

--[=[
--[[
Constructor: ForexDatasetHandler

Create a new instance of the class.
]]
function ForexDatasetHandler(options)
]=]
function Class:initialize(options)

end

--[[
Function: loadDataset

Method used to load the dataset
]]
function Class:loadDataset()
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
  
  return features, labels    
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

--[[
Function: readCSV

Helper method used to read a CSV file into a tensor
]]
function Class:readCSV(filename, nrows, ncols)
  -- Read data from CSV to tensor
  local csvFile = io.open(filename, 'r')  
  local header = csvFile:read()

  local data = torch.Tensor(nrows, ncols)

  local i = 0  
  for line in csvFile:lines('*l') do  
    i = i + 1
    local l = line:split(',')
    for key, val in ipairs(l) do
      data[i][key] = val
    end
  end

  csvFile:close()

  return data
end

--[[
Function: isUpdateRequired

Helper method used to check if a given file used as source
was updated (eg. modified) after a product file was generated
from that source
]]
function Class:isUpdateRequired(src,prod)
  local src_attr = lfs.attributes(src)
  local prod_attr = lfs.attributes(prod)

  return src_attr.modification > prod_attr.modification
end

return Class


