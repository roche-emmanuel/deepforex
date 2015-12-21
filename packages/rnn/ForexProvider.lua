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

  self.nfeatures = features:size(2)

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

  self:prepareBatches(features,labels)  
end

--[[
Function: getInputSize

Retrieve the input size for this dataset features:
]]
function Class:getInputSize()
  return self.nfeatures
end

--[[
Function: getOutputSize

return the output size for the RNN
]]
function Class:getOutputSize()
  return 1
end

--[[
Function: prepareBatches

Method used to prepare the X/Y batches
]]
function Class:prepareBatches(features,labels)
  local timer = torch.Timer()

  -- for each sequence we use seq_len samples row from the features
  -- in each batch we have batch_size sequences
  self.x_batches = {}
  self.y_batches = {}

  local bsize = self.batch_size
  local seq_len = self.seq_length
  local nbatches = features:size(1)/(bsize*seq_len)

  self.nbatches = nbatches
  
  -- lets try to be helpful here
  if self.nbatches < 50 then
      self:warn('less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
  end

  self:debug("Preparing batches...")

  local nf = features:size(2)
  -- we keep only one forcast per network:
  local nout = 1

  -- Select the forcast index we want to keep:
  local outid = 1 

  local offset = 0
  for i=1,nbatches do
    local xbatch = torch.Tensor(seq_len,bsize,nf)
    local ybatch = torch.Tensor(seq_len,bsize,nout)

    for t=1,seq_len do
      -- build a tensor corresponding to the sequence element t
      -- eg. we present all the rows of features in batch that arrive
      -- at time t in th sequence:
      local xmat = torch.Tensor(bsize,nf)
      local ymat = torch.Tensor(bsize,nout)

      -- fill the data for this tensor:
      for i=1,bsize do
        xmat[{i,{}}] = features[{offset+(i-1)*seq_len+t,{}}]
        ymat[{i,{}}] = labels[{offset+(i-1)*seq_len+t,outid}]
      end

      xbatch[t] = xmat
      ybatch[t] = ymat
    end

    table.insert(self.x_batches,xbatch)
    table.insert(self.y_batches,ybatch)
    
    offset = offset + bsize*seq_len
  end

  self:debug('Prepared batches in ', timer:time().real ,' seconds')  
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
Function: generateFeatures

Method used to turn the raw features inputs into sequences appropriated
for RNN input
]]
function Class:generateFeatures(features,labels)
  local timer = torch.Timer()

  local seq_len = self.seq_length
  self:debug("Generating features for sequence length of ",seq_len)

  -- First we prepare the final feature tensor:
  -- this tensor should contain the weektime, the daytime, and then the sequence
  -- of prices:
  -- Thus the total number of columns is given by:
  local stride = (features:size(2) - 2)
  local ncols = 2 + seq_len * stride

  -- The number of rows should be the same as the raw feature minus (seq_len-1)
  local nrows = features:size(1) - (seq_len-1)

  -- create the tensor:
  self:debug("Creating tensor of size ",nrows,"x",ncols)
  local data = torch.Tensor(nrows, ncols)

  -- First we can assign the weektime/daytime data, taken directly from the features tensor:
  data[{{},{1,2}}] = features[{{seq_len,-1},{1,2}}]

  -- Now fill the tensor with the additional sequence data:
  local offset = 3
  for i=1,seq_len do
    data[{{},{offset,offset+stride-1}}] = features[{{i,i+nrows-1},{3,3+stride-1}}]
    offset = offset + stride
  end

  -- print(data[{{1,10},{}}])
  -- torch.save("inputs/test_data.txt",data,"ascii")

  -- Also process the label data:
  -- we simply need to remove the first (seq_len-1) rows
  labels = labels:sub(seq_len,-1)

  self:debug('Feature generation completed in ' .. timer:time().real .. ' seconds')

  return data, labels
end

return Class


