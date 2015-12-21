local Class = createClass{name="ForexHandler",bases={"base.Object"}};

--[[
Class: utils.ForexHandler

Helper class used to create a Forex Loader

This class inherits from <base.Object>.
]]

--[=[
--[[
Constructor: ForexHandler

Create a new instance of the class.

Parameters:
  data_dir - directory where the raw input data is found
  batch_size - Size of the mini batch to use
  split_fractions - repartition of the data between training/validation/test
]]
function ForexHandler(options)
]=]
function Class:initialize(options)
  self:debug("Creating a Forex Loader instance.")

  CHECK(options.data_dir,"Invalid data dir")
  self.data_dir = options.data_dir

  CHECK(options.batch_size,"Invalid batch size")
  self.batch_size = options.batch_size

  CHECK(options.seq_length,"Invalid sequence length")
  self.seq_length = options.seq_length

  CHECK(options.split_fractions,"Invalid split fractions")
  self.split_fractions = options.split_fractions

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


  -- Perform the processing of the features and labels to build the
  -- input sequences for the training:
  -- self._features, self._labels = self:generateFeatures(features,labels)

  -- Cut off the end of the datasets to fit the requested batch size exactly:
  -- local len = self._features:size(1)
  -- CHECK(len==self._labels:size(1),"Mismatch in features and labels sizes")


  -- if len % bsize ~= 0 then
  --   self:debug('Cutting off end of data so that the batches divide evenly')
  --   self._features = self._features:sub(1, bsize * math.floor(len / (bsize)))
  --   self._labels = self._labels:sub(1, bsize * math.floor(len / (bsize)))
  -- end

  -- self:debug("Using final num samples: ", self._features:size(1))
  -- self.nbatches = self._features:size(1)/bsize

  -- Split the samples into train/eval/test:
  local split_fractions = self.split_fractions

  CHECK(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
  CHECK(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
  CHECK(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
  
  if split_fractions[3] == 0 then 
    -- catch a common special case where the user might not want a test set
    self.ntrain = math.floor(self.nbatches * split_fractions[1])
    self.nval = self.nbatches - self.ntrain
    self.ntest = 0
  else
    -- divide data to train/val and allocate rest to test
    self.ntrain = math.floor(self.nbatches * split_fractions[1])
    self.nval = math.floor(self.nbatches * split_fractions[2])
    self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
  end

  self.split_sizes = {self.ntrain, self.nval, self.ntest}
  self.batch_ix = {0,0,0}

  print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))

  collectgarbage()
end

--[[
Function: getInputSize

Retrieve the input size for this dataset features:
]]
function Class:getInputSize()
  return self.nfeatures
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
Function: resetBatchPointer

Reset the batch pointer for a given category (train,val or test)
]]
function Class:resetBatchPointer(split_index, batch_index)
  self.batch_ix[split_index] = batch_index or 0
end

--[[
Function: nextBatch

Retrieve the next batch pointers for a given category
]]
function Class:nextBatch(split_index)
  if self.split_sizes[split_index] == 0 then
    -- perform a check here to make sure the user isn't screwing something up
    local split_names = {'train', 'val', 'test'}
    print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
    os.exit() -- crash violently
  end

  -- split_index is integer: 1 = train, 2 = val, 3 = test
  self.batch_ix[split_index] = self.batch_ix[split_index] + 1
  if self.batch_ix[split_index] > self.split_sizes[split_index] then
      self.batch_ix[split_index] = 1 -- cycle around to beginning
  end

  -- pull out the correct next batch
  local ix = self.batch_ix[split_index]
  if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
  if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
  
  return self.x_batches[ix], self.y_batches[ix]

  -- local istart = (ix-1)*self.batch_size+1
  -- local iend = istart+self.batch_size-1
  -- local fea = self._features[{{istart,iend},{}}]
  -- local lbl = self._labels[{{istart,iend},{}}]
  -- return fea, lbl
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


