local Class = createClass{name="ProviderBase",bases={"base.Object"}};

--[[
Class: utils.ProviderBase

Base Provider class.

This class inherits from <base.Object>.
]]

--[=[
--[[
Constructor: ProviderBase

Create a new instance of the class.

Parameters:
  data_dir - directory where the raw input data is found
  batch_size - Size of the mini batch to use
  split_fractions - repartition of the data between training/validation/test
]]
function ProviderBase(options)
]=]
function Class:initialize(options)
  CHECK(options.data_dir,"Invalid data dir")
  self.data_dir = options.data_dir

  CHECK(options.batch_size,"Invalid batch size")
  self.batch_size = options.batch_size

  CHECK(options.seq_length,"Invalid sequence length")
  self.seq_length = options.seq_length

  CHECK(options.split_fractions,"Invalid split fractions")
  self.split_fractions = options.split_fractions

  self:setup(options)

  self:setSplitSizes()
end

--[[
Function: setup

main setup function for this provider,
must be reimplemented by derived classes
]]
function Class:setup(options)
  self:no_impl()
end

--[[
Function: setSplitSizes

Method used to set the split sizes (once the nbatches variable is ready)
]]
function Class:setSplitSizes()
 -- Split the samples into train/eval/test:
  local split_fractions = self.split_fractions

  CHECK(self.nbatches,"Batches not available.")
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
  self:no_impl()
end

--[[
Function: getOutputSize

return the output size for the RNN
]]
function Class:getOutputSize()
  self:no_impl()
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

return Class


