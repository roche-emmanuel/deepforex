local Class = createClass{name="CharProvider",bases={"rnn.ProviderBase"}};

--[[
Class: utils.CharProvider

Helper class used to reproduce the loading achieved in the original char-rnn project

In addition to the initial implementation, this version will also convert the
input char into "one hot" vector to be able to use the same  pipeline as with the 
ForexProvider

This class inherits from <base.Object>.
]]

--[=[
--[[
Constructor: CharProvider

Create a new instance of the class.

Parameters:
  data_dir - directory where the raw input data is found
  batch_size - Size of the mini batch to use
  split_fractions - repartition of the data between training/validation/test
]]
function CharProvider(options)
]=]
function Class:initialize(options)
  self:debug("Creating a Char Loader instance.")
end

--[[
Function: setup

Re-implementation of setup method
]]
function Class:setup(options)
  -- prepre the tensor files:
  self.input_file = path.join(self.data_dir, 'input.txt')
  self.vocab_file = path.join(self.data_dir, 'vocab.t7')
  self.tensor_file = path.join(self.data_dir, 'data.t7')

  -- preprocess the data if required:
  if self:isPreprocessingRequired() then
    self:preprocessDataset()
  end

  -- load the data:
  self:debug('Loading data files...')
  local data = torch.load(self.tensor_file)
  self.vocab_mapping = torch.load(self.vocab_file)


  local bsize = self.batch_size
  local seq_len = self.seq_length

  -- cut off the end so that it divides evenly
  local len = data:size(1)
  if len % (bsize * seq_len) ~= 0 then
    self:debug('Cutting off end of data so that the batches/sequences divide evenly')
    data = data:sub(1, bsize * seq_len * math.floor(len / (bsize * seq_len)))
  end

  -- count vocab
  self.vocab_size = 0
  for _ in pairs(self.vocab_mapping) do 
      self.vocab_size = self.vocab_size + 1 
  end

  self._inputSize = self.vocab_size
  self._outputSize = self.vocab_size

  -- prepare the label vector:
  local ydata = data:clone()
  ydata:sub(1,-2):copy(data:sub(2,-1))
  ydata[-1] = data[1]

  -- expand the features and labels vector into one hot matrices:
  local nrows = data:size(1)

  local features = torch.Tensor(nrows,self.vocab_size):zero()
  local labels = torch.Tensor(nrows,self.vocab_size):zero()

  for i=1,nrows do
    features[{i,data[i]}] = 1
    labels[{i,ydata[i]}] = 1
  end

  self:prepareBatches(features,labels)  
end

--[[
Function: isPreprocessingRequired

Method used to check if preprocessing is required
to build the tensor files from the raw inputs
]]
function Class:isPreprocessingRequired()
  -- fetch file attributes to determine if we need to rerun preprocessing
  if not (path.exists(self.vocab_file) and path.exists(self.tensor_file)) then
    -- prepro files do not exist, generate them
    self:debug('Tensor files do not exist. Running preprocessing...')
    return true
  else
    -- check if the input file was modified since last time we 
    -- ran the prepro. if so, we have to rerun the preprocessing
    if self:isUpdateRequired(self.input_file,self.vocab_file) then
      self:debug(self.vocab_file,' detected as stale. Re-running preprocessing...')
      return true
    end

    if self:isUpdateRequired(self.input_file,self.tensor_file) then
      self:debug(self.tensor_file,' detected as stale. Re-running preprocessing...')
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

    self:debug('loading text file...')
    local cache_len = 10000
    local rawdata
    local tot_len = 0
    local f = assert(io.open(self.input_file, "r"))

    -- create vocabulary if it doesn't exist yet
    self:debug('creating vocabulary mapping...')
    -- record all characters to a set
    local unordered = {}
    rawdata = f:read(cache_len)
    repeat
        for char in rawdata:gmatch'.' do
            if not unordered[char] then unordered[char] = true end
        end
        tot_len = tot_len + #rawdata
        rawdata = f:read(cache_len)
    until not rawdata
    f:close()
    -- sort into a table (i.e. keys become 1..N)
    local ordered = {}
    for char in pairs(unordered) do ordered[#ordered + 1] = char end
    table.sort(ordered)
    -- invert `ordered` to create the char->int mapping
    local vocab_mapping = {}
    for i, char in ipairs(ordered) do
        vocab_mapping[char] = i
    end
    -- construct a tensor with all the data
    self:debug('putting data into tensor...')
    local data = torch.ByteTensor(tot_len) -- store it into 1D first, then rearrange
    f = assert(io.open(self.input_file, "r"))
    local currlen = 0
    rawdata = f:read(cache_len)
    repeat
        for i=1, #rawdata do
            data[currlen+i] = vocab_mapping[rawdata:sub(i, i)] -- lua has no string indexing using []
        end
        currlen = currlen + #rawdata
        rawdata = f:read(cache_len)
    until not rawdata
    f:close()

    -- save output preprocessed files
    self:debug('saving ' .. self.vocab_file)
    torch.save(self.vocab_file, vocab_mapping)
    self:debug('saving ' .. self.tensor_file)
    torch.save(self.tensor_file, data)

  self:debug('Preprocessing completed in ' .. timer:time().real .. ' seconds')
end

return Class


