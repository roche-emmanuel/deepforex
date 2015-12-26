local Class = createClass{name="Utils",bases={"base.Object"}};

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'utils.misc'

--[[
Class: rnn.Utils

Utils class for RNN construction

This class inherits from <base.Object>.
]]

--[=[
--[[
Constructor: Utils

Create a new instance of the class.

Parameters:
   No parameter
]]
function Utils(options)
]=]
function Class:initialize(options)
	self:debug("Creating RNN Utils.")
	-- We use float tensors in this project:
	torch.setdefaulttensortype('torch.FloatTensor')
end


--[[
Function: setupGPU

Setup for the common GPU elements
]]
function Class:setupGPU(opt)
	self:debug("Setting up RNN GPU support.")
	CHECK(opt.seed~=nil,"Invalid seed value")
	CHECK(opt.gpuid~=nil,"Invalid gpuid value")
	CHECK(opt.opencl~=nil,"Invalid opencl value")
	
	torch.manualSeed(opt.seed)

	-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
	if opt.gpuid >= 0 and opt.opencl == 0 then
		self:debug("Initializing GPU support with CUDA...")
	  local ok, cunn = pcall(require, 'cunn')
	  local ok2, cutorch = pcall(require, 'cutorch')
	  if not ok then self:warn('package cunn not found!') end
	  if not ok2 then self:warn('package cutorch not found!') end
	  if ok and ok2 then
	    self:debug('using CUDA on GPU ' .. opt.gpuid .. '...')
	    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
	    cutorch.manualSeed(opt.seed)
	  else
	    self:warn('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
	    self:warn('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
	    self:warn('Falling back on CPU mode')
	    opt.gpuid = -1 -- overwrite user setting
	  end
	end

	-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
	if opt.gpuid >= 0 and opt.opencl == 1 then
		self:debug("Initializing GPU support with OpenCL...")
	  local ok, cunn = pcall(require, 'clnn')
	  local ok2, cutorch = pcall(require, 'cltorch')
	  if not ok then self:warn('package clnn not found!') end
	  if not ok2 then self:warn('package cltorch not found!') end
	  if ok and ok2 then
	    self:debug('using OpenCL on GPU ' .. opt.gpuid .. '...')
	    cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
	    torch.manualSeed(opt.seed)
	  else
	    self:warn('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
	    self:warn('Check your OpenCL driver installation, check output of clinfo command, and try again.')
	    self:warn('Falling back on CPU mode')
	    opt.gpuid = -1 -- overwrite user setting
	  end
	end
end

--[[
Function: createLSTM

Method used to create an LSTM prototype
]]
function Class:createLSTM(opt)
	CHECK(opt.num_layers,"Invalid num_layers")
	CHECK(opt.rnn_size,"Invalid rnn_size")
	CHECK(opt.dropout,"Invalid dropout")
	CHECK(opt.num_inputs,"Invalid num_inputs")
	CHECK(opt.num_outputs,"Invalid num_outputs")

	local input_size = opt.num_inputs
	local output_size = opt.output_size
	local rnn_size = opt.rnn_size
	local n = opt.num_layers
	local dropout = opt.dropout

	self:debug("Number of LSTM inputs: ".. opt.num_inputs)
	self:debug("Number of LSTM outputs: ".. opt.num_outputs)

  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
      -- x = OneHot(input_size)(inputs[1])
      x = nn.Identity()(inputs[1])
      input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  
  local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
  
  if output_size == 1 then
    -- after the dropout, we need to compute a linear single value,
    -- and apply a sigmoid on it to get value in the range (0,1)
    local final = nn.Sigmoid()(proj)
    table.insert(outputs, final)
  else
    local logsoft = nn.LogSoftMax()(proj)
    table.insert(outputs, logsoft)
  end

  return nn.gModule(inputs, outputs)
end

--[[
Function: createPrototype

Method used to create the RNN prototype
]]
function Class:createPrototype(opt)
	CHECK(opt.model=="lstm","Invalid RNN model")

	self:debug('Creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
	
	local protos = {}
	if opt.model == 'lstm' then
	  protos.rnn = self:createLSTM(opt)
	else
		self:throw("Unsupported RNN model: ", opt.model)
	end

	if opt.num_outputs == 1 then
		protos.criterion = nn.MSECriterion()	
	else
		protos.criterion = nn.ClassNLLCriterion()
	end

	-- ship the model to the GPU if desired
	if opt.gpuid >= 0 and opt.opencl == 0 then
	  for k,v in pairs(protos) do 
	    -- log:debug("Converting ",k," to GPU...")
	    v:cuda() 
	  end
	end

	if opt.gpuid >= 0 and opt.opencl == 1 then
	  for k,v in pairs(protos) do 
	    v:cl() 
	  end
	end
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

--[[
Function: loadRawInputs

Method used to load the raw inputs
]]
function Class:loadRawInputs(opt,fname)
	CHECK(opt.data_dir,"Invalid data_dir")
	local dataDir = opt.data_dir

	local dcfg = dofile(path.join(dataDir, 'dataset.lua'))

	fname = fname or "raw_inputs"

	local src = path.join(dataDir, fname ..".csv")
	local dest = path.join(dataDir, fname ..".t7")

	if not path.exists(dest) or self:isUpdateRequired(src,dest) then
		self:debug('Preprocessing: generating file ', dest, '...')
	  local timer = torch.Timer()

	  self:debug('Preprocessing ',dcfg.num_samples,' samples...')

	  local data = self:readCSV(src,dcfg.num_samples,dcfg.num_inputs)
	  
	  -- save the tensor to file:
	  torch.save(dest, data)

	  self:debug('Preprocessing completed in ' .. timer:time().real .. ' seconds')
	end

	return torch.load(dest)
end

--[[
Function: getNumSymbols

Retrieve the number of symbols considered in a given raw inputs dataset
]]
function Class:getNumSymbols(inputs)
  local np = inputs:size(2) - 2
  return np/4	
end

--[[
Function: normalizeTimes

Method called to normalize the times
]]
function Class:normalizeTimes(features)
  self:debug("Normalizing times...")
  local daylen = 24*60
  local weeklen = 5*daylen

  features[{{},1}] = (features[{{},1}]/weeklen - 0.5)*2.0
  features[{{},2}] = (features[{{},2}]/daylen - 0.5)*2.0  
end

--[[
Function: generateLogReturnFeatures

Method used to generate the log returns features from a raw_inputs tensor
]]
function Class:generateLogReturnFeatures(opt,prices)
  self:debug("Generating log return features")

  -- Retrive the number of symbols:
  local nsym = self:getNumSymbols(prices)

  print("Initial prices: ", prices:narrow(1,1,10))

  -- retrive the raw number of samples:
  local nrows = prices:size(1)

  -- for each symbol we keep on the close prices,
  -- So we prepare a new tensor of the proper size:
  local features = torch.Tensor(nrows,2+nsym)

  -- populate this new tensor:
  -- copy the week and day times:
  features[{{},{1,2}}] = prices[{{},{1,2}}]

  -- copy the close prices:
  local offset = 2
  for i=1,nsym do
    features[{{},offset+i}] = prices[{{},offset+4*i}]
  end

  print("Initial features: ", features:narrow(1,1,10))

  -- Convert the prices to log returns:
  self:debug("Converting prices to returns...")
  local fprices = features:narrow(2,3,nsym)

  local rets = torch.cdiv(fprices[{{2,-1},{}}],fprices[{{1,-2},{}}])
  fprices[{{2,-1},{}}] = rets
  fprices[{1,{}}] = 1.0

  print("Initial returns: ", features:narrow(1,1,10))

  self:debug("Taking log of returns...")
  fprices:log()

  print("Log returns: ", features:narrow(1,1,10))

  -- remove the first line of the features:
  features = features:sub(2,-1)

  print("Removed the first line: ", features:narrow(1,1,10))

  self:debug("Normalizing log returns...")
  opt.price_means = opt.price_means or {}
  opt.price_sigmas = opt.price_sigmas or {}

  for i=1,nsym do
    local cprice = features:narrow(2,offset+i,1)
    
    local cmean = opt.price_means[i] or cprice:mean(1):storage()[1]
    local csig = opt.price_sigmas[i] or cprice:std(1):storage()[1]

    self:debug("Symbol ",i," : log return mean=",cmean,", sigma=",csig)

    opt.price_means[i] = cmean
    opt.price_sigmas[i] = csig

    cprice[{}] = (cprice-cmean)/csig
  end

  print("Normalized log returns: ", features:narrow(1,1,10))

  -- Apply sigmoid transformation:
  local cprice = features:narrow(2,offset+1,nsym)

  cprice[{}] = torch.pow(torch.exp(-cprice)+1.0,-1.0)

  print("Sigmoid transformed log returns: ", features:narrow(1,1,10))

  -- Now apply normalization:
  self:normalizeTimes(features)
  print("Normalized times: ", features:narrow(1,1,10))

  return features
end

--[[
Function: generateLogReturnLabels

Method used to generate the labels from the features
]]
function Class:generateLogReturnLabels(opt, features)
  self:debug("Generating log return features")
  CHECK(opt.forcast_symbol,"Invalid forcast_symbol")
  CHECK(opt.num_classes,"Invalid num_classes")

  -- Now generate the desired labels:
  -- The label is just the next value of the sigmoid transformed log returns
  -- labels will be taken from a given symbol index:

  self:debug("Forcast symbol index: ", opt.forcast_symbol)

  local offset = 2
  local idx = offset+opt.forcast_symbol
  local labels = features:sub(2,-1,idx,idx)

  --  Should remove the last row from the features:
  features = features:sub(1,-2)

  print("Generated log return labels: ", labels:narrow(1,1,10))

  if opt.num_classes > 1 then
    labels = self:generateClasses(labels,0,1,opt.num_classes)
  end

  print("Labels classes: ", labels:narrow(1,1,10))
  print("Final features: ", features:narrow(1,1,10))

  return features, labels	
end

--[[
Function: generateClasses

Method used to generated classes for a given vector
]]
function Class:generateClasses(labels,rmin,rmax,nclasses)
  -- We now have normalized labels,
  -- but what we are interested in is in classifying the possible outputs:
  -- First we clamp the data with the max range (as number of sigmas):
  -- we assume here that we can replace the content of the provided vector.
  local eps = 1e-6
  labels = labels:clamp(rmin,rmax - eps)

  -- Now we cluster the labels in the different classes:
  self:debug("Number of classes: ", nclasses)
  local range = rmax - rmin
  self:debug("Labels range: ", range)
  local classSize = range/nclasses
  self:debug("Label class size: ", classSize)

  -- The labels classes should be 1 based, so we add 1 below:
  labels = torch.floor((labels - rmin)/classSize) + 1  

  return labels
end

return Class()
