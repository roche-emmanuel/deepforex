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
end

return Class()
