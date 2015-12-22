local Class = createClass{name="Agent",bases={"base.Object"}};

local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'

local model_utils = require 'utils.model_utils'

--[[
Class: rnn.Agent

Base class for RNN agent implementation

This class inherits from <base.Object>.
]]

--[=[
--[[
Constructor: Agent

Create a new instance of the class.

Parameters:
	 No parameter
]]
function Agent(options)
]=]
function Class:initialize(options)
	CHECK(options.provider,"Invalid provider")
	self._provider = options.provider

	CHECK(options.config,"Invalid config")
	self._config = options.config

	self:createPrototype()

	self:createInitState()

	self:initParameters()

	self:generateClones()
end

--[[
Function: createPrototype

Method used to create the prototype of the network
]]
function Class:createPrototype()
	-- Create the model:
	local opt = self._config

	self:debug('Creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')

	local ni = self._provider:getInputSize()
	self:debug("Number of inputs: ".. ni)
	local no = self._provider:getOutputSize()
	self:debug("Number of outputs: ".. no)

	local protos = {}
	if opt.model == 'lstm' then
	  protos.rnn = LSTM.lstm(ni, no, opt.rnn_size, opt.num_layers, opt.dropout)
	elseif opt.model == 'gru' then
	  protos.rnn = GRU.gru(ni, opt.rnn_size, opt.num_layers, opt.dropout)
	elseif opt.model == 'rnn' then
	  protos.rnn = RNN.rnn(ni, opt.rnn_size, opt.num_layers, opt.dropout)
	end

	if no == 1 then
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

	self._prototype = protos
end

--[[
Function: createInitState

Create the init state that will be used to store the current
initial state of the RNN
]]
function Class:createInitState()
	self:debug("Creating init state.")

	local opt = self._config

	-- the initial state of the cell/hidden states
	local init_state = {}
	for L=1,opt.num_layers do
	  local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
	  if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
	  if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
	  table.insert(init_state, h_init:clone())
	  if opt.model == 'lstm' then
	    table.insert(init_state, h_init:clone())
	  end
	end

	self._initState = init_state
	self._initStateGlobal = clone_list(init_state)
end

--[[
Function: initParameters

Method called to perform parameter initialization
]]
function Class:initParameters()
	self:debug("Initializing parameters...")
	local opt = self._config

	-- put the above things into one flattened parameters tensor
	self._params, self._grad_params = model_utils.combine_all_parameters(self._prototype.rnn)
	-- log:debug("Number of parameters: ", params:nElement())

	-- initialization
	local do_random_init = true

	if do_random_init then
	  self._params:uniform(-0.08, 0.08) -- small uniform numbers
	end

	-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
	if opt.model == 'lstm' then
	  for layer_idx = 1, opt.num_layers do
	    for _,node in ipairs(self._prototype.rnn.forwardnodes) do
	      if node.data.annotations.name == "i2h_" .. layer_idx then
	        self:debug('Setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
	        -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
	        node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
	      end
	    end
	  end
	end

	self:debug('Number of parameters in the model: ' .. self._params:nElement())
end

--[[
Function: generateClones

Method used to generate the clones of the protoype
]]
function Class:generateClones()
	self:debug("Generating clones...")
	-- make a bunch of clones after flattening, as that reallocates memory
	clones = {}
	for name,proto in pairs(self._prototype) do
	  self:debug('Cloning ' .. name)
	  clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
	end
	self._clones = clones
end

-- preprocessing helper function
function prepro(x,y)
  -- x = x:transpose(1,2):contiguous() -- swap the axes for faster indexing
  x = x:contiguous() -- swap the axes for faster indexing
  -- y = y:transpose(1,2):contiguous()
  y = y:contiguous()
  
  if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
    -- have to convert to float because integers can't be cuda()'d
    x = x:float():cuda()
    y = y:float():cuda()
  end

  if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
    x = x:cl()
    y = y:cl()
  end
  return x,y
end

--[[
Function: evaluateSplit

Method used to evaluate the current loss on a given split of the batches
]]
function Class:evaluateSplit(split_index, max_batches)
  self:debug('Evaluating loss over split index ' .. split_index)
  local prov = self._provider

  local n = prov.split_sizes[split_index]
  if max_batches ~= nil then n = math.min(max_batches, n) end

  prov:resetBatchPointer(split_index) -- move batch iteration pointer for this split to front
  local loss = 0
  local rnn_state = {[0] = self._initState}
  
  for i = 1,n do -- iterate over batches in the split
    -- fetch a batch
    local x, y = prov:nextBatch(split_index)
    x,y = prepro(x,y)
    -- forward pass
    for t=1,opt.seq_length do
      self._clones.rnn[t]:evaluate() -- for dropout proper functioning
      local lst = self._clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
      rnn_state[t] = {}
      for i=1,#self._initState do table.insert(rnn_state[t], lst[i]) end
      local prediction = lst[#lst] 
      loss = loss + self._clones.criterion[t]:forward(prediction, y[t])
    end

    -- carry over lstm state
    rnn_state[0] = rnn_state[#rnn_state]
    self:debug(i .. '/' .. n .. '...')
  end

  loss = loss / opt.seq_length / n
  return loss
end

--[[
Function: trainEval

Core method used during training.
Do fwd/bwd and return loss, grad_params
]]
function Class:trainEval(x)
  if x ~= self._params then
    self._params:copy(x)
  end
  self._grad_params:zero()
  
  local opt = self._config

  ------------------ get minibatch -------------------
  local x, y = self._provider:nextBatch(1)
  x,y = prepro(x,y)
  ------------------- forward pass -------------------
  local rnn_state = {[0] = self._initStateGlobal}
  local predictions = {}           -- softmax outputs
  local loss = 0

  -- observing dimensions of seq_length x batch_size below:
  -- print("x dimensions: ",x:size(1),"x",x:size(2))
  -- print("y dimensions: ",y:size(1),"x",y:size(2))

  for t=1,opt.seq_length do
    self._clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
    local lst = self._clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}

    -- print("x[".. t.."]:", x[t])
    -- line below will always return #lst=5 (with 2 layers)
    -- and  #lst=7 with 3 layers 
    -- This correspond to the description of the LSTM model (2 outputs per layers + final output)
    -- print("Size of lst is: ".. #lst .. " at seq = "..t)

    -- We anticipate that the value below should be 4 when we have 2 layers: OK
    -- print("Size of init_state: ".. #init_state)

    rnn_state[t] = {}
    for i=1,#self._initState do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output

    predictions[t] = lst[#lst] -- last element is the prediction
    -- self:debug("predictions[",t,"] dims= ",predictions[t]:nDimension(),": ",predictions[t]:size(1),"x",predictions[t]:size(2))
    -- self:debug("y[",t,"] dims= ",y[t]:nDimension(),": ",y[t]:size(1))

    loss = loss + self._clones.criterion[t]:forward(predictions[t], y[t])
    -- self:debug("New loss value: ",loss)
  end
  loss = loss / opt.seq_length
  
  ------------------ backward pass -------------------
  -- initialize gradient at time t to be zeros (there's no influence from future)
  local drnn_state = {[opt.seq_length] = clone_list(self._initState, true)} -- true also zeros the self._clones
  for t=opt.seq_length,1,-1 do
    -- backprop through loss, and softmax/linear
    local doutput_t = self._clones.criterion[t]:backward(predictions[t], y[t])
    table.insert(drnn_state[t], doutput_t)
    local dlst = self._clones.rnn[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
    drnn_state[t-1] = {}
    for k,v in pairs(dlst) do
      if k > 1 then -- k == 1 is gradient on x, which we dont need
          -- note we do k-1 because first item is dembeddings, and then follow the 
          -- derivatives of the state, starting at index 2. I know...
        drnn_state[t-1][k-1] = v
      end
    end
  end
  ------------------------ misc ----------------------
  -- transfer final state to initial state (BPTT)
  self._initStateGlobal = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
  -- self._grad_params:div(opt.seq_length) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
  -- clip gradient element-wise
  self._grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  return loss, self._grad_params	
end

--[[
Function: train

Main entry point to perform the training of the network
]]
function Class:train()
	-- start optimization here
	local train_losses = {}
	local val_losses = {}
	local opt = self._config
	local prov = self._provider

	local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
	local iterations = opt.max_epochs * prov.ntrain
	local iterations_per_epoch = prov.ntrain
	local loss0 = nil

	local feval = function(x)
		return self:trainEval(x)
	end

	for i = 1, iterations do
	  local epoch = i / prov.ntrain

	  local timer = torch.Timer()
	  local _, loss = optim.rmsprop(feval, self._params, optim_state)
	  if opt.accurate_gpu_timing == 1 and opt.gpuid >= 0 then
	    --[[
	    Note on timing: The reported time can be off because the GPU is invoked async. If one
	    wants to have exactly accurate timings one must call cutorch.synchronize() right here.
	    I will avoid doing so by default because this can incur computational overhead.
	    --]]
	    cutorch.synchronize()
	  end
	  local time = timer:time().real
	  
	  local train_loss = loss[1] -- the loss is inside a list, pop it
	  train_losses[i] = train_loss

	  -- exponential learning rate decay
	  if i % prov.ntrain == 0 and opt.learning_rate_decay < 1 then
	    if epoch >= opt.learning_rate_decay_after then
	      local decay_factor = opt.learning_rate_decay
	      optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
	      self:debug('Decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
	    end
	  end

	  -- every now and then or on last iteration
	  if i % opt.eval_val_every == 0 or i == iterations then
	    -- evaluate loss on validation data
	    local val_loss = self:evaluateSplit(2) -- 2 = validation
	    val_losses[i] = val_loss

	    local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
	    self:debug('Saving checkpoint to ' .. savefile)
	    local checkpoint = {}
	    checkpoint.protos = self._prototype
	    checkpoint.opt = opt
	    checkpoint.train_losses = train_losses
	    checkpoint.val_loss = val_loss
	    checkpoint.val_losses = val_losses
	    checkpoint.i = i
	    checkpoint.epoch = epoch
	    self._provider:addCheckpointData(checkpoint)
	    torch.save(savefile, checkpoint)
	  end

	  if i % opt.print_every == 0 then
	    self:debug(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, self._grad_params:norm() / self._params:norm(), time))
	  end
	 
	  if i % 10 == 0 then collectgarbage() end

	  -- handle early stopping if things are going really bad
	  if loss[1] ~= loss[1] then
	    self:warn('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
	    break -- halt
	  end

	  if loss0 == nil then loss0 = loss[1] end
	  if loss[1] > loss0 * 300 then
	    self:warn('loss is exploding, aborting.')
	    break -- halt
	  end
	end

	-- Write the train losses to file:
	self:writeArray('train_losses.csv',train_losses)

	-- write the eval losses to file:
	-- first we need to fill the val_losses array:
	local cval = 0.0
	for i=1,#train_losses do
		if not val_losses[i] then
			val_losses[i] = cval
		else
			cval = val_losses[i]
		end
	end
	self:writeArray('val_losses.csv',val_losses)

end

--[[
Function: writeArray


]]
function Class:writeArray(filename,array)
	local file = io.open(filename,"w")
	for _,v in ipairs(array) do
		file:write(v.."\n")
	end
	file:close()
end

return Class