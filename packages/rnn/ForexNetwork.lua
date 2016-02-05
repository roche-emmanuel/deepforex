local Class = createClass{name="ForexNetwork",bases={"base.Object"}};

local utils = require "rnn.Utils"

--[[
Class: rnn.ForexNetwork
]]

--[=[
--[[
Constructor: ForexNetwork

Create a new instance of the class.
]]
function ForexNetwork(options)
]=]
function Class:initialize(options)
  CHECK(options.id,"Invalid ForexNetwork ID")
  CHECK(options.parent,"Invalid parent")
  CHECK(options.opt,"Invalid opt")

  self._id = options.id
  self:debug("Creating a ForexNetwork instance with id=", self._id)

  self.opt = options.opt
  self._parent = options.parent

  -- Current train session:
  self._session = 0

  -- Network is not ready initially:
  self._isReady = false

  self._isTraining = false;

  local opt = self.opt

  -- Create the RNN prototype:
  self:debug("Creating RNN prototype")
  self._proto = utils:createPrototype(opt)

  -- Create the init state:
  self:debug("Creating init state")
  self._initState = utils:createInitState(opt)

  -- We also keep a reference on a global init state table:
  -- self:debug("Creating global train state")
  -- self._globalTrainState = utils:cloneList(self._initState)

  -- also prepare a dedicated evaluation state:
  self:debug("Creating global eval state")
  self._globalEvalState = utils:createInitState(opt,1)

  -- Perform parameter initialization:
  self:debug("Retrieving network parameters...")  
  self._params, self._gradParams = utils:initParameters(opt, self._proto)

  -- Generate the clones from the prototype:
  self:debug("Generating RNN clones from prototypes")
  self._clones = utils:generateClones(opt, self._proto)
end

--[[
Function: isReady

Check if this network is ready for usage
]]
function Class:isReady()
  return self._isReady
end

--[[
Function: getID

Retrieve the ID of this network
]]
function Class:getID()
  return self._id
end

--[[
Function: trainEval

Core method used during training.
Do fwd/bwd and return loss, grad_params
]]
function Class:trainEval(x)
  local opt = self.opt

  if x ~= self._params then
    self._params:copy(x)
  end
  self._gradParams:zero()

  ------------------ get minibatch -------------------
  local idx = self._iteration
  local x,y;
  x = self.x_batches[idx]
  y = self.y_batches[idx]

  ------------------- forward pass -------------------
  local rnn_state = {[0] = self._globalTrainState}
  local predictions = {}           -- softmax outputs
  local loss = 0

  -- observing dimensions of seq_length x batch_size below:
  -- print("x dimensions: ",x:size(1),"x",x:size(2))
  -- print("y dimensions: ",y:size(1),"x",y:size(2))
  local bsize = opt.batch_size

  for t=1,opt.seq_length do
    self._clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
    local input = x[t]

    -- self:debug("input size: ", input:size())

    local lst = self._clones.rnn[t]:forward{input, unpack(rnn_state[t-1])}

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
  local drnn_state = {[opt.seq_length] = utils:cloneList(self._initState, true)} -- true also zeros the self._clones
  for t=opt.seq_length,1,-1 do
    -- backprop through loss, and softmax/linear
    local doutput_t = self._clones.criterion[t]:backward(predictions[t], y[t])
    table.insert(drnn_state[t], doutput_t)
    
    local input = bsize == 1 and x:narrow(1,t,1) or x[t]

    local dlst = self._clones.rnn[t]:backward({input, unpack(rnn_state[t-1])}, drnn_state[t])
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
  -- here we use only the subsequence rnn_state, not the final one,
  -- because the next sequence will start with the next time step, and not
  -- the final sequence time step.
  self._globalTrainState = rnn_state[1]

  -- note that we only keep the latest state from the latest batch because this is where
  -- we will start the evaluation from:
  -- local src = rnn_state[#rnn_state]
  -- also note that we start from the state 1 for this global state too as we will
  -- take the first (seq_len - 1) element for the first sequence from the training set.

  -- For now, we do not assign any evaluation state at all:
  -- local src = rnn_state[1]
  -- for k,tens in ipairs(src) do
  --   tdesc.global_eval_state[k] = tens[bsize]
  -- end

  -- tdesc.global_init_state = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
  -- self._grad_params:div(opt.seq_length) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
  -- clip gradient element-wise
  self._gradParams:clamp(-opt.grad_clip, opt.grad_clip)
  return loss, self._gradParams
end

--[[
Function: prepareMiniBatchFeatures

Prepare the mini batches before starting to train.
]]
function Class:prepareMiniBatchFeatures()
  local opt = self.opt

  self:debug("Preparing Mini batch features/labels...")
  local nrows = opt.train_size

  -- Assume that this is already the size of the features/labels:
  CHECK(self._raw_features:size(1)==nrows,"Invalid number of features")
  CHECK(self._raw_labels:size(1)==nrows,"Invalid number of labels")

  local features = self._raw_features:narrow(1,self._train_offset+1, nrows)
  local labels = self._raw_labels:narrow(1,self._train_offset+1, nrows)

  -- Complete length of the sequence in each batch:
  local seq_len = opt.seq_length

  local nbatches = opt.seq_length * opt.batch_num_seqs
  local bsize = opt.batch_size
  local nf = self._raw_features:size(2)

  local x_batches = {}
  local y_batches = {}
  
  local stride = opt.seq_length * opt.batch_num_seqs
  
  local offset = 0
  local idx;

  for i=1,nbatches do
    local xbatch = torch.Tensor(seq_len,bsize,nf)
    local ybatch = torch.Tensor(seq_len,bsize)

    for t=1,seq_len do
      -- build a tensor corresponding to the sequence element t
      -- eg. we present all the rows of features in batch that arrive
      -- at time t in the sequence:
      local xmat = torch.Tensor(bsize,nf)
      local ymat = torch.Tensor(bsize)

      -- fill the data for this tensor:
      for i=1,bsize do
        idx = offset+(i-1)*stride+t
        xmat[{i,{}}] = features[{idx,{}}]
        ymat[i] = labels[idx]
      end

      xbatch[t] = xmat
      ybatch[t] = ymat
    end

    table.insert(x_batches,utils:prepro(opt,xbatch))
    table.insert(y_batches,utils:prepro(opt,ybatch))
    
    offset = offset + 1    
  end

  self.x_batches = x_batches
  self.y_batches = y_batches
  self:debug("Done preparing Mini batch features/labels.")
end

--[[
Function: performTrainingSession

Method used to perform a training session
]]
function Class:performTrainingSession()
  local opt = self.opt

  CHECK(opt.learning_rate,"Invalid learning_rate")
  CHECK(opt.decay_rate,"Invalid decay_rate")
  CHECK(opt.max_epochs,"Invalid max_epochs")
  CHECK(opt.initial_max_epochs,"Invalid initial_max_epochs")
  CHECK(opt.batch_num_seqs,"Invalid batch_num_seqs")
  CHECK(opt.train_size,"Invalid train_size")
  CHECK(opt.seq_length,"Invalid seq_length")
  CHECK(opt.accurate_gpu_timing,"Invalid accurate_gpu_timing")

  -- There is no train offset to apply here:
  self._train_offset = 0

  -- Keep reference on the train losses:
  self._trainLosses = {}

  -- Prepare optimization state:
  local optim_state = {}
    -- rmsprop state:
  optim_state.learningRate = opt.learning_rate
  optim_state.alpha = opt.decay_rate

  -- The number of iteration is the number of times we can extract a sequence of seq_length
  -- in the train_size, numtiplied by the number of epochs we allow:
  local ntrain_per_epoch = opt.batch_num_seqs*opt.seq_length

  -- If we use minibatch, then we also need to setup the features/labels 
  -- appropriately:
  self:prepareMiniBatchFeatures()

  local iterations = self._max_epochs * ntrain_per_epoch
  self:debug("Need to perform ", iterations, " training iterations")

  local loss0 = nil

  local optimfunc = optim.rmsprop

  local feval = function(x)
    return self:trainEval(x)
  end

  self._iteration = 0
  for i = 1, iterations do
    local epoch = i / ntrain_per_epoch
    
    self._iteration = self._iteration + 1
    if self._iteration > ntrain_per_epoch then
      self._iteration = 1
    end

    local timer = torch.Timer()

    local _, loss = optimfunc(feval, self._params, optim_state)
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
    table.insert(self._trainLosses, train_loss)

    if i % opt.print_every == 0 then
      self:debug(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, self._gradParams:norm() / self._params:norm(), time))
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

  self._isTraining = false
  self._isReady = true
  self:debug("Training done.")
end

--[[
Function: isTraining

Check if this network is currently training
]]
function Class:isTraining()
  return self._isTraining
end

--[[
Function: train

Perform the training on the given inputs
]]
function Class:train(features,labels,timetags)
  if self._isTraining then
    self:debug("Network already training.")
    return
  end

  self._raw_features = features
  self._raw_labels = labels
  self._raw_timetags = timetags

  self._max_epochs = opt.max_epochs
  if self._session == 0 then
    self._max_epochs = opt.initial_max_epochs
  end

  self._session = self._session + 1
  self:debug("Training on session ", self._session)

  -- No init state influence between training sessions:
  self._globalTrainState = utils:cloneList(self._initState,true)

  self._isTraining = true

  -- This method should be called in an auxiliary thread.
  self:performTrainingSession()
end

--[[
Function: evaluate

Use to evaluate the prediction of the network on a given feature sequence
]]
function Class:evaluate(features,labels)
  CHECK(self._isReady,"Trying to evaluate a not ready network")

  if self._isTraining then
    self:debug("Network ", self._id," currently training: returning 0 for evaluation.")
    
    -- Yet we still have to keep track of this evaluation request, to execute it later,
    -- ant thus update the global evaluation state.
    self._pendingEvals = self._pendingEvals or {}
    table.insert(self._pendingEvals, features:clone())
    return 0.0
  end

  -- process any pending evaluation:
  if self._pendingEvals then

    -- displace the table to avoid infinite loop:
    local previous = self._pendingEvals
    -- Now remove the pendingEvals:
    self._pendingEvals = nil

    for _,feats in ipairs(previous) do
      self:evaluate(feats)
    end
  end

  self:debug("Evaluating with features tensor: ", features)
  
  self._gradParams:zero()

  local opt = self.opt

  -- the x and y to use are the features and labels we received:
  local x = features;
  local y = labels; -- this may be nil

  ------------------- forward pass -------------------
  local rnn_state = {[0] = self._globalEvalState}
  -- local predictions = {}           -- softmax outputs
  local loss = nil
  local pred;

  -- observing dimensions of seq_length x batch_size below:
  -- print("x dimensions: ",x:size(1),"x",x:size(2))
  -- print("y dimensions: ",y:size(1),"x",y:size(2))
  local len = opt.seq_length

  for t=1,len do
    self._clones.rnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
    local lst = self._clones.rnn[t]:forward{x:narrow(1,t,1), unpack(rnn_state[t-1])}

    -- print("x[".. t.."]:", x[t])
    -- line below will always return #lst=5 (with 2 layers)
    -- and  #lst=7 with 3 layers 
    -- This correspond to the description of the LSTM model (2 outputs per layers + final output)
    -- print("Size of lst is: ".. #lst .. " at seq = "..t)

    -- We anticipate that the value below should be 4 when we have 2 layers: OK
    -- print("Size of init_state: ".. #init_state)

    rnn_state[t] = {}
    for i=1,#self._initState do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output

    -- override the prediction value each time: we only need the latest one:
    pred = lst[#lst]


    -- predictions[t] = lst[#lst] -- last element is the prediction
    -- self:debug("predictions[",t,"] dims= ",predictions[t]:nDimension(),": ",predictions[t]:size(1),"x",predictions[t]:size(2))
    -- self:debug("y[",t,"] dims= ",y[t]:nDimension(),": ",y[t]:size(1))

    -- loss = loss + self._clones.criterion[t]:forward(predictions[t], y[t])
    -- self:debug("New loss value: ",loss)
  end

  -- Compute the loss if the corresponding label is provided
  -- but do not complain otherwise:
  if y then
    -- The loss should only be considered on the latest prediction:
    loss = self._clones.criterion[len]:forward(pred, y[len])
    -- loss = loss / opt.seq_length
  end

  -- Update the global init state:
  -- tdesc.global_eval_state = rnn_state[1]
  -- clone the tensor to ensure we get a separated copy:
  local src = rnn_state[1]
  for k,tens in ipairs(src) do
    self._globalEvalState[k] = tens:clone()
  end

  -- if we use a classifier, then we should extract the classification weights:  
  pred = pred:storage()[1]

  local yval = nil 
  if y then
    yval = y[{len,1}]
  end

  return pred, loss, yval 
end

return Class


