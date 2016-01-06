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
  self:debug("Creating a ForexNetwork instance with id", self._id)

  self.opt = options.opt
  self._parent = options.parent

  -- Current train session:
  self._session = 0

  -- Network is not ready initially:
  self._isReady = false

  local opt = self.opt

  -- Create the RNN prototype:
  self:debug("Creating RNN prototype")
  self._proto = utils:createPrototype(opt)

  -- Create the init state:
  self:debug("Creating init state")
  self._initState = utils:createInitState(opt)

  -- We also keep a reference on a global init state table:
  self:debug("Creating global train state")
  self._globalTrainState = utils:cloneList(self._initState)

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
Function: prepareMiniBatchFeatures

Prepare the mini batches before starting to train.
]]
function Class:prepareMiniBatchFeatures()
  local opt = self.opt

  self:debug("Preparing Mini batch features/labels...")
  local nrows = opt.train_size

  -- Assume that this is already the size of the features/labels:
  CHECK(self._features:size(1)==nrows,"Invalid number of features")
  CHECK(self._labels:size(1)==nrows,"Invalid number of labels")

  local features = tdesc.raw_features:narrow(1,tdesc.train_offset+1, nrows)
  local labels = tdesc.raw_labels:narrow(1,tdesc.train_offset+1, nrows)

  -- Complete length of the sequence in each batch:
  local seq_len = opt.seq_length

  local nbatches = opt.seq_length * opt.batch_num_seqs
  local bsize = opt.batch_size
  local nf = self._features:size(2)

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

    table.insert(x_batches,prepro(opt,xbatch))
    table.insert(y_batches,prepro(opt,ybatch))
    
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
function Class:performTrainingSession(opt,features,labels,timetags)
  local opt = self.opt

  CHECK(opt.learning_rate,"Invalid learning_rate")
  CHECK(opt.decay_rate,"Invalid decay_rate")
  CHECK(opt.max_epochs,"Invalid max_epochs")
  CHECK(opt.initial_max_epochs,"Invalid initial_max_epochs")
  CHECK(opt.batch_num_seqs,"Invalid batch_num_seqs")
  CHECK(opt.train_size,"Invalid train_size")
  CHECK(opt.seq_length,"Invalid seq_length")
  CHECK(opt.accurate_gpu_timing,"Invalid accurate_gpu_timing")

  self._features = features
  self._labels = labels
  self._timetags = timetags

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

  local max_epochs = opt.max_epochs
  if self._session == 0 then
    max_epochs = opt.initial_max_epochs
  end

  local iterations = max_epochs * ntrain_per_epoch
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

  self:debug("Training done.")
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

  if self._session == 0 then
    self:debug("Should perform initial training here")
  end

  self._session = self._session + 1
  self:debug("Training on session ", self._session)


  local tdesc = {}
  tdesc.raw_features = features
  tdesc.raw_labels = labels
  tdesc.timetags = timetags
  tdesc.params = self._params
  tdesc.grad_params = self._gradParams
  tdesc.init_state = self._initState
  tdesc.train_offset = 0
  tdesc.clones = self._clones

  CHECK(opt.learning_rate_decay,"Invalid learning_rate_decay")
  CHECK(opt.learning_rate_decay_after,"Invalid learning_rate_decay_after")
  CHECK(opt.print_every,"Invalid print_every")
  CHECK(opt.ema_adaptation,"Invalid ema_adaptation")
  CHECK(opt.optim,"Invalid optim")
  CHECK(opt.batch_size,"Invalid batch_size")

  -- start optimization here
  -- tdesc.train_losses = tdesc.train_losses or {}
  -- -- local val_losses = {}

  -- local optim_state = {}
  -- -- rmsprop state:
  -- optim_state.learningRate = opt.learning_rate
  -- optim_state.alpha = opt.decay_rate
  
  -- -- conjugate gradient state:
  -- optim_state.maxEval = 6
  -- optim_state.maxIter = 2

  -- The number of iteration is the number of times we can extract a sequence of seq_length
  -- in the train_size, numtiplied by the number of epochs we allow:
  -- local ntrain_by_epoch = (opt.train_size - opt.seq_length + 1)

  -- -- Note that, if the batch_size is specified, then the number of iterations in each
  -- -- epoch is given by opt.batch_num_seqs*opt.seq_length:
  -- if opt.batch_size > 0 then
  --   ntrain_by_epoch = opt.batch_num_seqs*opt.seq_length

  --   -- If we use minibatch, then we also need to setup the features/labels 
  --   -- appropriately:
  --   self:prepareMiniBatchFeatures(opt,tdesc)
  -- end

  -- Upload the raw features/labels if needed:
  -- but we keep a CPU version untouched as we might need it
  -- to generate the subsequent x/y batches.
  tdesc.features = prepro(opt,tdesc.raw_features:clone())
  tdesc.labels = prepro(opt,tdesc.raw_labels:clone())

  tdesc.ntrain_per_epoch = ntrain_by_epoch

  local iterations = opt.max_epochs * ntrain_by_epoch
  local loss0 = nil

  local optimfunc = optim.rmsprop
  -- Change the optimization function if needed:
  if opt.optim == 'cg' then
    optimfunc = optim.cg
  end

  local feval = function(x)
    return self:trainEval(opt, tdesc, x)
  end

  tdesc.iteration = 0
  for i = 1, iterations do
    local epoch = i / ntrain_by_epoch
    
    tdesc.iteration = tdesc.iteration + 1
    if tdesc.iteration > tdesc.ntrain_per_epoch then
      tdesc.iteration = 1
    end

    local timer = torch.Timer()

    local _, loss = optimfunc(feval, tdesc.params, optim_state)
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
    table.insert(tdesc.train_losses, train_loss)

    if i % opt.print_every == 0 then
      self:debug(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, tdesc.grad_params:norm() / tdesc.params:norm(), time))
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

  tdesc.eval_losses = tdesc.eval_losses or {}
  tdesc.correct_signs = tdesc.correct_signs or {}
  tdesc.evalidx_values = tdesc.evalidx_values or {}
  tdesc.timetags_values = tdesc.timetags_values or {}
  tdesc.pred_values = tdesc.pred_values or {}
  tdesc.label_values = tdesc.label_values or {}

  tdesc.current_sign = tdesc.current_sign or 0.5

  local alpha = opt.ema_adaptation

  -- Start the evaluation just after the train samples:
  -- note that the first (seq_len-1) observations are taken from the
  -- training set, so we have to account for this offset:
  -- Additionally note that the +1 is done in the loop below directly:
  tdesc.iteration = opt.train_size - opt.seq_length

  tdesc.eval_offset = tdesc.eval_offset or (opt.train_size-1)

  -- Now that we are done with the training part we should evaluate the network predictions:
  self:debug("Starting evaluation at offset position: ", tdesc.eval_offset)

  for i=1,opt.eval_size do

    --  Move to the nex iteration each time:
    tdesc.eval_offset = tdesc.eval_offset + 1
    tdesc.iteration = tdesc.iteration + 1 
    local loss, pred, yval = self:evaluate(opt, tdesc)

    self:debug("Prediction: ", pred, ", real value: ",yval)

    table.insert(tdesc.timetags_values, tdesc.timetags and tdesc.timetags[tdesc.eval_offset] or tdesc.eval_offset)
    table.insert(tdesc.evalidx_values,i)
    table.insert(tdesc.pred_values,pred)
    table.insert(tdesc.label_values,yval)

    -- We also check if we have the proper sign for the prediction:
    local goodSign = 0

    if opt.num_classes == 1 and ((pred-0.5) * (yval-0.5) > 0.0) then
      goodSign = 1
    elseif opt.num_classes > 1 and (pred - opt.num_classes/2 - 0.5) * (yval - opt.num_classes - 0.5) > 0.0 then
      goodSign = 1 
    end

    tdesc.current_loss = tdesc.current_loss and (tdesc.current_loss * (1.0 - alpha) + loss * alpha) or loss
    table.insert(tdesc.eval_losses,tdesc.current_loss)

    if opt.num_classes == 1 then
      tdesc.current_sign = (tdesc.current_sign * (1.0 - alpha) + goodSign * alpha)
    end

    table.insert(tdesc.correct_signs, tdesc.current_sign)
    self:debug("Evaluation ",i,": Loss EMA=",tdesc.current_loss,", correct sign EMA=",tdesc.current_sign)
  end

end


return Class


