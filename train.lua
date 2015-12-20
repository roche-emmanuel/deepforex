-- print("OS: ", jit.os)
-- print("arch: ", jit.arch)

root_path = paths.cwd()
-- print("Root path: ", root_path)

package.path = root_path.."/?.lua;"..root_path.."/?/init.lua;"..root_path.."/packages/?.lua;"..package.path
-- package.cpath = path.."bin/"..flavor.."/modules/?.dll;".. path.."bin/"..flavor.."/modules/?51.dll;" ..package.cpath

-- global level definition of comment methods:
createClass = require("base.ClassBuilder")()
log = require("log.DefaultLogger")
trace = require("log.DefaultTracer")

CHECK = function(cond,msg,...)
  if not cond then
    log:error(msg,...)
    log:error("Stack trace: ",debug.traceback())
    error("Stopping because a static assertion error occured.")
  end
end

PROTECT = function(func,...)
  local status, res = pcall(func,...)
  if not status then
    log:error("Error detected: ",res)
  end
end

config_file = "dforex_config"


require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

-- We use float tensors in this project:
torch.setdefaulttensortype('torch.FloatTensor')

require 'utils.misc'

local model_utils = require 'utils.model_utils'
local ForexLoader = require "utils.ForexLoader"

local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a FOREX trading agent')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','inputs/2004_01_to_2004_04','data directory. Should contain the input data for the training')
-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm, gru or rnn')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',20,'number of timesteps to unroll for')
cmd:option('-batch_size',40,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0 then
  local ok, cunn = pcall(require, 'cunn')
  local ok2, cutorch = pcall(require, 'cutorch')
  if not ok then print('package cunn not found!') end
  if not ok2 then print('package cutorch not found!') end
  if ok and ok2 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    cutorch.manualSeed(opt.seed)
  else
    print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
    print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
    print('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end

-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1 then
  local ok, cunn = pcall(require, 'clnn')
  local ok2, cutorch = pcall(require, 'cltorch')
  if not ok then print('package clnn not found!') end
  if not ok2 then print('package cltorch not found!') end
  if ok and ok2 then
    print('using OpenCL on GPU ' .. opt.gpuid .. '...')
    cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    torch.manualSeed(opt.seed)
  else
    print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
    print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
    print('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end

-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
opt.split_fractions = {opt.train_frac, opt.val_frac, test_frac} 

-- prepare the loader:
local loader = ForexLoader(opt)

-- Create the model:
print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')

local ni = loader:getInputSize()
print("Number of inputs: ".. ni)

local protos = {}
if opt.model == 'lstm' then
  protos.rnn = LSTM.lstm(ni, opt.rnn_size, opt.num_layers, opt.dropout)
elseif opt.model == 'gru' then
  protos.rnn = GRU.gru(ni, opt.rnn_size, opt.num_layers, opt.dropout)
elseif opt.model == 'rnn' then
  protos.rnn = RNN.rnn(ni, opt.rnn_size, opt.num_layers, opt.dropout)
end
-- protos.criterion = nn.ClassNLLCriterion()
protos.criterion = nn.MSECriterion()


-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
  local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
  if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
  if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
  table.insert(init_state, h_init:clone())
  if opt.model == 'lstm' then
    table.insert(init_state, h_init:clone())
  end
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

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)
-- log:debug("Number of parameters: ", params:nElement())

-- initialization
if do_random_init then
  params:uniform(-0.08, 0.08) -- small uniform numbers
end

-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.model == 'lstm' then
  for layer_idx = 1, opt.num_layers do
    for _,node in ipairs(protos.rnn.forwardnodes) do
      if node.data.annotations.name == "i2h_" .. layer_idx then
        print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
        -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
        node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
      end
    end
  end
end

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
  print('cloning ' .. name)
  clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- preprocessing helper function
function prepro(x,y)
  x = x:transpose(1,2):contiguous() -- swap the axes for faster indexing
  y = y:transpose(1,2):contiguous()
  
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

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
  print('evaluating loss over split index ' .. split_index)
  local n = loader.split_sizes[split_index]
  if max_batches ~= nil then n = math.min(max_batches, n) end

  loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
  local loss = 0
  local rnn_state = {[0] = init_state}
  
  for i = 1,n do -- iterate over batches in the split
    -- fetch a batch
    local x, y = loader:nextBatch(split_index)
    x,y = prepro(x,y)
    -- forward pass
    for t=1,opt.seq_length do
      clones.rnn[t]:evaluate() -- for dropout proper functioning
      local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
      rnn_state[t] = {}
      for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
      prediction = lst[#lst] 
      loss = loss + clones.criterion[t]:forward(prediction, y[t])
    end
    -- carry over lstm state
    rnn_state[0] = rnn_state[#rnn_state]
    print(i .. '/' .. n .. '...')
  end

  loss = loss / opt.seq_length / n
  return loss
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
function feval(x)
  if x ~= params then
    params:copy(x)
  end
  grad_params:zero()

  ------------------ get minibatch -------------------
  local x, y = loader:nextBatch(1)
  x,y = prepro(x,y)
  ------------------- forward pass -------------------
  local rnn_state = {[0] = init_state_global}
  local predictions = {}           -- softmax outputs
  local loss = 0

  -- observing dimensions of seq_length x batch_size below:
  -- print("x dimensions: ",x:size(1),"x",x:size(2))
  -- print("y dimensions: ",y:size(1),"x",y:size(2))

  for t=1,opt.seq_length do
    clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
    local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}

    -- line below will always return #lst=5 (with 2 layers)
    -- and  #lst=7 with 3 layers 
    -- This correspond to the description of the LSTM model (2 outputs per layers + final output)
    -- print("Size of lst is: ".. #lst .. " at seq = "..t)

    -- We anticipate that the value below should be 4 when we have 2 layers: OK
    -- print("Size of init_state: ".. #init_state)

    rnn_state[t] = {}
    for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output

    predictions[t] = lst[#lst] -- last element is the prediction
    loss = loss + clones.criterion[t]:forward(predictions[t], y[t])
  end
  loss = loss / opt.seq_length
  
  ------------------ backward pass -------------------
  -- initialize gradient at time t to be zeros (there's no influence from future)
  local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
  for t=opt.seq_length,1,-1 do
    -- backprop through loss, and softmax/linear
    local doutput_t = clones.criterion[t]:backward(predictions[t], y[t])
    table.insert(drnn_state[t], doutput_t)
    local dlst = clones.rnn[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
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
  init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
  -- grad_params:div(opt.seq_length) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
  -- clip gradient element-wise
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
for i = 1, iterations do
  local epoch = i / loader.ntrain

  local timer = torch.Timer()
  local _, loss = optim.rmsprop(feval, params, optim_state)
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
  if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
    if epoch >= opt.learning_rate_decay_after then
      local decay_factor = opt.learning_rate_decay
      optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
      print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
    end
  end

  -- every now and then or on last iteration
  if i % opt.eval_val_every == 0 or i == iterations then
    -- evaluate loss on validation data
    local val_loss = eval_split(2) -- 2 = validation
    val_losses[i] = val_loss

    local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
    print('saving checkpoint to ' .. savefile)
    local checkpoint = {}
    checkpoint.protos = protos
    checkpoint.opt = opt
    checkpoint.train_losses = train_losses
    checkpoint.val_loss = val_loss
    checkpoint.val_losses = val_losses
    checkpoint.i = i
    checkpoint.epoch = epoch
    checkpoint.vocab = loader.vocab_mapping
    torch.save(savefile, checkpoint)
  end

  if i % opt.print_every == 0 then
    print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
  end
 
  if i % 10 == 0 then collectgarbage() end

  -- handle early stopping if things are going really bad
  if loss[1] ~= loss[1] then
    print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
    break -- halt
  end

  if loss0 == nil then loss0 = loss[1] end
  if loss[1] > loss0 * 3 then
    print('loss is exploding, aborting.')
    break -- halt
  end
end

print("Training done.")
