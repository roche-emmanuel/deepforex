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

local utils = require "rnn.Utils"

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an online FOREX trading agent')
cmd:text()
cmd:text('Options')

-- data
cmd:option('-data_dir','inputs/raw_2004_01_to_2007_01','data directory. Should contain the file input.txt with input data')

-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm,gru or rnn')

-- Base setup options:
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-dropout',0.5,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',25,'number of timesteps to unroll for')

cmd:option('-forcast_index',1,'Index of the feature that should be forcasted.')
cmd:option('-num_classes',1,'Number of classes to consider when performing classification.')
cmd:option('-batch_size',-1,'Number of sequences to train on in parallel or -1 if we use only sequential training')
cmd:option('-batch_num_seqs',1,'Number of consecutive sequences in each batch slice')
cmd:option('-train_size',2000,'Number of steps used for each training session')
cmd:option('-eval_size',100,'Number of steps used for each evaluation session')
cmd:option('-max_sessions',200,'Max number of training/eval sessions to perform')
cmd:option('-print_every',100,'how many steps/minibatches between printing out the loss')

cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-max_epochs',1.0,'number of full passes through the training data')
cmd:option('-initial_max_epochs',3.0,'number of full passes through the training data on the first training session')

cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-ema_adaptation',0.001,'Moving average adaptation factor')
cmd:option('-suffix','vxx','suffix to append to all written files')
cmd:option('-num_emas',0,'Number of EMA features generated for each symbol')
cmd:option('-num_remas',0,'Number of return EMA features generated for each symbol')
cmd:option('-ema_base_period',5,'Base period for the EMA addition')
cmd:option('-rsi_period',0,'Period to use for RSI of 0 if disabled')
cmd:option('-log_return_offsets',"",'list of comma separated offset values that should be used to compute additional log return features')
cmd:option('-feature_offset',20,'Offset applied at the start of the features tensor before starting the training process')
cmd:option('-with_timetag',0,'Set this to 1 if the raw inputs dataset provides a timetag column')
cmd:option('-with_close_only',0,'Set this to 1 if the raw inputs dataset only provides the close prices for each symbol')
cmd:option('-start_offset',0,'Offset applied on raw inputs before training anything')

cmd:option('-optim','rmsprop','Optimization algorithm')


cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- Setup the common GPU setup elements:
utils:setupGPU(opt)

-- Before we can create a prototype we must know what will be the input/output sizes
-- for the network
-- And thus we must load the data
local raw_inputs, timetags = utils:loadRawInputs(opt)
log:debug("Number of raw input samples: ", raw_inputs:size(1))

if opt.start_offset > 0 then
  log:debug("Applying start offset of ", opt.start_offset)
  raw_inputs = raw_inputs:sub(opt.start_offset+1,-1)
  if timetags then
    timetags = timetags:sub(opt.start_offset+1,-1)
  end
end

-- Once we have loaded the raw inputs we can build the desired features/labels from them
-- The raw inputs will contain 2 cols for the week and day times plus 4 cols per symbol
-- so we can extract the number of symbols:
local nsym = utils:getNumSymbols(opt,raw_inputs)
log:debug("Detected ",nsym," symbols in raw inputs.")


opt.num_input_symbols = nsym

-- Now we can build the features tensor:
local features, timetags = utils:generateLogReturnFeatures(opt, raw_inputs, timetags)

-- From the features, we can build the labels tensor:
-- not that this method will also change the features tensor.
local features, labels, timetags = utils:generateLogReturnLabels(opt, features, timetags)
CHECK(features:size(1)==labels:size(1),"Mismatch in features/labels sizes")

-- Later in the training we train on train_size samples, then eval on eval_size
-- then train again, then eval again etc... right now we can already compute what should
-- be the appropriate size of the features/labels to use:
local nsamples = features:size(1)

-- Note that: if we use minibatch training then we must adapt the training size:
if opt.batch_size > 0 then
  opt.train_size = opt.batch_size*opt.batch_num_seqs*opt.seq_length + (opt.batch_num_seqs*opt.seq_length - 1)
  log:debug("Using minibatch, updated training size to: ", opt.train_size)
end

local nsessions = math.floor((nsamples - opt.train_size)/opt.eval_size)
log:debug("Can perform at most ", nsessions, " training/eval sessions")

-- We can now enter the train/eval loop on the features/labels:
-- we should train as long as
nsessions = opt.max_sessions < 0 and nsessions or opt.max_sessions

nsamples = opt.train_size + nsessions*opt.eval_size

-- we keep only the needed features/labels:
log:debug("Cut features/labels to ", nsamples, " samples")
features = features:sub(1,nsamples)
labels = labels:sub(1,nsamples)
if timetags then
  timetags = timetags:sub(1,nsamples)
  CHECK(timetags:size(1)==features:size(1),"Mismatch with timetags size.")
end

log:debug("Prediction offset: ", opt.train_size + opt.feature_offset)

-- Once the features and labels are ready we can build the RNN prototype since
-- we have the number of inputs and outputs:
opt.num_inputs = features:size(2)
opt.num_outputs = opt.num_classes

-- Create the RNN prototype:
local proto = utils:createPrototype(opt)

-- Create the init state:
-- Note that the batch_size will be limited to one in this implementation:
local init_state = utils:createInitState(opt)

-- We also keep a reference on a global init state table:
local global_init_state = utils:cloneList(init_state)

-- also prepare a dedicated evaluation state:
local global_eval_state = utils:createInitState(opt,1)

-- Perform parameter initialization:
local params, grad_params = utils:initParameters(opt, proto)

-- Generate the clones from the prototype:
local clones = utils:generateClones(opt, proto)

local tdesc = {}
tdesc.raw_features = features
tdesc.raw_labels = labels
tdesc.params = params
tdesc.grad_params = grad_params
tdesc.init_state = init_state
tdesc.train_offset = 0
tdesc.clones = clones
tdesc.timetags = timetags

local timer = torch.Timer()

-- keep a backup of the max epochs:
local max_epochs = opt.max_epochs

local suffix = opt.suffix

local total_elapsed = 0
for i=1,nsessions do
  
  local itimer = torch.Timer()

  log:debug("Performing session ",i,"...")

  opt.max_epochs = i==1 and opt.initial_max_epochs or max_epochs

  -- No init state influence between training sessions:
  tdesc.global_init_state = utils:cloneList(global_init_state,true)

  -- For now try to keep the global eval state for subsequent calls:
  tdesc.global_eval_state = utils:cloneList(global_eval_state,true)

  utils:performTrainSession(opt, tdesc)
  
  tdesc.train_offset = tdesc.train_offset + opt.eval_size

  -- Now we should write the result arrays:
  utils:writeArrays("misc/eval_results_" .. suffix .. ".csv",
    {tdesc.timetags_values,tdesc.evalidx_values,tdesc.pred_values,tdesc.label_values},
    {"timetag","eval_index","prediction","label"})

  utils:writeArray("misc/train_losses_" .. suffix .. ".csv", tdesc.train_losses)
  utils:writeArray("misc/eval_losses_" .. suffix .. ".csv", tdesc.eval_losses)
  utils:writeArray("misc/correct_signs_" .. suffix .. ".csv", tdesc.correct_signs)

  -- Also correct the number of epochs from the first iteration if applicable:
  local elapsed = itimer:time().real * max_epochs/opt.max_epochs

  total_elapsed = total_elapsed + elapsed
  local meantime = total_elapsed/i

  -- Check how many steps are left:
  local left = (nsessions - i)*meantime
  local hours = math.floor(left/3600)
  left = left - hours * 3600
  local mins = math.floor(left/60)
  left = left - mins * 60
  local secs = math.ceil(left)
  log:debug(("Estimated time left to completion: %02d:%02d:%02d"):format(hours,mins,secs))
end

print("Training done in ",timer:time().real .. ' seconds')
