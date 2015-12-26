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
cmd:option('-seq_length',100,'number of timesteps to unroll for')

cmd:option('-forcast_symbol',1.0,'Symbol that should be forcasted.')
cmd:option('-num_classes',1,'Number of classes to consider when performing classification.')
cmd:option('-train_size',2000,'Number of steps used for each training session')
cmd:option('-eval_size',100,'Number of steps used for each evaluation session')
cmd:option('-max_sessions',20,'Max number of training/eval sessions to perform')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')

cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-max_epochs',1.0,'number of full passes through the training data')

cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-ema_adaptation',0.001,'Moving average adaptation factor')

cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- Setup the common GPU setup elements:
utils:setupGPU(opt)

-- Before we can create a prototype we must know what will be the input/output sizes
-- for the network
-- And thus we must load the data
local raw_inputs = utils:loadRawInputs(opt)

-- Once we have loaded the raw inputs we can build the desired features/labels from them
-- The raw inputs will contain 2 cols for the week and day times plus 4 cols per symbol
-- so we can extract the number of symbols:
local nsym = utils:getNumSymbols(raw_inputs)
log:debug("Detected ",nsym," symbols in raw inputs.")


-- Now we can build the features tensor:
local features = utils:generateLogReturnFeatures(opt, raw_inputs)

-- From the features, we can build the labels tensor:
-- not that this method will also change the features tensor.
local features, labels = utils:generateLogReturnLabels(opt, features)
CHECK(features:size(1)==labels:size(1),"Mismatch in features/labels sizes")


-- Later in the training we train on train_size samples, then eval on eval_size
-- then train again, then eval again etc... right now we can already compute what should
-- be the appropriate size of the features/labels to use:
local nsamples = features:size(1)
local nsessions = math.floor((nsamples - opt.train_size)/opt.eval_size)
log:debug("Can perform at most ", nsessions, " training/eval sessions")
nsamples = opt.train_size + nsessions*opt.eval_size

-- we keep only the needed features/labels:
log:debug("Cut features/labels to ", nsamples, " samples")
features = features:sub(1,nsamples)
labels = labels:sub(1,nsamples)


-- Once the features and labels are ready we can build the RNN prototype since
-- we have the number of inputs and outputs:
opt.num_inputs = features:size(2)
opt.num_outputs = opt.num_classes

-- Create the RNN prototype:
local proto = utils:createPrototype(opt)

-- Create the init state:
-- Note that the batch_size will be limited to one in this implementation:
opt.batch_size = 1
local init_state = utils:createInitState(opt)

-- We also keep a reference on a global init state table:
local global_init_state = utils:cloneList(init_state)


-- Perform parameter initialization:
local params, grad_params = utils:initParameters(opt, proto)


-- Generate the clones from the prototype:
local clones = utils:generateClones(opt, proto)

-- We can now enter the train/eval loop on the features/labels:
-- we should train as long as
nsessions = opt.max_sessions < 0 and nsessions or opt.max_sessions

local tdesc = {}
tdesc.features = features
tdesc.labels = labels
tdesc.params = params
tdesc.grad_params = grad_params
tdesc.init_state = init_state
tdesc.train_offset = 0
tdesc.clones = clones

local timer = torch.Timer()

for i=1,nsessions do
  log:debug("Performing session ",i,"...")

  -- No init state influence between training sessions:
  tdesc.global_init_state = utils:cloneList(global_init_state,true)

  utils:performTrainSession(opt, tdesc)
  

  tdesc.train_offset = tdesc.train_offset + opt.eval_size

  -- Now we should write the result arrays:
  utils:writeArray("train_losses.csv", tdesc.train_losses)
  utils:writeArray("eval_losses.csv", tdesc.eval_losses)
  utils:writeArray("correct_signs.csv", tdesc.correct_signs)
end

print("Training done in ",timer:time().real .. ' seconds')