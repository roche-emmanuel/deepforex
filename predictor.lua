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

-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm,gru or rnn')
cmd:option('-dropout',0.5,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')

-- app params
cmd:option('-suffix','vxx','suffix to append to all written files')
cmd:option('-local_port',30000,'Local port used for the socket connection')

cmd:option('-batch_size',80,'Number of sequences to train on in parallel or -1 if we use only sequential training')
cmd:option('-batch_num_seqs',1,'Number of consecutive sequences in each batch slice')
cmd:option('-seq_length',25,'number of timesteps to unroll for')
cmd:option('-num_emas',1,'Number of EMA features generated for each symbol')
cmd:option('-num_remas',2,'Number of return EMA features generated for each symbol')
cmd:option('-ema_base_period',5,'Base period for the EMA addition')
cmd:option('-rsi_period',9,'Period to use for RSI of 0 if disabled')
cmd:option('-log_return_offsets',"3",'list of comma separated offset values that should be used to compute additional log return features')
cmd:option('-warmup_offset',20,'Offset applied at the start of the features tensor before starting the training process')
cmd:option('-forcast_index',1,'Index of the feature that should be forcasted.')
cmd:option('-num_classes',1,'Number of classes to consider when performing classification.')

cmd:option('-num_networks',5,'Number of networks that should be trained in parallel by this predictor')
cmd:option('-train_frequency',1,'Frequency at which we try to perform a training (in number of samples received)')

-- Base setup options:
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')


-- data
cmd:option('-data_dir','inputs/raw_2004_01_to_2007_01','data directory. Should contain the file input.txt with input data')



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
cmd:option('-with_timetag',0,'Set this to 1 if the raw inputs dataset provides a timetag column')
cmd:option('-with_close_only',0,'Set this to 1 if the raw inputs dataset only provides the close prices for each symbol')
cmd:option('-start_offset',0,'Offset applied on raw inputs before training anything')

cmd:option('-optim','rmsprop','Optimization algorithm')

cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- Setup the common GPU setup elements:
utils:setupGPU(opt)

local Predictor = require "rnn.Predictor"

Predictor(opt)

log:debug("Predictor app done.")
