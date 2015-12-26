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

cmd:option('-forcast_symbol',1.0,'Symbol that should be forcasted.')
cmd:option('-num_classes',2,'Number of classes to consider when performing classification.')


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

-- Create the RNN prototype:
-- local proto = utils:createPrototype(opt)

print("Training done.")
