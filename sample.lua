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

local man = require "rnn.Manager"
-- local Sampler = require "rnn.CharSampler"
local Sampler = require "rnn.ForexSampler"

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a FOREX trading agent')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
-- cmd:option('-output','result_gen.txt','File were to write the result')
cmd:option('-output','inputs/test_2007_01_to_2007_04/result_gen.txt','File were to write the result')
cmd:option('-data_dir','inputs/test_2007_01_to_2007_04','File were to write the result')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',2000,'number of characters to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- Setup the common elements:
man:setup(opt)

-- Build the RNN agent:
local sampler = Sampler(opt)

sampler:sample(opt.length,opt.sample)

sampler:writeResults(opt.output)

print("Sampling done.")
