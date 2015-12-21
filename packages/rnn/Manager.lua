local Class = createClass{name="Manager",bases={"base.Object"}};

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'utils.misc'

--[[
Class: rnn.Manager

Manager class for RNN construction

This class inherits from <base.Object>.
]]

--[=[
--[[
Constructor: Manager

Create a new instance of the class.

Parameters:
   No parameter
]]
function Manager(options)
]=]
function Class:initialize(options)
	self:debug("Creating RNN Manager.")
	-- We use float tensors in this project:
	torch.setdefaulttensortype('torch.FloatTensor')
end


--[[
Function: setup

Setup for the common elements
]]
function Class:setup(opt)
	self:debug("Setting up RNN network system.")

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

return Class()
