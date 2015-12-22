local Class = createClass{name="SamplerBase",bases={"base.Object"}};

--[[
Class: utils.SamplerBase

Base Sampler class.

This class inherits from <base.Object>.
]]

--[=[
--[[
Constructor: SamplerBase

Create a new instance of the class.

Parameters:
  data_dir - directory where the raw input data is found
  batch_size - Size of the mini batch to use
  split_fractions - repartition of the data between training/validation/test
]]
function SamplerBase(options)
]=]
function Class:initialize(options)
  CHECK(options.model,"Invalid data dir")
  self.model_file = options.model

  CHECK(path.exists(self.model_file),"Invalid model file ",self.model_file)

  self._config = options

  self:debug("Loading checkpoint ", self.model_file)
  
  -- load the checkpoint data:
  self._checkpoint = torch.load(self.model_file)
  self._prototype = self._checkpoint.protos

  -- put in eval mode so that dropout works properly
  self._prototype.rnn:evaluate()

  self:initializeState()
end

--[[
Function: initializeState

Method used to initialize the RNN state
]]
function Class:initializeState()
  -- initialize the rnn state to all zeros
  local checkpoint = self._checkpoint

  self:debug('Creating an ' .. checkpoint.opt.model .. '...')
  local current_state
  current_state = {}
  for L = 1,checkpoint.opt.num_layers do
      -- c and h for all layers
      local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
      if opt.gpuid >= 0 and opt.opencl == 0 then h_init = h_init:cuda() end
      if opt.gpuid >= 0 and opt.opencl == 1 then h_init = h_init:cl() end
      table.insert(current_state, h_init:clone())
      if checkpoint.opt.model == 'lstm' then
          table.insert(current_state, h_init:clone())
      end
  end
  self._currentState = current_state
  self._stateSize = #current_state
end


return Class


