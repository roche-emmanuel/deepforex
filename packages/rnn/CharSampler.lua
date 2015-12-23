local Class = createClass{name="CharSampler",bases={"rnn.SamplerBase"}};

--[[
Class: utils.CharSampler

Char Sampler class.

This class inherits from <rnn.SamplerBase>.
]]

--[=[
--[[
Constructor: CharSampler

Create a new instance of the class.

]]
function CharSampler(options)
]=]
function Class:initialize(options)
  -- initialize the vocabulary (and its inverted version)
  self._vocab = self._checkpoint.vocab
  self._ivocab = {}
  self._vocabSize = 0
  for c,i in pairs(self._vocab) do 
    self._ivocab[i] = c 
    self._vocabSize = self._vocabSize + 1
  end

  local seed_text = options.primetext
  if #seed_text > 0 then
    self:seedInitialSteps(seed_text)
  else
    self:uniformSeed()
  end
end

--[[
Function: seedInitialSteps

Method used to seed some initial steps with provided text
]]
function Class:seedInitialSteps(text)
  local opt = self._config

  seld:debug('seeding with ' .. text)
  seld:debug('--------------------------')
  for c in text:gmatch'.' do
    local prev_char = torch.Tensor{self._vocab[c]}
    table.insert(self._result,self._ivocab[prev_char[1]])

    if opt.gpuid >= 0 and opt.opencl == 0 then prev_char = prev_char:cuda() end
    if opt.gpuid >= 0 and opt.opencl == 1 then prev_char = prev_char:cl() end
    local lst = protos.rnn:forward{prev_char, unpack(self._currentState)}
    -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
    self._currentState = {}
    for i=1,self._stateSize do table.insert(self._currentState, lst[i]) end
    self._prediction = lst[#lst] -- last element holds the log probabilities
  end  
end

--[[
Function: uniformSeed

Method used to perform uniform initialisation
]]
function Class:uniformSeed()
  local opt = self._config
  -- fill with uniform probabilities over characters (? hmm)
  self:debug('missing seed text, using uniform probability over first character')
  self:debug('--------------------------')
  self._prediction = torch.Tensor(1, #self._ivocab):fill(1)/(#self._ivocab)
  if opt.gpuid >= 0 and opt.opencl == 0 then self._prediction = self._prediction:cuda() end
  if opt.gpuid >= 0 and opt.opencl == 1 then self._prediction = self._prediction:cl() end  
end

--[[
Function: sample

Perform the sampling
]]
function Class:sample(length, sample)
  local prev_char
  local protos = self._prototype

  self:debug("Sampling...")

  -- start sampling/argmaxing
  for i=1, length do
    -- log probabilities from the previous timestep
    if sample == 0 then
        -- use argmax
        local _, prev_char_ = self._prediction:max(2)
        prev_char = prev_char_:resize(1)
    else
        -- use sampling
        self._prediction:div(opt.temperature) -- scale by temperature
        local probs = torch.exp(self._prediction):squeeze()
        probs:div(torch.sum(probs)) -- renormalize so probs sum to one
        prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
    end

    -- To feed the network we need to pass a tensor of size 1xvocab_size:
    local input = torch.Tensor(1,self._vocabSize):zero()
    -- local input = torch.Tensor(self._vocabSize):zero()
    input[{1,prev_char[1]}] = 1
    -- input[prev_char[1]] = 1

    if opt.gpuid >= 0 and opt.opencl == 0 then input = input:cuda() end
    if opt.gpuid >= 0 and opt.opencl == 1 then input = input:cl() end

    -- forward the rnn for next character
    -- local lst = protos.rnn:forward{prev_char, unpack(self._currentState)}
    local lst = protos.rnn:forward{input, unpack(self._currentState)}

    self._currentState = {}
    for i=1,self._stateSize do table.insert(self._currentState, lst[i]) end
    self._prediction = lst[#lst] -- last element holds the log probabilities

    table.insert(self._result,self._ivocab[prev_char[1]])
  end
end


return Class


