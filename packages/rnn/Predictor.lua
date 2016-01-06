local Class = createClass{name="Predictor",bases={"base.Object"}};

local sys = require "sys"
local zmq = require "zmq"
local utils = require "rnn.Utils"
local Network = require "rnn.ForexNetwork"

--[[
Class: rnn.Predictor
]]

--[=[
--[[
Constructor: Predictor

Create a new instance of the class.
]]
function Predictor(options)
]=]
function Class:initialize(opt)
  self:debug("Creating a Predictor instance.")
  CHECK(opt.local_port,"Invalid local port")
  CHECK(opt.batch_size > 0,"Invalid batch size value")
  CHECK(opt.num_networks>0,"Invalid number of networks")
  CHECK(opt.train_frequency>0,"Invalid training frequency")

  self.opt = opt

  -- Number of networks that should be trained in parallel by this model:
  self._numNets = opt.num_networks
  
  -- references on the nets:
  self._nets = {}

  -- ID of the network that should be trained
  self._trainNet = 0

  -- Frequency at which the networks should be trained:
  self._trainFreq = opt.train_frequency

  --  Setup the log system:
  local lm = require "log.LogManager"
  local FileLogger = require "log.FileLogger"
  local sys = require "sys"

  lm:addSink(FileLogger{file="predictor_".. opt.suffix ..".log"})

  -- Compute the required train size:
  opt.train_size = opt.batch_size*opt.batch_num_seqs*opt.seq_length + (opt.batch_num_seqs*opt.seq_length - 1)
  self:debug("Using minibatches with training size: ", opt.train_size)

  -- Create a new socket as a server:
  self._socket = zmq.socket(zmq.PAIR)

  self._socket:bind("tcp://*:"..opt.local_port)

  -- Start the app:
  self:run()
end

--[[
Function: sendData

Method used to send data back to MT5
]]
function Class:sendData(data)
  self._socket:send(table.concat(data,","))
end

--[[
Function: performTraining

Method used to check if we can perform a training session
]]
function Class:performTraining()
  -- if we don't have enough samples, we don't train anything:
  if self._numSamples < self._trainSize then
    self:debug("Not performing training: only have ", self._numSamples," samples")
    return
  end

  -- Check if we should perform a training:
  if ((self._numSamples - self._trainSize) % self._trainFreq) ~= 0 then
    -- No need to perform training: mismatch on train frequency
    return
  end

  self:debug("Performing training on num samples ",self._numSamples)

  -- Now we need to decide which of the networks should be updated:
  self._trainNet = self._trainNet+1
  if self._trainNet > self._numNets then
    -- loop back:
    self._trainNet = 1
  end

  -- Now we retrieve this network, or we create it:
  local net = self._nets[self._trainNet]
  if not net then
    -- Create a new network:
    net = Network{parent=self,opt=self.opt,id=self._trainNet}
    self._nets[self._trainNet] = net
  end

  -- Now that we have a network, we can request a training on it
  net:train(self._rawInputs, self._timetags)
end

--[[
Function: getPrediction

Method used to retrieve the current prediction
]]
function Class:getPrediction()
  -- TODO provide implementation
  return 0.0
end

--[[
Function: handleSingleInput

Method used to handle a jsut received single input
]]
function Class:handleSingleInput(data)
  CHECK(self._rawInputs and self._timetags,"Not initialized yet")

  -- The data should contain the timetag, followed by the feature values.
  local tag = table.remove(data,1)
  -- self:debug("Received timetag: ", tag)
  -- self:debug("Received data: ", data)
  CHECK(#data==self._nf,"Mismatch in number of features")

  -- move the previous samples one "step up":
  self._timetags[{{1,-2}}] = self._timetags[{{2,-1}}]
  self._rawInputs[{{1,-2},{}}] = self._rawInputs[{{2,-1},{}}]

  -- Write the data into the last row:
  self._timetags[-1] = tonumber(tag);
  for i=1,self._nf do 
    self._rawInputs[-1][i] = tonumber(data[i])
  end

  -- Increment the sample count:
  self._numSamples = self._numSamples + 1

  -- check if we can perform a training session:
  self:performTraining()

  -- retrieve the current prediction if any:
  local pred = self:getPrediction()

  self:debug("Current prediction: pred")

  -- Now we need to send a prediction back:
  self:sendData{"prediction",tag,pred}
end

--[[
Function: handleMultiInputs

Method used to handle the reception of multiple inputs
]]
function Class:handleMultiInputs(data)
  CHECK(self._rawInputs and self._timetags,"Not initialized yet")
  local nrows = tonumber(table.remove(data,1))
  local nf = tonumber(table.remove(data,1))-1

  self:debug("Received multiple inputs: ", nrows,"x",nf+1)
  CHECK(nf==self._nf,"Mismatch in number of features: ",nf,"!=",self._nf)
  CHECK(nrows<=self._trainSize,"Received too many samples: ",nrows,">=",self._trainSize)

  if nrows<self._trainSize then
    self._timetags[{{1,-1-nrows}}] = self._timetags[{{1+nrows,-1}}]
    self._rawInputs[{{1,-1-nrows},{}}] = self._rawInputs[{{1+nrows,-1},{}}]
  end

  -- Read all the data in a tensor:
  -- We can use a sub part of the raw inputs/timetags tensors:

  local sub_inputs = self._rawInputs:narrow(1,self._trainSize-nrows+1,nrows)
  local sub_timetags = self._timetags:narrow(1,self._trainSize-nrows+1,nrows)

  CHECK(#data==(nrows*(nf+1)),"Invalid data size: ", #data, "!=", nrows*(nf+1))

  -- self:debug("Received data: ",data)

  local idx = 1
  for i=1,nrows do
    sub_timetags[i] = tonumber(data[idx])
    idx = idx+1
    for j=1,nf do
      sub_inputs[i][j] = tonumber(data[idx])
      idx = idx+1
    end
  end

  -- self:debug("Received sub timetags: ",sub_timetags)
  -- self:debug("Received sub inputs: ",sub_inputs)

  -- Increment the sample count:
  self._numSamples = self._numSamples + nrows

  -- check if we can perform a training session:
  self:performTraining()  
end

--[[
Function: handleInit

Method used to perform the initialization of the predictor
]]
function Class:handleInit(data)
  -- For now we assume that the initialization data will only contain:
  -- 1. the number of features.
  local nf = tonumber(data[1])
  self:debug("Initializing with ",nf," features")

  local opt = self.opt

  -- But the raw input tensor:
  self:debug("Creating raw input tensor of size ", opt.train_size , "x", nf )
  self._rawInputs = torch.Tensor(opt.train_size, nf):zero()

  -- Also create the timetag tensor here:
  self._timetags = torch.LongTensor(opt.train_size):zero()

  self._trainSize = opt.train_size
  self._nf = nf

  -- Also update the number of inputs/outputs for the networks here:
  self.opt.num_inputs = nf
  self.opt.num_outputs = self.opt.num_classes

  -- We also store the number of samples received to
  -- perform the training in a timed fashion:
  self._numSamples = 0

  -- Also ensure that we discard the previous nets:
  self:debug("Discarding previous networks")
  self._nets = {}

  -- Send a reply to state that we need train_size samples to start training
  self:debug("Sending request for ", opt.train_size, " training samples.");
  self:sendData{"request_samples",opt.train_size}
  -- self:sendData{"request_samples",10}
end

local handlers = {
  init = Class.handleInit,
  single_input = Class.handleSingleInput,
  multi_inputs = Class.handleMultiInputs,
}

--[[
Function: dispatchMessage

Method used to dispatch a received message appropriately
]]
function Class:dispatchMessage(msg)
  -- First we have to split this message:
  local data = utils:split(msg)
  local cid = table.remove(data,1)
  -- self:debug("Handling command ID: ", cid)

  local handler = handlers[cid]
  CHECK(handler,"No handler available for ID: '",cid,"'")

  -- Call the handler:
  handler(self,data)
end

--[[
Function: run

Main method to run the predictor
]]
function Class:run()
  self:debug("Starting predictor app.")

  self._running = true
  local msg
  local active
  while self._running do 
    -- self:debug("On run iteration...")
    msg = self._socket:receive()
    active = false
    while msg do
      active = true
      -- self:debug("Received message: ",msg)
      
      -- we have to handle this message:
      self:dispatchMessage(msg)

      msg = self._socket:receive()
    end

    if not active then
      -- Sleep for a short time if there was no activity at all:
      sys.sleep(0.002)
    end
  end
end

return Class


