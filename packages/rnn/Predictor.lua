local Class = createClass{name="Predictor",bases={"base.Object"}};

local sys = require "sys"
local zmq = require "zmq"
local utils = require "rnn.Utils"

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

  self.opt = opt

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
Function: handleSingleInput

Method used to handle a jsut received single input
]]
function Class:handleSingleInput(data)
  CHECK(self._rawInputs and self._timetags,"Not initialized yet")

  -- The data should contain the timetag, followed by the feature values.
  local tag = table.remove(data,1)
  -- self:debug("Received timetag: ", tag)
  -- self:debug("Received data: ", data)

  -- Now we need to send a prediction back:
  self:sendData{"prediction",tag,0.0}
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

  self:debug("Received sub timetags: ",sub_timetags)
  self:debug("Received sub inputs: ",sub_inputs)
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

  self._trainSize = opt.train_size
  self._nf = nf

  -- Also create the timetag tensor here:
  self._timetags = torch.LongTensor(opt.train_size):zero()

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


