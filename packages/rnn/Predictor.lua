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

  if opt.data_dir then
    self:debug("Loading data from ", opt.data_dir)
    -- open the data file:
    self._dataFile = io.open(opt.data_dir.."/raw_inputs.csv","r")
    CHECK(self._dataFile,"Invalid data file")

    -- Discard the header line:
    self._dataFile:read()

    -- Perform initialization right here:
    self:handleInit{9}

    -- Assign the global mean and devs values:
    self:setFeatureMeanDev(1,1.1167946922797e-06, 0.0002137926639989)
    self:setFeatureMeanDev(2,1.0937851584458e-05, 0.00068201060639694)
    self:setFeatureMeanDev(3,1.0904000997543, 0.0070609608665109)
    self:setFeatureMeanDev(4,1.1133430461996e-06, 0.00010457327880431)
    self:setFeatureMeanDev(5,1.1144239806526e-06, 7.5600197305903e-05)
    self:setFeatureMeanDev(7,-1.3221814469944e-06, 0.00023287530348171)
    self:setFeatureMeanDev(8,-1.4317037312139e-05, 0.00061438168631867)
    self:setFeatureMeanDev(9,0.72002333402634, 0.0079078543931246)
    self:setFeatureMeanDev(10,-1.3234327980172e-06, 0.0001018301336444)
    self:setFeatureMeanDev(11,-1.317716282756e-06, 7.150272722356e-05)
    self:setFeatureMeanDev(13,-1.256926680071e-06, 0.00014564374578185)
    self:setFeatureMeanDev(14,-1.2687735761574e-05, 0.0004287903138902)
    self:setFeatureMeanDev(15,1.4894667863846, 0.018019337207079)
    self:setFeatureMeanDev(16,-1.2421646715666e-06, 6.6827808041126e-05)
    self:setFeatureMeanDev(17,-1.2356510978861e-06, 4.8035315558081e-05)
    self:setFeatureMeanDev(19,-2.3461093690003e-07, 0.00025856320280582)
    self:setFeatureMeanDev(20,-3.9922815631144e-06, 0.00065629271557555)
    self:setFeatureMeanDev(21,0.67469322681427, 0.0076227253302932)
    self:setFeatureMeanDev(22,-2.3637034018975e-07, 0.00011158806591993)
    self:setFeatureMeanDev(23,-2.3277350180706e-07, 7.7527816756628e-05)
    self:setFeatureMeanDev(25,2.0155125639576e-06, 0.00017177524568979)
    self:setFeatureMeanDev(26,2.0215464246576e-05, 0.00043567316606641)
    self:setFeatureMeanDev(27,1.385294675827, 0.015465151518583)
    self:setFeatureMeanDev(28,2.0162792679912e-06, 7.362956966972e-05)
    self:setFeatureMeanDev(29,2.0165434762021e-06, 5.0995357014472e-05)
    self:setFeatureMeanDev(31,-1.133232558459e-06, 0.00021026007016189)
    self:setFeatureMeanDev(32,-1.0959840437863e-05, 0.00065794604597613)
    self:setFeatureMeanDev(33,0.99289971590042, 0.007167172152549)
    self:setFeatureMeanDev(34,-1.1440915841376e-06, 0.00010107940033777)
    self:setFeatureMeanDev(35,-1.1457550499472e-06, 7.286367326742e-05)
    self:setFeatureMeanDev(37,-1.6480402109664e-06, 0.00016235667862929)
    self:setFeatureMeanDev(38,-1.5622450519004e-05, 0.00042902864515781)
    self:setFeatureMeanDev(39,120.65016937256, 1.3561074733734)
    self:setFeatureMeanDev(40,-1.6354434819732e-06, 7.0794332714286e-05)
    self:setFeatureMeanDev(41,-1.6214153220062e-06, 4.9545655201655e-05)
  end

  if opt.with_zmq then
    -- Create a new socket as a server:
    self._socket = zmq.socket(zmq.PAIR)

    self._socket:bind("tcp://*:"..opt.local_port)
  end

  -- Start the app:
  self:run()

  -- When done close the data file if any:
  if self._dataFile then
    self._dataFile:close()
  end
end

--[[
Function: setFeatureMeanDEv

Assign the mean and deviation values to use for a given feature:
]]
function Class:setFeatureMeanDev(fid, mean, dev)
  self.opt.feature_means = self.opt.feature_means or {}
  self.opt.feature_devs = self.opt.feature_devs or {}

  self.opt.feature_means[fid] = mean
  self.opt.feature_devs[fid] = dev
end

--[[
Function: sendData

Method used to send data back to MT5
]]
function Class:sendData(data)
  if self._socket then
    self._socket:send(table.concat(data,","))
  end
end

--[[
Function: receiveMessage

Receive a message from socket
]]
function Class:receiveMessage()
  if self._socket then
    return self._socket:receive()
  end
end

--[[
Function: hasEnoughSamples

Used to check if we received enough samples already to start training/predicting.
]]
function Class:hasEnoughSamples()
  return self._numSamples >= self._rawInputSize
end

--[[
Function: performTraining

Method used to check if we can perform a training session
]]
function Class:performTraining()
  -- self:debug("Entering performTraining()")

  -- if we don't have enough samples, we don't train anything:
  if not self:hasEnoughSamples() then
    self:debug("Not performing training: only have ", self._numSamples," samples")
    return
  end

  -- Check if we should perform a training:
  if ((self._numSamples - self._rawInputSize) % self._trainFreq) ~= 0 then
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

  self:debug("Training network ", self._trainNet)

  -- Now that we have a network, we can request a training on it
  net:train(self._features, self._labels, self._timetags)
end

--[[
Function: getNetworkPrediction

Retrieve the current prediction of a given network for the
current evaluation features:
]]
function Class:getNetworkPrediction(network)
  CHECK(network:isReady(),"Network ", network:getID()," is not ready!")
  CHECK(self._evalFeatures,"Invalid evaluation features.")

  -- Otherwise we can request the prediction:
  return network:evaluate(self._evalFeatures)
end

--[[
Function: getPrediction

Method used to retrieve the current prediction
]]
function Class:getPrediction(tag)
  -- Iterate on all the available networks:
  local result = 0.0;
  local count = 0;

  local preds = {}
  for k,net in ipairs(self._nets) do
    local pred = self:getNetworkPrediction(net)
    if pred then
      table.insert(preds,pred)
      pred = (pred-0.5)*2.0
      self:debug("Prediction from network ", k,": ",pred)
      result = result + pred;
      count = count + 1
    else
      table.insert(preds,0.0)
    end
  end

  -- Complete the non created nets:
  local num = #preds
  for i=num+1,self._numNets do
    table.insert(preds,0.0)
  end

  self:writePrediction(tag,preds)

  result = count==0 and 0.0 or result/count
  self:debug("Global prediction: ",result)

  return result
end

--[[
Function: writeRawInput

Write the raw inputs to a file
]]
function Class:writeRawInput(tag,data)
  -- Open the file for writing:
  local f = io.open("misc/" .. self.opt.suffix .. "_raw_inputs.csv","aw")
  local msg = tag.. "," .. table.concat(data,",") .. "\n"
  -- self:debug("Writing raw input line: ", msg)
  f:write(msg)
  f:close()
end

--[[
Function: writePrediction

Method used to write the predictions that this predictor is generating:
]]
function Class:writePrediction(tag, preds)
  -- Open the file for writing:
  local f = io.open("misc/" .. self.opt.suffix .. "_predictions.csv","aw")
  local msg = tag.. "," .. table.concat(preds,",") .. "\n"
  self:debug("Writing prediction line: ", msg)
  f:write(msg)
  f:close()  
end

--[[
Function: writeFeatures

Method used to write the features that this predictor is generating:
]]
function Class:writeFeatures(tag,features)
  -- Open the file for writing:
  local f = io.open("misc/" .. self.opt.suffix .. "_features.csv","aw")
  local msg = tag..""

  local len = features:size(1)
  for i=1,len do
    msg = msg .. "," .. features[i]
  end

  -- self:debug("Writing feature line: ", msg)
  f:write(msg)
  f:close()  
end

--[[
Function: handleSingleInput

Method used to handle a jsut received single input
]]
function Class:handleSingleInput(data)
  CHECK(self._rawInputs and self._rawTimetags,"Not initialized yet")

  -- The data should contain the timetag, followed by the feature values.
  local tag = table.remove(data,1)
  -- self:debug("Received timetag: ", tag)
  -- self:debug("Received data: ", data)
  CHECK(#data==self._numRawInputs,"Mismatch in number of features")

  self:writeRawInput(tag,data)

  -- move the previous samples one "step up":
  self._rawTimetags[{{1,-2}}] = self._rawTimetags[{{2,-1}}]
  self._rawInputs[{{1,-2},{}}] = self._rawInputs[{{2,-1},{}}]

  -- Write the data into the last row:
  self._rawTimetags[-1] = tonumber(tag);
  for i=1,self._numRawInputs do 
    self._rawInputs[-1][i] = tonumber(data[i])
  end

  -- Update the feature tensor:
  self:updateFeatures(1)

  -- check if we can perform a training session:
  self:performTraining()

  -- prepare the current timetag:

  -- retrieve the current prediction if any:
  local pred = self:getPrediction(tag)

  -- self:debug("Current prediction: ",pred)

  -- Now we need to send a prediction back:
  self:sendData{"prediction",tag,pred}
end

--[[
Function: handleMultiInputs

Method used to handle the reception of multiple inputs
]]
function Class:handleMultiInputs(data)
  CHECK(self._rawInputs and self._rawTimetags,"Not initialized yet")
  local nrows = tonumber(table.remove(data,1))
  local nf = tonumber(table.remove(data,1))-1

  self:debug("Received multiple inputs: ", nrows,"x",nf+1)
  CHECK(nf==self._numRawInputs,"Mismatch in number of features: ",nf,"!=",self._numRawInputs)
  CHECK(nrows<=self._rawInputSize,"Received too many samples: ",nrows,">=",self._rawInputSize)

  if nrows<self._rawInputSize then
    self._rawTimetags[{{1,-1-nrows}}] = self._rawTimetags[{{1+nrows,-1}}]
    self._rawInputs[{{1,-1-nrows},{}}] = self._rawInputs[{{1+nrows,-1},{}}]
  end

  -- Read all the data in a tensor:
  -- We can use a sub part of the raw inputs/timetags tensors:

  local sub_inputs = self._rawInputs:narrow(1,self._rawInputSize-nrows+1,nrows)
  local sub_timetags = self._rawTimetags:narrow(1,self._rawInputSize-nrows+1,nrows)

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
  self:updateFeatures(nrows)

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
  local ni = tonumber(data[1])
  self:debug("Initializing with ",ni," raw inputs")

  local opt = self.opt

  -- The raw input size should account for the warm_up period and the last
  -- rows that will be removed from the features to built the labels:
  self._rawInputSize = opt.train_size + opt.warmup_offset + opt.label_offset
  self:debug("Raw input size: ", self._rawInputSize)

  -- Also store the number of features:
  self._numRawInputs = ni

  -- But the raw input tensor:
  self:debug("Creating raw input tensor of size ", self._rawInputSize , "x", ni )
  self._rawInputs = torch.Tensor(self._rawInputSize, ni):zero()

  -- Also create the timetag tensor here:
  self:debug("Creating timetag tensor of size ", self._rawInputSize )
  self._rawTimetags = torch.LongTensor(self._rawInputSize):zero()

  -- We also store the number of samples received to
  -- perform the training in a timed fashion:
  self._numSamples = 0

  -- Also ensure that we discard the previous nets:
  self:debug("Discarding previous networks")
  self._nets = {}

  -- Reset the number of input/outputs:
  self.opt.num_inputs = nil
  self.opt.num_outputs = nil

  -- Reset the eval tensor:
  self._evalFeatures = nil
  self._evalTimetags = nil

  -- Reset number of features:
  self._nf = nil

  -- Send a reply to state that we need train_size samples to start training
  self:debug("Sending request for ", opt.train_size, " training samples.");
  self:sendData{"request_samples",opt.train_size}
  -- self:sendData{"request_samples",10}
end

--[[
Function: getNumRawInputs

Method used to retrieve the number of features
]]
function Class:getNumRawInputs()
  CHECK(self._numRawInputs,"Not initialized yet.")
  return self._numRawInputs
end

--[[
Function: getNumFeatures

Retrieve the number of features if available
]]
function Class:getNumFeatures()
  CHECK(self._nf,"Number of features not initialized.")
  return self._nf
end

--[[
Function: updateFeatures

Method used to update the current features/timetags
This is done each time we receive a new sample raw, provided we
have enough data
]]
function Class:updateFeatures(num)

  -- self:debug("Entering updateFeatures()")

  -- Increment the sample count:
  self._numSamples = self._numSamples + num

  if not self:hasEnoughSamples() then
    -- We cannot prepare the features yet.
    return;
  end

  self:debug("Updating features at num samples ", self._numSamples)

  local opt = self.opt
  opt.num_input_symbols = self._rawInputs:size(2)-2

  -- first we need to generate the features from the raw inputs
  -- Now we can build the features tensor:
  local features, timetags = utils:generateLogReturnFeatures(opt, self._rawInputs, self._rawTimetags)
  CHECK(features,"Invalid features.")

  -- Assign the number of features:
  self._nf = features:size(2)

  -- Also update the number of inputs/outputs for the networks here:
  self.opt.num_inputs = self:getNumFeatures()
  self.opt.num_outputs = self.opt.num_classes

  -- Prepare the evaluation features:
  local writeAll = false
  if not self._evalFeatures then
    self._evalFeatures = torch.Tensor(opt.seq_length,self:getNumFeatures()):zero()
    -- Preprocess this tensor on GPU if needed:
    self._evalFeatures = utils:prepro(opt,self._evalFeatures)
    writeAll = true
  end

  if not self._evalTimetags then
    self._evalTimetags = torch.LongTensor(opt.seq_length):zero()
  end

  -- CRITICAL part:
  -- When we generate the labels below, the last row of the features and timetags tensor
  -- will be removed, because we cannot train on that row yet, since we don't have any label.
  -- yet, this is the latest input we received, and thus the end of the feature/timetags tensor
  -- is at this specific point the inputs sequence we want to use for evaluation!
  -- Thus, before removing this row we want to copy the current features/timetag for evaluation
  -- later:

  -- self:debug("EvalFeatures size: ", self._evalFeatures:size())
  -- self:debug("Features subset size: ", (features[{{-opt.seq_length,-1},{}}]):size())

  self._evalFeatures[{}] = features[{{-opt.seq_length,-1},{}}]
  self._evalTimetags[{}] = timetags[{{-opt.seq_length,-1}}]
  local nsamples1 = features:size(1)

  -- we write the last line of the eval tensor to file:
  if writeAll then
    local len = features:size(1)
    for i=1,len do 
      self:writeFeatures(timetags[i],features[{i,{}}])
    end
  else
    self:writeFeatures(self._evalTimetags[-1],self._evalFeatures[{-1,{}}])
  end

  -- From the features, we can build the labels tensor:
  -- not that this method will also change the features tensor.
  local features, labels, timetags = utils:generateLogReturnLabels(opt, features, timetags)
  CHECK(features:size(1)==labels:size(1),"Mismatch in features/labels sizes")
  CHECK(features:size(1)==timetags:size(1),"Mismatch in features/timetags sizes")

  local nsamples = features:size(1)
  CHECK(nsamples == (nsamples1-opt.label_offset),"Features tensor was not resized as expected.")
  
  -- The number of samples we have should match the desired train size:
  CHECK(nsamples==opt.train_size,"Number of samples doesn't match train_size: ", nsamples,"!=",opt.train_size)  

  -- Keep references on those arrays:
  self._features = features;
  self._timetags = timetags;
  self._labels = labels
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
    msg = self:receiveMessage()
    active = false
    while msg do
      active = true
      -- self:debug("Received message: ",msg)
      
      -- we have to handle this message:
      self:dispatchMessage(msg)

      msg = self:receiveMessage()
    end

    if self._dataFile then
      active = true
      self:readDataInput()
    end

    if not active then
      -- Sleep for a short time if there was no activity at all:
      sys.sleep(0.002)
    end
  end
end

--[[
Function: readDataInput

Read an input from the data file
]]
function Class:readDataInput()
  CHECK(self._dataFile, "Invalid data file")

  local line = self._dataFile:read()
  if not line then
    -- Stop running the processor:
    self._running = false;
    return
  end

  -- We have a valid line so we have to send it for processing:
  -- self:debug("Sending single line for processing: ", line)
  local data = utils:split(line)
  self:handleSingleInput(data)
end

return Class


