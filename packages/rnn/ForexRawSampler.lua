local Class = createClass{name="ForexRawSampler",bases={"rnn.SamplerBase","rnn.ForexRawHandler"}};

--[[
Class: utils.ForexRawSampler

Char Sampler class.

This class inherits from <rnn.SamplerBase>.
]]

--[=[
--[[
Constructor: ForexRawSampler

Create a new instance of the class.

]]
function ForexRawSampler(options)
]=]
function Class:initialize(options)
  self._sep = "\n"

  self._rcfg = self._checkpoint.config
  CHECK(self._rcfg,"Invalid config for Raw dataset")
end

--[[
Function: sample

Perform the sampling using an input file
]]
function Class:sample()

  local features, labels = self:loadDataset()

  local prev_char
  local protos = self._prototype

  local len = features:size(1)
  local nf = features:size(2)

  self:debug("Sampling on ",len," steps...")

  local sample = 0

  local prediction
  local input = torch.Tensor(1,nf):zero()

  if opt.gpuid >= 0 and opt.opencl == 0 then input = input:cuda() end
  if opt.gpuid >= 0 and opt.opencl == 1 then input = input:cl() end  

  -- We also prepare a tensor to store the prediction values:
  local preds = torch.Tensor(len):zero()

  -- start sampling/argmaxing
  for i=1, len do
    -- Take one row from the features:
    input[{1,{}}] = features[{i,{}}]

    -- forward the rnn for next character
    local lst = protos.rnn:forward{input, unpack(self._currentState)}
    self._currentState = {}
    for i=1,self._stateSize do table.insert(self._currentState, lst[i]) end
    prediction = lst[#lst] -- last element holds the log probabilities or regression value

    -- if sample == 0 then
        -- use argmax
        local _, prev_class_ = prediction:max(2)
        local pred = prev_class_:storage()[1]
    -- else
    --     -- use sampling
    --     self._prediction:div(opt.temperature) -- scale by temperature
    --     local probs = torch.exp(self._prediction):squeeze()
    --     probs:div(torch.sum(probs)) -- renormalize so probs sum to one
    --     prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
    -- end

    -- self:debug("Prediction dims: ",prediction:size())
    -- self:debug("Prediction: ",prediction[1][1])
    preds[i] = pred

    self:debug("Prediction at ",i,"/",len,": ",pred)

    table.insert(self._result, pred)
  end

  --  Now compare with the labels:
  labels = labels:narrow(2,1,1)

  -- compute the overall error:
  local correct = torch.sum(torch.eq(labels,preds))/len
  self:debug("Overall error: ", 1.0 - correct)

  -- Also compute the efficiency of the predictions:
  -- eg. how often we have the "proper direction" for the prediction:
  -- We need to know how many classes we have:
  local nclasses = self._checkpoint.config.num_classes
  self:debug("Number of classes used: ", nclasses)
  CHECK(nclasses%2==0,"Not even number of classes...")
  local half = nclasses/2

  local mres = torch.cmul(labels - half - 0.5,preds - half - 0.5)
  
  -- This will set the elements to 1 if mres[i] > 0 and 0 otherwise
  -- we need to count the number of 1 to get the number of appropriate predictions.
  local lres = torch.gt(mres,0)
  local goodPreds = lres:sum()

  -- Now we can compute the precision ratio (eg. efficiency)
  local eff = goodPreds/lres:size(1)
  self:debug("Overall efficiency: ",eff)
end

return Class


