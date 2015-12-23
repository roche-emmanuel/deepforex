local Class = createClass{name="ForexSampler",bases={"rnn.SamplerBase","rnn.ForexDatasetHandler"}};

--[[
Class: utils.ForexSampler

Char Sampler class.

This class inherits from <rnn.SamplerBase>.
]]

--[=[
--[[
Constructor: ForexSampler

Create a new instance of the class.

]]
function ForexSampler(options)
]=]
function Class:initialize(options)
  self._sep = "\n"
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

    -- self:debug("Prediction dims: ",prediction:size())
    -- self:debug("Prediction: ",prediction[1][1])
    local pred = prediction[1][1]
    preds[i] = pred

    self:debug("Prediction at ",i,"/",len,": ",pred)

    table.insert(self._result, pred)
  end

  --  Now compare with the labels:
  labels = labels:narrow(2,1,1)

  -- compute the MSE:
  local loss = torch.sum(torch.pow(labels - preds,2))/len
  self:debug("Overall loss: ", loss)

  -- Also compute the efficiency of the predictions:
  -- eg. how often we have the "proper direction" for the prediction:
  -- we just need to multiply the labels by the predictions element by element:
  local mres = torch.cmul(labels,preds)
  
  -- This will set the elements to 1 if mres[i] > 0 and 0 otherwise
  -- we need to count the number of 1 to get the number of appropriate predictions.
  local lres = torch.gt(mres,0)
  local goodPreds = lres:sum()

  -- Now we can compute the precision ratio (eg. efficiency)
  local eff = goodPreds/lres:size(1)
  self:debug("Overall efficiency: ",eff)
end

return Class


