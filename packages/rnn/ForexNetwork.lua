local Class = createClass{name="ForexNetwork",bases={"base.Object"}};

local utils = require "rnn.Utils"

--[[
Class: rnn.ForexNetwork
]]

--[=[
--[[
Constructor: ForexNetwork

Create a new instance of the class.
]]
function ForexNetwork(options)
]=]
function Class:initialize(options)
  CHECK(options.id,"Invalid ForexNetwork ID")
  CHECK(options.parent,"Invalid parent")
  CHECK(options.opt,"Invalid opt")

  self._id = options.id
  self:debug("Creating a ForexNetwork instance with id", self._id)

  self.opt = options.opt
  self._parent = options.parent

  -- Current train session:
  self._session = 0

  -- Network is not ready initially:
  self._isReady = false

  local opt = self.opt

  -- Create the RNN prototype:
  self:debug("Creating RNN prototype")
  self._proto = utils:createPrototype(opt)

  -- Create the init state:
  self:debug("Creating init state")
  self._initState = utils:createInitState(opt)

  -- We also keep a reference on a global init state table:
  self:debug("Creating global train state")
  self._globalTrainState = utils:cloneList(self._initState)

  -- also prepare a dedicated evaluation state:
  self:debug("Creating global eval state")
  self._globalEvalState = utils:createInitState(opt,1)

  -- Perform parameter initialization:
  self:debug("Retrieving network parameters...")  
  self._params, self._gradParams = utils:initParameters(opt, self._proto)

  -- Generate the clones from the prototype:
  self:debug("Generating RNN clones from prototypes")
  self._clones = utils:generateClones(opt, self._proto)
end

--[[
Function: isReady

Check if this network is ready for usage
]]
function Class:isReady()
  return self._isReady
end

--[[
Function: train

Perform the training on the given inputs
]]
function Class:train(features,labels,timetags)
  if self._isTraining then
    self:debug("Network already training.")
    return
  end

  if self._session == 0 then
    self:debug("Should perform initial training here")
  end

  self._session = self._session + 1
  self:debug("Training on session ", self._session)

  local tdesc = {}
  tdesc.raw_features = features
  tdesc.raw_labels = labels
  tdesc.timetags = timetags
  tdesc.params = self._params
  tdesc.grad_params = self._gradParams
  tdesc.init_state = self._initState
  tdesc.train_offset = 0
  tdesc.clones = self._clones

end

return Class


