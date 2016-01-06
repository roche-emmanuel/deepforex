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
  self:debug("Creating a ForexNetwork instance with id")

  self.opt = options.opt
  self._parent = options.parent

  -- Create the RNN prototype:
  self._proto = utils:createPrototype(self.opt)
end

--[[
Function: train

Perform the training on the given inputs
]]
function Class:train()
  self:warn("Network:train() not implemented yet.")
end

return Class


