------------------------------------------------------------------------
--[[ Recurrence ]]--
-- A general container for implementing a recurrence.
-- Unlike Recurrent, this module doesn't manage a separate input layer,
-- nor does it have a startModule. Instead for the first step, it
-- just forwards a zero tensor through the recurrent layer (like LSTM).
-- The recurrentModule should output Tensor or table : output(t)
-- given input table : {input(t), output(t-1)}
------------------------------------------------------------------------
local _ = require 'moses'
local Recurrence, parent = torch.class('nn.Recurrence', 'nn.AbstractRecurrent')

function Recurrence:__init(recurrentModule, outputSize, nInputDim, rho)
   parent.__init(self, rho or 9999)

   assert(_.contains({'table','torch.LongStorage','number'}, torch.type(outputSize)), "Unsupported size type")
   self.outputSize = torch.type(outputSize) == 'number' and {outputSize} or outputSize
   -- for table outputs, this is the number of dimensions in the first (left) tensor (depth-first).
   assert(torch.type(nInputDim) == 'number', "Expecting nInputDim number for arg 2")
   self.nInputDim = nInputDim
   assert(torch.isTypeOf(recurrentModule, 'nn.Module'), "Expecting recurrenModule nn.Module for arg 3")
   self.recurrentModule = recurrentModule

   -- make it work with nn.Container and nn.Decorator
   self.module = self.recurrentModule
   self.modules[1] = self.recurrentModule
   self.sharedClones[1] = self.recurrentModule

   -- just so we can know the type of this module
   self.typeTensor = torch.Tensor()
end

-- recursively creates a zero tensor (or table thereof) (or table thereof).
-- This zero Tensor is forwarded as output(t=0).
function Recurrence:recursiveResizeZero(tensor, size, batchSize)
   local isTable = torch.type(size) == 'table'
   if isTable and torch.type(size[1]) ~= 'number' then
      tensor = (torch.type(tensor) == 'table') and tensor or {}
      for k,v in ipairs(size) do
         tensor[k] = self:recursiveResizeZero(tensor[k], v, batchSize)
      end
   elseif torch.type(size) == 'torch.LongStorage'  then
      local size_ = size:totable()
      tensor = torch.isTensor(tensor) and tensor or self.typeTensor.new()
      if batchSize then
         tensor:resize(batchSize, unpack(size_))
      else
         tensor:resize(unpack(size_))
      end
      tensor:zero()
   elseif isTable and torch.type(size[1]) == 'number' then
      tensor = torch.isTensor(tensor) and tensor or self.typeTensor.new()
      if batchSize then
         tensor:resize(batchSize, unpack(size))
      else
         tensor:resize(unpack(size))
      end
      tensor:zero()
   else
      error("Unknown size type : "..torch.type(size))
   end
   return tensor
end

-- get the batch size.
-- When input is a table, we use the first tensor (depth first).
function Recurrence:getBatchSize(input, nInputDim)
   local nInputDim = nInputDim or self.nInputDim
   if torch.type(input) == 'table' then
      return self:getBatchSize(input[1])
   else
      assert(torch.isTensor(input))
      if input:dim() == nInputDim then
         return nil
      elseif input:dim() - 1 == nInputDim then
         return input:size(1)
      else
         error("inconsitent tensor dims "..input:dim())
      end
   end
end

function Recurrence:getHiddenState(step, input)
   local prevOutput
   if step == 0 then
      if input then
         -- first previous output is zeros
         local batchSize = self:getBatchSize(input)
         self.zeroTensor = self:recursiveResizeZero(self.zeroTensor, self.outputSize, batchSize)
      end
      prevOutput = self.userPrevOutput or self.outputs[step] or self.zeroTensor
   else
      -- previous output of this module
      prevOutput = self.outputs[step]
   end
   -- call getHiddenState on recurrentModule as they may contain AbstractRecurrent instances...
   return {prevOutput, nn.Container.getHiddenState(self, step)}
end

function Recurrence:setHiddenState(step, hiddenState)
   assert(torch.type(hiddenState) == 'table')
   assert(#hiddenState >= 1)
   self.outputs[step] = hiddenState[1]

   if hiddenState[2] then
      -- call setHiddenState on recurrentModule as they may contain AbstractRecurrent instances...
      nn.Container.setHiddenState(self, step, hiddenState[2])
   end
end

function Recurrence:updateOutput(input)
   -- output(t-1)
   local prevOutput = self:getHiddenState(self.step-1, input)[1]

   -- output(t) = recurrentModule{input(t), output(t-1)}
   local output
   if self.train ~= false then
      self:recycle()
      local recurrentModule = self:getStepModule(self.step)
      -- the actual forward propagation
      output = recurrentModule:updateOutput{input, prevOutput}
   else
      output = self.recurrentModule:updateOutput{input, prevOutput}
   end

   self.outputs[self.step] = output

   self.output = output

   self.step = self.step + 1
   self.gradPrevOutput = nil
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil

   return self.output
end

function Recurrence:getGradHiddenState(step)
   local gradOutput
   if step == self.step-1 then
      gradOutput = self.userNextGradOutput or self.gradOutputs[step] or self.zeroTensor
   else
      gradOutput = self.gradOutputs[step]
   end
   return {gradOutput, nn.Container.getGradHiddenState(self, step)}
end

function Recurrence:setGradHiddenState(step, gradHiddenState)
   assert(torch.type(gradHiddenState) == 'table')
   assert(#gradHiddenState >= 1)

   self.gradOutputs[step] = gradHiddenState[1]
   if gradHiddenState[2] then
      nn.Container.setGradHiddenState(self, step, gradHiddenState[2])
   end
end

function Recurrence:_updateGradInput(input, gradOutput)
   assert(self.step > 1, "expecting at least one updateOutput")
   local step = self.updateGradInputStep - 1
   assert(step >= 1)

   -- set the output/gradOutput states of current Module
   local recurrentModule = self:getStepModule(step)

   -- backward propagate through this step
   local _gradOutput = self:getGradHiddenState(step)[1]
   self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], _gradOutput)
   nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
   gradOutput = self._gradOutputs[step]

   local gradInputTable = recurrentModule:updateGradInput({input, self:getHiddenState(step-1)[1]}, gradOutput)

   local _ = require 'moses'
   self:setGradHiddenState(step-1, _.slice(gradInputTable, 2, #gradInputTable))

   return gradInputTable[1]
end

function Recurrence:_accGradParameters(input, gradOutput, scale)
   local step = self.accGradParametersStep - 1
   assert(step >= 1)

   local recurrentModule = self:getStepModule(step)

   -- backward propagate through this step
   local gradOutput = self._gradOutputs[step] or self:getGradHiddenState(step)[1]
   recurrentModule:accGradParameters({input, self:getHiddenState(step-1)[1]}, gradOutput, scale)
end

Recurrence.__tostring__ = nn.Decorator.__tostring__
