local _ = require 'moses'

assert(not nn.AbstractRecurrent, "update nnx package : luarocks install nnx")
local AbstractRecurrent, parent = torch.class('nn.AbstractRecurrent', 'nn.Container')

function AbstractRecurrent:__init(rho)
   parent.__init(self)
   
   self.rho = rho --the maximum number of time steps to BPTT
   
   self.fastBackward = true
   self.copyInputs = true
   self.copyGradOutputs = true
   
   self.inputs = {}
   self.outputs = {}
   self._gradOutputs = {}
   self.gradOutputs = {}
   self.scales = {}
   
   self.gradParametersAccumulated = false
   self.onlineBackward = false
   self.step = 1
   
   -- stores internal states of Modules at different time-steps
   self.sharedClones = {}
   
   self:reset()
end

function AbstractRecurrent:getStepModule(step)
   assert(step, "expecting step at arg 1")
   local recurrentModule = self.sharedClones[step]
   if not recurrentModule then
      recurrentModule = self.recurrentModule:stepClone()
      self.sharedClones[step] = recurrentModule
   end
   return recurrentModule
end

function AbstractRecurrent:maskZero(nInputDim)
   self.recurrentModule = nn.MaskZero(self.recurrentModule, nInputDim)
   return self
end

function AbstractRecurrent:updateGradInput(input, gradOutput)      
   if self.onlineBackward then
      -- updateGradInput will be called in reverse order of time
      self.updateGradInputStep = self.updateGradInputStep or self.step
      
      -- set inputs and gradOutputs for updateGradInputThroughTime
      assert(not self.copyInputs)
      assert(not self.copyGradOutputs)
      local step = self.updateGradInputStep - 1
      self.inputs[step] = nn.rnn.recursiveSet(self.inputs[step], input)
      self.gradOutputs[step] = nn.rnn.recursiveSet(self.gradOutputs[step], gradOutput)
      
      -- BPTT for one time-step (rho = 1)
      self.gradInput = self:updateGradInputThroughTime(self.updateGradInputStep, 1)
      
      self.updateGradInputStep = self.updateGradInputStep - 1
      assert(self.gradInput, "Missing gradInput")
      return self.gradInput
   else
      -- Back-Propagate Through Time (BPTT) happens in updateParameters()
      -- for now we just keep a list of the gradOutputs
      if self.copyGradOutputs then
         self.gradOutputs[self.step-1] = nn.rnn.recursiveCopy(self.gradOutputs[self.step-1] , gradOutput)
      else
         self.gradOutputs[self.step-1] = self.gradOutputs[self.step-1] or nn.rnn.recursiveNew(gradOutput)
         nn.rnn.recursiveSet(self.gradOutputs[self.step-1], gradOutput)
      end
   end
end

function AbstractRecurrent:accGradParameters(input, gradOutput, scale)
   if self.onlineBackward then
      -- accGradParameters will be called in reverse order of time
      assert(self.updateGradInputStep < self.step, "Missing updateGradInput")
      self.accGradParametersStep = self.accGradParametersStep or self.step
      self.scales[self.accGradParametersStep-1] = scale or 1
      
      -- BPTT for one time-step (rho = 1)
      assert(not self.copyInputs)
      local step = self.accGradParametersStep - 1
      self.inputs[step] = nn.rnn.recursiveSet(self.inputs[step], input)
      self:accGradParametersThroughTime(self.accGradParametersStep, 1)
      
      self.accGradParametersStep = self.accGradParametersStep - 1
   else
      -- Back-Propagate Through Time (BPTT) happens in updateParameters()
      -- for now we just keep a list of the scales
      self.scales[self.step-1] = scale or 1
   end
end

function AbstractRecurrent:backwardThroughTime(step, rho)
   return self.gradInput
end

function AbstractRecurrent:updateGradInputThroughTime(step, rho)
end

function AbstractRecurrent:accGradParametersThroughTime(step, rho)
end

function AbstractRecurrent:accUpdateGradParametersThroughTime(lr, step, rho)
end

function AbstractRecurrent:backwardUpdateThroughTime(learningRate)
   local gradInput = self:updateGradInputThroughTime()
   self:accUpdateGradParametersThroughTime(learningRate)
   return gradInput
end

-- this is only useful when calling updateParameters directly on the rnn
-- Note that a call to updateParameters on an rnn container DOES NOT call this method
function AbstractRecurrent:updateParameters(learningRate)
   if self.gradParametersAccumulated then
      for i=1,#self.modules do
         self.modules[i]:updateParameters(learningRate)
      end
   else
      self:backwardUpdateThroughTime(learningRate)
   end
end

-- goes hand in hand with the next method : forget()
-- this methods brings the oldest memory to the current step
function AbstractRecurrent:recycle(offset)
   -- offset can be used to skip initialModule (if any)
   offset = offset or 0
   -- pad rho with one extra time-step of memory (helps for Sequencer:remember()).
   -- also, rho could have been manually increased or decreased
   local rho = math.max(self.rho+1, _.size(self.sharedClones) or 0)
   if self.step > rho + offset then
      assert(self.sharedClones[self.step] == nil)
      self.sharedClones[self.step] = self.sharedClones[self.step-rho]
      self.sharedClones[self.step-rho] = nil
   end
   
   rho = math.max(self.rho+1, _.size(self.outputs) or 0)
   if self.step > rho + offset then
      -- need to keep rho+1 of these
      assert(self.outputs[self.step] == nil)
      self.outputs[self.step] = self.outputs[self.step-rho-1] 
      self.outputs[self.step-rho-1] = nil
   end
   
   rho = math.max(self.rho+1, _.size(self.inputs) or 0)
   if self.step > rho then
      assert(self.inputs[self.step] == nil)
      assert(self.gradOutputs[self.step] == nil)
      assert(self._gradOutputs[self.step] == nil)
      self.inputs[self.step] = self.inputs[self.step-rho]
      self.inputs[self.step-rho] = nil      
      self.gradOutputs[self.step] = self.gradOutputs[self.step-rho] 
      self._gradOutputs[self.step] = self._gradOutputs[self.step-rho]
      self.gradOutputs[self.step-rho] = nil
      self._gradOutputs[self.step-rho] = nil
      self.scales[self.step-rho] = nil
   end
   
   return self
end

-- this method brings all the memory back to the start
function AbstractRecurrent:forget(offset)
   offset = offset or 0
   
    -- bring all states back to the start of the sequence buffers
   if self.train ~= false then
      self.outputs = _.compact(self.outputs)
      self.sharedClones = _.compact(self.sharedClones)
      self.inputs = _.compact(self.inputs)
      
      self.scales = {}
      self.gradOutputs = _.compact(self.gradOutputs)
      self._gradOutputs = _.compact(self._gradOutputs)
   end
   
   -- forget the past inputs; restart from first step
   self.step = 1
   return self
end

function AbstractRecurrent:includingSharedClones(f)
   local modules = self.modules
   local sharedClones = self.sharedClones
   self.sharedClones = nil
   self.modules = {}
   for i,modules in ipairs{modules, sharedClones} do
      for j, module in pairs(modules) do
         table.insert(self.modules, module)
      end
   end
   local r = {f()}
   self.modules = modules
   self.sharedClones = sharedClones
   return unpack(r)
end

function AbstractRecurrent:type(type)
   return self:includingSharedClones(function()
      return parent.type(self, type)
   end)
end

function AbstractRecurrent:training()
   return self:includingSharedClones(function()
      return parent.training(self)
   end)
end

function AbstractRecurrent:evaluate()
   return self:includingSharedClones(function()
      return parent.evaluate(self)
   end)
end

function AbstractRecurrent:reinforce(reward)
   return self:includingSharedClones(function()
      return parent.reinforce(self, reward)
   end)
end

function AbstractRecurrent:sharedClone(shareParams, shareGradParams, clones, pointers, stepClone)
   if stepClone then 
      return self 
   else
      return parent.sharedClone(self, shareParams, shareGradParams, clones, pointers, stepClone)
   end
end

function AbstractRecurrent:backwardOnline(online)
   parent.backwardOnline(self, online)
   self.onlineBackward = (online == nil) and true or online
end

function AbstractRecurrent:maxBPTTstep(rho)
   self.rho = rho
end

-- backwards compatibility
AbstractRecurrent.recursiveResizeAs = rnn.recursiveResizeAs
AbstractRecurrent.recursiveSet = rnn.recursiveSet
AbstractRecurrent.recursiveCopy = rnn.recursiveCopy
AbstractRecurrent.recursiveAdd = rnn.recursiveAdd
AbstractRecurrent.recursiveTensorEq = rnn.recursiveTensorEq
AbstractRecurrent.recursiveNormal = rnn.recursiveNormal
