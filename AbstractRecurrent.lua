local _ = require 'moses'

assert(not nn.AbstractRecurrent, "update nnx package : luarocks install nnx")
local AbstractRecurrent, parent = torch.class('nn.AbstractRecurrent', 'nn.Container')

AbstractRecurrent.dpnn_stepclone = true

function AbstractRecurrent:__init(rho)
   parent.__init(self)
   
   self.rho = rho --the maximum number of time steps to BPTT
   
   self.outputs = {}
   self._gradOutputs = {}

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
   -- updateGradInput should be called in reverse order of time
   self.updateGradInputStep = self.updateGradInputStep or self.step
   
   -- BPTT for one time-step
   self.gradInput = self:_updateGradInput(input, gradOutput, self.updateGradInputStep)
   
   self.updateGradInputStep = self.updateGradInputStep - 1
   assert(self.gradInput, "Missing gradInput")
   return self.gradInput
end

function AbstractRecurrent:accGradParameters(input, gradOutput, scale)
   -- accGradParameters should be called in reverse order of time
   assert(self.updateGradInputStep < self.step, "Missing updateGradInput")
   self.accGradParametersStep = self.accGradParametersStep or self.step
   
   -- BPTT for one time-step 
   local step = self.accGradParametersStep - 1
   self:_accGradParameters(input, gradOutput, scale)
   
   self.accGradParametersStep = self.accGradParametersStep - 1
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
   
   rho = math.max(self.rho+1, _.size(self._gradOutputs) or 0)
   if self.step > rho then
      assert(self._gradOutputs[self.step] == nil)
      self._gradOutputs[self.step] = self._gradOutputs[self.step-rho]
      self._gradOutputs[self.step-rho] = nil
   end
   
   return self
end

-- this method brings all the memory back to the start
function AbstractRecurrent:forget(offset)
   offset = offset or 0
   
   -- the recurrentModule may contain an AbstractRecurrent instance (issue 107)
   parent.forget(self) 
   
    -- bring all states back to the start of the sequence buffers
   if self.train ~= false then
      self.outputs = _.compact(self.outputs)
      self.sharedClones = _.compact(self.sharedClones)
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
      for j, module in pairs(modules or {}) do
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

-- used by Recursor() after calling stepClone.
-- this solves a very annoying bug...
function AbstractRecurrent:setOutputStep(step)
   self.output = self.outputs[step] --or self:getStepModule(step).output
   assert(self.output, "no output for step "..step)
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



function AbstractRecurrent:backwardThroughTime(step, rho)
   error"DEPRECATED Jan 8, 2016"
end

function AbstractRecurrent:updateGradInputThroughTime(step, rho)
   error"DEPRECATED Jan 8, 2016"
end

function AbstractRecurrent:accGradParametersThroughTime(step, rho)
   error"DEPRECATED Jan 8, 2016"
end

function AbstractRecurrent:accUpdateGradParametersThroughTime(lr, step, rho)
   error"DEPRECATED Jan 8, 2016"
end

function AbstractRecurrent:backwardUpdateThroughTime(learningRate)
   error"DEPRECATED Jan 8, 2016"
end
