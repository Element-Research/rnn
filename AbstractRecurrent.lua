local _ = require 'moses'

assert(not nn.AbstractRecurrent, "update nnx package : luarocks install nnx")
local AbstractRecurrent, parent = torch.class('nn.AbstractRecurrent', 'nn.Container')

AbstractRecurrent.dpnn_stepclone = true

function AbstractRecurrent:__init(rho)
   parent.__init(self)
   
   self.rho = rho or 99999 --the maximum number of time steps to BPTT
   
   self.outputs = {}
   self.gradInputs = {}
   self._gradOutputs = {}

   self.step = 1
   
   -- stores internal states of Modules at different time-steps
   self.sharedClones = {}
   
   self:reset()
end

function AbstractRecurrent:getStepModule(step)
   local _ = require 'moses'
   assert(step, "expecting step at arg 1")
   local recurrentModule = self.sharedClones[step]
   if not recurrentModule then
      recurrentModule = self.recurrentModule:stepClone()
      self.sharedClones[step] = recurrentModule
      self.nSharedClone = _.size(self.sharedClones)
   end
   return recurrentModule
end

function AbstractRecurrent:maskZero(nInputDim)
   self.recurrentModule = nn.MaskZero(self.recurrentModule, nInputDim, true)
   self.sharedClones = {self.recurrentModule}
   self.modules[1] = self.recurrentModule
   return self
end

function AbstractRecurrent:trimZero(nInputDim)
   if torch.typename(self)=='nn.GRU' and self.p ~= 0 then
      assert(self.mono, "TrimZero for BGRU needs `mono` option.")
   end
   self.recurrentModule = nn.TrimZero(self.recurrentModule, nInputDim, true)
   self.sharedClones = {self.recurrentModule}
   self.modules[1] = self.recurrentModule
   return self
end

function AbstractRecurrent:updateGradInput(input, gradOutput)  
   -- updateGradInput should be called in reverse order of time
   self.updateGradInputStep = self.updateGradInputStep or self.step
   
   -- BPTT for one time-step
   self.gradInput = self:_updateGradInput(input, gradOutput)
   
   self.updateGradInputStep = self.updateGradInputStep - 1
   self.gradInputs[self.updateGradInputStep] = self.gradInput
   return self.gradInput
end

function AbstractRecurrent:accGradParameters(input, gradOutput, scale)
   -- accGradParameters should be called in reverse order of time
   assert(self.updateGradInputStep < self.step, "Missing updateGradInput")
   self.accGradParametersStep = self.accGradParametersStep or self.step
   
   -- BPTT for one time-step 
   self:_accGradParameters(input, gradOutput, scale)
   
   self.accGradParametersStep = self.accGradParametersStep - 1
end

-- goes hand in hand with the next method : forget()
-- this methods brings the oldest memory to the current step
function AbstractRecurrent:recycle(offset)
   -- offset can be used to skip initialModule (if any)
   offset = offset or 0
   
   local _ = require 'moses'
   self.nSharedClone = self.nSharedClone or _.size(self.sharedClones) 

   local rho = math.max(self.rho + 1, self.nSharedClone)
   if self.sharedClones[self.step] == nil then
      self.sharedClones[self.step] = self.sharedClones[self.step-rho]
      self.sharedClones[self.step-rho] = nil
      self._gradOutputs[self.step] = self._gradOutputs[self.step-rho]
      self._gradOutputs[self.step-rho] = nil
   end
   
   self.outputs[self.step-rho-1] = nil
   self.gradInputs[self.step-rho-1] = nil
   
   return self
end

function nn.AbstractRecurrent:clearState()
   self:forget()
   -- keep the first two sharedClones
   nn.utils.clear(self, '_input', '_gradOutput', '_gradOutputs', 'gradPrevOutput', 'cell', 'cells', 'gradCells', 'outputs', 'gradInputs')
   for i, clone in ipairs(self.sharedClones) do
      clone:clearState()
   end
   self.recurrentModule:clearState()
   return parent.clearState(self)
end

-- this method brings all the memory back to the start
function AbstractRecurrent:forget()
   -- the recurrentModule may contain an AbstractRecurrent instance (issue 107)
   parent.forget(self) 
   local _ = require 'moses'
   
    -- bring all states back to the start of the sequence buffers
   if self.train ~= false then
      self.outputs = {}
      self.gradInputs = {}
      self.sharedClones = _.compact(self.sharedClones)
      self._gradOutputs = _.compact(self._gradOutputs)
   end
   
   -- forget the past inputs; restart from first step
   self.step = 1
   
   
  if not self.rmInSharedClones then
      -- Asserts that issue 129 is solved. In forget as it is often called.
      -- Asserts that self.recurrentModule is part of the sharedClones.
      -- Since its used for evaluation, it should be used for training. 
      local nClone, maxIdx = 0, 1
      for k,v in pairs(self.sharedClones) do -- to prevent odd bugs
         if torch.pointer(v) == torch.pointer(self.recurrentModule) then
            self.rmInSharedClones = true
            maxIdx = math.max(k, maxIdx)
         end
         nClone = nClone + 1
      end
      if nClone > 1 then
         if not self.rmInSharedClones then
            print"WARNING : recurrentModule should be added to sharedClones in constructor."
            print"Adding it for you."
            assert(torch.type(self.sharedClones[maxIdx]) == torch.type(self.recurrentModule))
            self.recurrentModule = self.sharedClones[maxIdx]
            self.rmInSharedClones = true
         end
      end
   end
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

function AbstractRecurrent:type(type, tensorcache)
   return self:includingSharedClones(function()
      return parent.type(self, type, tensorcache)
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
   if torch.type(reward) == 'table' then
      -- multiple rewards, one per time-step
      local rewards = reward
      for step, reward in ipairs(rewards) do
         local sm = self:getStepModule(step)
         sm:reinforce(reward)
      end
   else
      -- one reward broadcast to all time-steps
      return self:includingSharedClones(function()
         return parent.reinforce(self, reward)
      end)
   end
end

-- used by Recursor() after calling stepClone.
-- this solves a very annoying bug...
function AbstractRecurrent:setOutputStep(step)
   self.output = self.outputs[step] --or self:getStepModule(step).output
   assert(self.output, "no output for step "..step)
   self.gradInput = self.gradInputs[step]
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

function AbstractRecurrent:__tostring__()
   if self.inputSize and self.outputSize then
       return self.__typename .. string.format("(%d -> %d)", self.inputSize, self.outputSize)
   else
       return parent.__tostring__(self)
   end
end
