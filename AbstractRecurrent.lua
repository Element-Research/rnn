local AbstractRecurrent, parent
if nn.AbstractRecurrent then -- prevent name conflicts with nnx
   AbstractRecurrent, parent = nn.AbstractRecurrent, nn.Container
else
   AbstractRecurrent, parent = torch.class('nn.AbstractRecurrent', 'nn.Container')
end

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
   self.step = 1
   
   -- stores internal states of Modules at different time-steps
   self.sharedClones = {}
   
   self:reset()
end

function AbstractRecurrent:getStepModule(step)
   assert(step, "expecting step at arg 1")
   local recurrentModule = self.sharedClones[step]
   if not recurrentModule then
      recurrentModule = self.recurrentModule:sharedClone()
      self.sharedClones[step] = recurrentModule
   end
   return recurrentModule
end

function AbstractRecurrent:updateGradInput(input, gradOutput)
   -- Back-Propagate Through Time (BPTT) happens in updateParameters()
   -- for now we just keep a list of the gradOutputs
   if self.copyGradOutputs then
      self.gradOutputs[self.step-1] = nn.rnn.recursiveCopy(self.gradOutputs[self.step-1] , gradOutput)
   else
      self.gradOutputs[self.step-1] = self.gradOutputs[self.step-1] or gradOutput.new()
      self.gradOutputs[self.step-1]:set(gradOutput)
   end
end

function AbstractRecurrent:accGradParameters(input, gradOutput, scale)
   -- Back-Propagate Through Time (BPTT) happens in updateParameters()
   -- for now we just keep a list of the scales
   self.scales[self.step-1] = scale or 1
end

function AbstractRecurrent:backwardThroughTime()
   return self.gradInput
end

function AbstractRecurrent:updateGradInputThroughTime()
end

function AbstractRecurrent:accGradParametersThroughTime()
end

function AbstractRecurrent:accUpdateGradParametersThroughTime(lr)
end

function AbstractRecurrent:backwardUpdateThroughTime(learningRate)
   local gradInput = self:updateGradInputThroughTime()
   self:accUpdateGradParametersThroughTime(learningRate)
   return gradInput
end

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
function AbstractRecurrent:recycle(offset)
   offset = offset or 0
   -- offset can be used to skip initialModule (if any)
   if self.step > self.rho + offset then
      assert(self.sharedClones[self.step] == nil)
      assert(self.sharedClones[self.step-self.rho] ~= nil)
      self.sharedClones[self.step] = self.sharedClones[self.step-self.rho]
      self.sharedClones[self.step-self.rho] = nil
      -- need to keep rho+1 of these
      self.outputs[self.step] = self.outputs[self.step-self.rho-1] 
      self.outputs[self.step-self.rho-1] = nil
   end
   if self.step > self.rho then
      assert(self.inputs[self.step] == nil)
      assert(self.inputs[self.step-self.rho] ~= nil)
      self.inputs[self.step] = self.inputs[self.step-self.rho] 
      self.gradOutputs[self.step] = self.gradOutputs[self.step-self.rho] 
      self.inputs[self.step-self.rho] = nil
      self.gradOutputs[self.step-self.rho] = nil
      self.scales[self.step-self.rho] = nil
   end
   
   return self
end

function AbstractRecurrent:forget(offset)
   offset = offset or 0
   if self.train ~= false then
      -- bring all states back to the start of the sequence buffers
      local lastStep = self.step - 1
      
      if lastStep > self.rho + offset then
         local i = 1 + offset
         for step = lastStep-self.rho+offset,lastStep do
            assert(self.sharedClones[i] == nil)
            self.sharedClones[i] = self.sharedClones[step]
            self.sharedClones[step] = nil
            -- we keep rho+1 of these : outputs[k]=outputs[k+rho+1]
            assert(self.outputs[i-1] == nil)
            self.outputs[i-1] = self.outputs[step]
            self.outputs[step] = nil
            i = i + 1
         end
         
      end
      
      if lastStep > self.rho then
         local i = 1
         for step = lastStep-self.rho+1,lastStep do
            assert(self.inputs[i] == nil)
            assert(self.gradOutputs[i] == nil)
            self.inputs[i] = self.inputs[step]
            self.gradOutputs[i] = self.gradOutputs[step]
            self.inputs[step] = nil
            self.gradOutputs[step] = nil
            self.scales[step] = nil
            i = i + 1
         end

      end
   end
   
   -- forget the past inputs; restart from first step
   self.step = 1
   
   return self
end

function AbstractRecurrent:includingSharedClones(f)
   local modules = self.modules
   self.modules = {}
   for i,modules in ipairs{modules, self.sharedClones} do
      for j, module in pairs(modules) do
         table.insert(self.modules, module)
      end
   end
   local r = f()
   self.modules = modules
   return r
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

-- backwards compatibility
AbstractRecurrent.recursiveResizeAs = rnn.recursiveResizeAs
AbstractRecurrent.recursiveSet = rnn.recursiveSet
AbstractRecurrent.recursiveCopy = rnn.recursiveCopy
AbstractRecurrent.recursiveAdd = rnn.recursiveAdd
AbstractRecurrent.recursiveTensorEq = rnn.recursiveTensorEq
AbstractRecurrent.recursiveNormal = rnn.recursiveNormal
