------------------------------------------------------------------------
--[[ Recursor ]]--
-- Decorates module to be used within an AbstractSequencer.
-- It does this by making the decorated module conform to the 
-- AbstractRecurrent interface (which is inherited by LSTM/Recurrent) 
------------------------------------------------------------------------
local Recursor, parent = torch.class('nn.Recursor', 'nn.AbstractRecurrent')

function Recursor:__init(module, rho)
   parent.__init(self, rho or 9999999)

   self.recurrentModule = module
   self.recurrentModule:backwardOnline()
   self.onlineBackward = true
   
   self.module = module
   self.modules = {module}
end

function Recursor:updateOutput(input)
   local output
   if self.train ~= false then
      -- set/save the output states
      self:recycle()
      local recurrentModule = self:getStepModule(self.step)
      output = recurrentModule:updateOutput(input)
   else
      output = self.recurrentModule:updateOutput(input)
   end
   
   if self.train ~= false then
      local input_ = self.inputs[self.step]
      self.inputs[self.step] = self.copyInputs 
         and nn.rnn.recursiveCopy(input_, input) 
         or nn.rnn.recursiveSet(input_, input)     
   end
   
   self.outputs[self.step] = output
   self.output = output
   self.step = self.step + 1
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil
   self.gradParametersAccumulated = false
   return self.output
end

function Recursor:backwardThroughTime(timeStep, timeRho)
   timeStep = timeStep or self.step
   local rho = math.min(timeRho or self.rho, timeStep-1)
   local stop = timeStep - rho
   local gradInput
   if self.fastBackward then
      self.gradInputs = {}
      for step=timeStep-1,math.max(stop, 1),-1 do
         -- backward propagate through this step
         local recurrentModule = self:getStepModule(step)
         gradInput = recurrentModule:backward(self.inputs[step], self.gradOutputs[step] , self.scales[step])
         table.insert(self.gradInputs, 1, gradInput)
      end
      
      self.gradParametersAccumulated = true
   else
      gradInput = self:updateGradInputThroughTime(timeStep, timeRho)
      self:accGradParametersThroughTime(timeStep, timeRho)
   end
   return gradInput
end

function Recursor:updateGradInputThroughTime(timeStep, rho)
   assert(self.step > 1, "expecting at least one updateOutput")
   self.gradInputs = {}
   timeStep = timeStep or self.step
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho
   local gradInput
   for step=timeStep-1,math.max(stop,1),-1 do
      -- backward propagate through this step
      local recurrentModule = self:getStepModule(step)
      gradInput = recurrentModule:updateGradInput(self.inputs[step], self.gradOutputs[step])
      table.insert(self.gradInputs, 1, gradInput)
   end
   
   return gradInput
end

function Recursor:accGradParametersThroughTime(timeStep, rho)
   timeStep = timeStep or self.step
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho
   for step=timeStep-1,math.max(stop,1),-1 do
      -- backward propagate through this step
      local recurrentModule = self:getStepModule(step)
      recurrentModule:accGradParameters(self.inputs[step], self.gradOutputs[step], self.scales[step])
   end
   
   self.gradParametersAccumulated = true
   return gradInput
end

function Recursor:accUpdateGradParametersThroughTime(lr, timeStep, rho)
   timeStep = timeStep or self.step
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho
   for step=timeStep-1,math.max(stop,1),-1 do
      -- backward propagate through this step
      local recurrentModule = self:getStepModule(step)
      recurrentModule:accUpdateGradParameters(self.inputs[step], self.gradOutputs[step], lr*self.scales[step])
   end
   
   return gradInput
end

function Recursor:includingSharedClones(f)
   local modules = self.modules
   self.modules = {}
   local sharedClones = self.sharedClones
   self.sharedClones = nil
   for i,modules in ipairs{modules, sharedClones} do
      for j, module in pairs(modules) do
         table.insert(self.modules, module)
      end
   end
   local r = f()
   self.modules = modules
   self.sharedClones = sharedClones
   return r
end

function Recursor:backwardOnline(online)
   assert(online ~= false, "Recursor only supports online backwards")
   parent.backwardOnline(self)
end

function Recursor:forget(offset)
   parent.forget(self, offset)
   nn.Module.forget(self)
   return self
end

function Recursor:maxBPTTstep(rho)
   self.rho = rho
   nn.Module.maxBPTTstep(self, rho)
end

Recursor.__tostring__ = nn.Decorator.__tostring__
