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
   
   self.module = module
   self.modules = {module}
   self.sharedClones[1] = self.recurrentModule
end

function Recursor:updateOutput(input)
   local output
   if self.train ~= false then -- if self.train or self.train == nil then
      -- set/save the output states
      self:recycle()
      local recurrentModule = self:getStepModule(self.step)
      output = recurrentModule:updateOutput(input)
   else
      output = self.recurrentModule:updateOutput(input)
   end
   
   self.outputs[self.step] = output
   self.output = output
   self.step = self.step + 1
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil
   return self.output
end

function Recursor:_updateGradInput(input, gradOutput)
   assert(self.step > 1, "expecting at least one updateOutput")
   local step = self.updateGradInputStep - 1
   assert(step >= 1)
   
   local recurrentModule = self:getStepModule(step)
   recurrentModule:setOutputStep(step)
   local gradInput = recurrentModule:updateGradInput(input, gradOutput)
   
   return gradInput
end

function Recursor:_accGradParameters(input, gradOutput, scale)
   local step = self.accGradParametersStep - 1
   assert(step >= 1)
   
   local recurrentModule = self:getStepModule(step)
   recurrentModule:setOutputStep(step)
   recurrentModule:accGradParameters(input, gradOutput, scale)
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
   local r = {f()}
   self.modules = modules
   self.sharedClones = sharedClones
   return unpack(r)
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
