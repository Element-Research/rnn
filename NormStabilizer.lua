------------------------------------------------------------------------
--[[ Norm Stabilization]]
-- Regularizing RNNs by Stabilizing Activations
-- Ref. A:  http://arxiv.org/abs/1511.08400
------------------------------------------------------------------------

local NS, parent = torch.class("nn.NormStabilizer", "nn.AbstractRecurrent")

function NS:__init(beta, rho)
   parent.__init(self, rho or 9999)
   self.recurrentModule = nn.CopyGrad()
   self.beta = beta
end

function NS:_accGradParameters(input, gradOutput, scale)
   -- No parameters to update
end

function NS:updateOutput(input)
   local output
   if self.train ~= false then
      self:recycle()
      local recurrentModule = self:getStepModule(self.step)
      output = recurrentModule:updateOutput(input)
   else
      output = self.recurrentModule:updateOutput(input)
   end

   self.outputs[self.step] = output

   self.output = output
   self.step = self.step + 1
   self.gradPrevOutput = nil
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil

   return self.output
end

function NS:_updateGradInput(input, gradOutput)    
   -- First grab h[t] and h[t+1] :
   -- backward propagate through this step
   local gradInput = self.recurrentModule:updateGradInput(input, gradOutput)
   local curStep = self.updateGradInputStep-1
   local hiddenModule = self:getStepModule(curStep)
   local hiddenState = hiddenModule.output
   hiddenModule.gradInput = gradInput

   if curStep < self.step then
      local batchSize = hiddenState:size(1)
      if curStep > 1 then
         local prevHiddenModule = self:getStepModule(curStep - 1)
         local prevHiddenState = prevHiddenModule.output
         -- Add norm stabilizer cost function directly to respective CopyGrad.gradInput tensors
         for i=1,batchSize do
            local dRegdNorm =  self.beta * 2 * (hiddenState[i]:norm()-prevHiddenState[i]:norm()) / batchSize
            local dNormdHid = torch.div(hiddenState[i], hiddenState[i]:norm())
            hiddenModule.gradInput[i]:add(torch.mul(dNormdHid, dRegdNorm))
         end
      end
      if curStep < self.step-1 then
         local nextHiddenModule = self:getStepModule(curStep + 1)
         local nextHiddenState = nextHiddenModule.output
         for i=1,batchSize do
            local dRegdNorm = self.beta * -2 * (nextHiddenState[i]:norm() - hiddenState[i]:norm()) / batchSize
            local dNormdHid = torch.div(hiddenState[i], hiddenState[i]:norm()) 
            hiddenModule.gradInput[i]:add(torch.mul(dNormdHid, dRegdNorm))
         end
      end
   end
   return hiddenModule.gradInput
end

function NS:__tostring__()
   return "nn.NormStabilizer"
end
