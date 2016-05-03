------------------------------------------------------------------------
--[[ Norm Stabilization]]
-- Regularizing RNNs by Stabilizing Activations
-- Ref. A:  http://arxiv.org/abs/1511.08400
-- For training, this module only works in batch mode.
------------------------------------------------------------------------

local NS, parent = torch.class("nn.NormStabilizer", "nn.AbstractRecurrent")

function NS:__init(beta)
   parent.__init(self, 99999)

   self.beta = beta or 1
   self.recurrentModule = nn.CopyGrad()
   
   -- make it work with nn.Container
   self.modules[1] = self.recurrentModule
   self.sharedClones[1] = self.recurrentModule 
end

function NS:_accGradParameters(input, gradOutput, scale)
   -- No parameters to update
end

function NS:updateOutput(input)
   assert(input:dim() == 2)
   local output
   if self.train ~= false then
      self:recycle()
      local rm = self:getStepModule(self.step)
      output = rm:updateOutput(input)
      -- in training mode, we also calculate norm of hidden state
      rm.norm = rm.norm or output.new()
      rm.norm:norm(output, 2, 2)
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

-- returns norm-stabilizer loss as defined in ref. A
function NS:updateLoss()
   self.loss = 0
   self._normsum = self._normsum or self.output.new()
   
   for step=2,self.step-1 do
      local rm1 = self:getStepModule(step-1)
      local rm2 = self:getStepModule(step)
      self._normsum:add(rm1.norm, rm2.norm)
      self._normsum:pow(2)
      local steploss = self._normsum:mean() -- sizeAverage
      self.loss = self.loss +  steploss
   end
   
   -- the loss is divided by the number of time-steps (but not the gradients)
   self.loss = self.beta * self.loss / (self.step-1)
   return self.loss
end

function NS:_updateGradInput(input, gradOutput)    
   -- First grab h[t] :
   -- backward propagate through this step
   local curStep = self.updateGradInputStep-1
   local hiddenModule = self:getStepModule(curStep)
   local gradInput = hiddenModule:updateGradInput(input, gradOutput)
   assert(curStep < self.step)
   
   -- buffers
   self._normsum = self._normsum or self.output.new()
   self._gradInput = self._gradInput or self.output.new()
   
   local batchSize = hiddenModule.output:size(1)
   
   -- Add gradient of norm stabilizer cost function directly to respective CopyGrad.gradInput tensors
   
   if curStep > 1 then
      -- then grab h[t-1]
      local prevHiddenModule = self:getStepModule(curStep - 1)
      
      self._normsum:resizeAs(hiddenModule.norm):copy(hiddenModule.norm)
      self._normsum:add(-1, prevHiddenModule.norm)
      self._normsum:mul(self.beta*2)
      self._normsum:cdiv(hiddenModule.norm)
      
      self._gradInput:mul(hiddenModule.output, 1/batchSize)
      self._gradInput:cmul(self._normsum:expandAs(self._gradInput))
      hiddenModule.gradInput:add(self._gradInput)
   end
   
   if curStep < self.step-1 then
      local nextHiddenModule = self:getStepModule(curStep + 1)
      
      self._normsum:resizeAs(hiddenModule.norm):copy(hiddenModule.norm)
      self._normsum:add(-1, nextHiddenModule.norm)
      self._normsum:mul(self.beta*2)
      self._normsum:cdiv(hiddenModule.norm)
      
      self._gradInput:mul(hiddenModule.output, 1/batchSize)
      self._gradInput:cmul(self._normsum:expandAs(self._gradInput))
      hiddenModule.gradInput:add(self._gradInput)
   end
   
   return hiddenModule.gradInput
end

function NS:__tostring__()
   return "nn.NormStabilizer"
end
