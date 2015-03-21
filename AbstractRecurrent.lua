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
   
   self.inputs = {}
   self.outputs = {}
   self.gradOutputs = {}
   self.scales = {}
   
   self.gradParametersAccumulated = false
   self.step = 1
   
   -- stores internal states of Modules at different time-steps
   self.sharedClones = {}
   
   self:reset()
end

local function recursiveResizeAs(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = recursiveResizeAs(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:resizeAs(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end
AbstractRecurrent.recursiveResizeAs = recursiveResizeAs

local function recursiveSet(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = recursiveSet(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = t1 or t2.new()
      t1:set(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end
AbstractRecurrent.recursiveSet = recursiveSet

local function recursiveCopy(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = recursiveCopy(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:resizeAs(t2):copy(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end
AbstractRecurrent.recursiveCopy = recursiveCopy

local function recursiveAdd(t1, t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = recursiveAdd(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) and torch.isTensor(t2) then
      t1:add(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end
AbstractRecurrent.recursiveAdd = recursiveAdd

local function recursiveTensorEq(t1, t2)
   if torch.type(t2) == 'table' then
      local isEqual = true
      if torch.type(t1) ~= 'table' then
         return false
      end
      for key,_ in pairs(t2) do
          isEqual = isEqual and recursiveTensorEq(t1[key], t2[key])
      end
      return isEqual
   elseif torch.isTensor(t2) and torch.isTensor(t2) then
      local diff = t1-t2
      local err = diff:abs():max()
      return err < 0.00001
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
end
AbstractRecurrent.recursiveTensorEq = recursiveTensorEq

local function recursiveNormal(t2)
   if torch.type(t2) == 'table' then
      for key,_ in pairs(t2) do
         t2[key] = recursiveNormal(t2[key])
      end
   elseif torch.isTensor(t2) then
      t2:normal()
   else
      error("expecting tensor or table thereof. Got "
           ..torch.type(t2).." instead")
   end
   return t2
end
AbstractRecurrent.recursiveNormal = recursiveNormal

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
   self.gradOutputs[self.step-1] = self.recursiveCopy(self.gradOutputs[self.step-1] , gradOutput)
end

function AbstractRecurrent:accGradParameters(input, gradOutput, scale)
   -- Back-Propagate Through Time (BPTT) happens in updateParameters()
   -- for now we just keep a list of the scales
   self.scales[self.step-1] = scale
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
end

function AbstractRecurrent:forget(offset)
   offset = offset or 1
   if self.train ~= false then
      -- bring all states back to the start of the sequence buffers
      local lastStep = self.step - 1
      
      if lastStep > self.rho + offset then
         local i = 1 + offset
         for step = lastStep-self.rho+offset,lastStep do
            self.sharedClone[i] = self.sharedClone[step]
            self.sharedClone[step] = nil
            -- we keep rho+1 of these : outputs[k]=outputs[k+rho+1]
            self.outputs[i-1] = self.outputs[step]
            self.outputs[step] = nil
            i = i + 1
         end
         
      end
      
      if lastStep > self.rho then
         local i = 1
         for step = lastStep-self.rho+1,lastStep do
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
end
