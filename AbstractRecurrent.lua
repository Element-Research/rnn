local AbstractRecurrent, parent = torch.class('nn.AbstractRecurrent', 'nn.Container')

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
   self.recurrentOutputs = {}
   self.recurrentGradInputs = {}
   
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
function AbstractRecurrent:recycle()
   -- +1 is to skip initialModule
   if self.step > self.rho + 1 then
      assert(self.recurrentOutputs[self.step] == nil)
      assert(self.recurrentOutputs[self.step-self.rho] ~= nil)
      self.recurrentOutputs[self.step] = self.recurrentOutputs[self.step-self.rho]
      self.recurrentGradInputs[self.step] = self.recurrentGradInputs[self.step-self.rho]
      self.recurrentOutputs[self.step-self.rho] = nil
      self.recurrentGradInputs[self.step-self.rho] = nil
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
            self.recurrentOutputs[i] = self.recurrentOutputs[step]
            self.recurrentGradInputs[i] = self.recurrentGradInputs[step]
            self.recurrentOutputs[step] = nil
            self.recurrentGradInputs[step] = nil
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

-- tests whether or not the mlp can be used internally for recursion.
-- forward A, backward A, forward B, forward A should be consistent with
-- forward B, backward B, backward A where A and B each 
-- have their own gradInputs/outputs.
function AbstractRecurrent.isRecursable(mlp, input)
   local output = recursiveCopy(nil, mlp:forward(input)) --forward A
   local gradOutput = recursiveNormal(recursiveCopy(nil, output))
   mlp:zeroGradParameters()
   local gradInput = recursiveCopy(nil, mlp:backward(input, gradOutput)) --backward A
   local params, gradParams = mlp:parameters()
   gradParams = recursiveCopy(nil, gradParams)
   
   -- output/gradInput are the only internal module states that we track
   local recurrentOutputs = {}
   local recurrentGradInputs = {}
   
   local modules = mlp:listModules()
   
   -- save the output/gradInput states of A
   for i,modula in ipairs(modules) do
      recurrentOutputs[i]  = modula.output
      recurrentGradInputs[i] = modula.gradInput
   end
   -- set the output/gradInput states for B
   local recurrentOutputs2 = {}
   local recurrentGradInputs2 = {}
   for i,modula in ipairs(modules) do
      modula.output = recursiveResizeAs(recurrentOutputs2[i], modula.output)
      modula.gradInput = recursiveResizeAs(recurrentGradInputs2[i], modula.gradInput)
   end
   
   local input2 = recursiveNormal(recursiveCopy(nil, input))
   local gradOutput2 = recursiveNormal(recursiveCopy(nil, gradOutput))
   local output2 = mlp:forward(input2) --forward B
   mlp:zeroGradParameters()
   local gradInput2 = mlp:backward(input2, gradOutput2) --backward B
   
   -- save the output/gradInput state of B
   for i,modula in ipairs(modules) do
      recurrentOutputs2[i]  = modula.output
      recurrentGradInputs2[i] = modula.gradInput
   end
   
   -- set the output/gradInput states for A
   for i,modula in ipairs(modules) do
      modula.output = recursiveResizeAs(recurrentOutputs[i], modula.output)
      modula.gradInput = recursiveResizeAs(recurrentGradInputs[i], modula.gradInput)
   end
   
   mlp:zeroGradParameters()
   local gradInput3 = mlp:backward(input, gradOutput) --forward A
   local gradInputTest = recursiveTensorEq(gradInput, gradInput3)
   local params3, gradParams3 = mlp:parameters()
   local nEq = 0
   for i,gradParam in ipairs(gradParams) do
      nEq = nEq + (recursiveTensorEq(gradParam, gradParams3[i]) and 1 or 0)
   end
   local gradParamsTest = (nEq == #gradParams3)
   mlp:zeroGradParameters()
   return gradParamsTest and gradInputTest, gradParamsTest, gradInputTest
end
