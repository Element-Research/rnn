------------------------------------------------------------------------
--[[ Recurrent ]]--
-- Ref. A.: http://goo.gl/vtVGkO (Mikolov et al.)
-- B. http://goo.gl/hu1Lqm
-- Processes the sequence one timestep (forward/backward) at a time. 
-- A call to backward only keeps a log of the gradOutputs and scales.
-- Back-Propagation Through Time (BPTT) is done when updateParameters
-- is called. The Module keeps a list of all previous representations 
-- (Module.outputs), including intermediate ones for BPTT.
-- To use this module with batches, we suggest using different 
-- sequences of the same size within a batch and calling 
-- updateParameters() at the end of the Sequence. 
-- Note that this won't work with modules that use more than the
-- output attribute to keep track of their internal state between 
-- forward and backward.
------------------------------------------------------------------------
assert(not nn.Recurrent, "update nnx package : luarocks install nnx")
local Recurrent, parent = torch.class('nn.Recurrent', 'nn.AbstractRecurrent')

function Recurrent:__init(start, input, feedback, transfer, rho, merge)
   parent.__init(self, rho or 5)
   
   local ts = torch.type(start)
   if ts == 'torch.LongStorage' or ts == 'number' then
      start = nn.Add(start)
   elseif ts == 'table' then
      start = nn.Add(torch.LongStorage(start))
   elseif not torch.isTypeOf(start, 'nn.Module') then
      error"Recurrent : expecting arg 1 of type nn.Module, torch.LongStorage, number or table"
   end
   
   self.startModule = start
   self.inputModule = input
   self.feedbackModule = feedback
   self.transferModule = transfer or nn.Sigmoid()
   self.mergeModule = merge or nn.CAddTable()
   
   self.modules = {self.startModule, self.inputModule, self.feedbackModule, self.transferModule, self.mergeModule}
   
   self:buildInitialModule()
   self:buildRecurrentModule()
   self.sharedClones[2] = self.recurrentModule 
end

-- build module used for the first step (steps == 1)
function Recurrent:buildInitialModule()
   self.initialModule = nn.Sequential()
   self.initialModule:add(self.inputModule:sharedClone())
   self.initialModule:add(self.startModule)
   self.initialModule:add(self.transferModule:sharedClone())
end

-- build module used for the other steps (steps > 1)
function Recurrent:buildRecurrentModule()
   local parallelModule = nn.ParallelTable()
   parallelModule:add(self.inputModule)
   parallelModule:add(self.feedbackModule)
   self.recurrentModule = nn.Sequential()
   self.recurrentModule:add(parallelModule)
   self.recurrentModule:add(self.mergeModule)
   self.recurrentModule:add(self.transferModule)
end

function Recurrent:updateOutput(input)
   -- output(t) = transfer(feedback(output_(t-1)) + input(input_(t)))
   local output
   if self.step == 1 then
      output = self.initialModule:updateOutput(input)
   else
      if self.train ~= false then
         -- set/save the output states
         self:recycle()
         local recurrentModule = self:getStepModule(self.step)
          -- self.output is the previous output of this module
         output = recurrentModule:updateOutput{input, self.output}
      else
         -- self.output is the previous output of this module
         output = self.recurrentModule:updateOutput{input, self.output}
      end
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
   self.gradPrevOutput = nil
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil
   self.gradParametersAccumulated = false
   return self.output
end

-- not to be confused with the hit movie Back to the Future
function Recurrent:backwardThroughTime(timeStep, timeRho)
   timeStep = timeStep or self.step
   local rho = math.min(timeRho or self.rho, timeStep-1)
   local stop = timeStep - rho
   local gradInput
   if self.fastBackward then
      self.gradInputs = {}
      for step=timeStep-1,math.max(stop, 2),-1 do
         local recurrentModule = self:getStepModule(step)
         
         -- backward propagate through this step
         local input = self.inputs[step]
         local output = self.outputs[step-1]
         local gradOutput = self.gradOutputs[step] 
         if self.gradPrevOutput then
            self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
            nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
            gradOutput = self._gradOutputs[step]
         end
         local scale = self.scales[step]
         
         gradInput, self.gradPrevOutput = unpack(recurrentModule:backward({input, output}, gradOutput, scale))
         
         table.insert(self.gradInputs, 1, gradInput)
      end
      
      if stop <= 1 then
         -- backward propagate through first step
         local input = self.inputs[1]
         local gradOutput = self.gradOutputs[1]
         if self.gradPrevOutput then
            self._gradOutputs[1] = nn.rnn.recursiveCopy(self._gradOutputs[1], self.gradPrevOutput)
            nn.rnn.recursiveAdd(self._gradOutputs[1], gradOutput)
            gradOutput = self._gradOutputs[1]
         end
         local scale = self.scales[1]
         gradInput = self.initialModule:backward(input, gradOutput, scale)
         table.insert(self.gradInputs, 1, gradInput)
      end
      self.gradParametersAccumulated = true
   else
      gradInput = self:updateGradInputThroughTime(timeStep, timeRho)
      self:accGradParametersThroughTime(timeStep, timeRho)
   end
   return gradInput
end

function Recurrent:updateGradInputThroughTime(timeStep, rho)
   assert(self.step > 1, "expecting at least one updateOutput")
   timeStep = timeStep or self.step
   self.gradInputs = {}
   local gradInput
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho
   for step=timeStep-1,math.max(stop,2),-1 do
      local recurrentModule = self:getStepModule(step)
      
      -- backward propagate through this step
      local input = self.inputs[step]
      local output = self.outputs[step-1]
      local gradOutput = self.gradOutputs[step]
      if self.gradPrevOutput then
         self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
         nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
         gradOutput = self._gradOutputs[step]
      end

      gradInput, self.gradPrevOutput = unpack(recurrentModule:updateGradInput({input, output}, gradOutput))
      table.insert(self.gradInputs, 1, gradInput)
   end
   
   if stop <= 1 then      
      -- backward propagate through first step
      local input = self.inputs[1]
      local gradOutput = self.gradOutputs[1]
      if self.gradPrevOutput then
         self._gradOutputs[1] = nn.rnn.recursiveCopy(self._gradOutputs[1], self.gradPrevOutput)
         nn.rnn.recursiveAdd(self._gradOutputs[1], gradOutput)
         gradOutput = self._gradOutputs[1]
      end
      gradInput = self.initialModule:updateGradInput(input, gradOutput)
      table.insert(self.gradInputs, 1, gradInput)
   end
   
   return gradInput
end

function Recurrent:accGradParametersThroughTime(timeStep, rho)
   timeStep = timeStep or self.step
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho
   for step=timeStep-1,math.max(stop,2),-1 do
      local recurrentModule = self:getStepModule(step)
      
      -- backward propagate through this step
      local input = self.inputs[step]
      local output = self.outputs[step-1]
      local gradOutput = (step == self.step-1) and self.gradOutputs[step] or self._gradOutputs[step]

      local scale = self.scales[step]
      recurrentModule:accGradParameters({input, output}, gradOutput, scale)
   end
   
   if stop <= 1 then
      -- backward propagate through first step
      local input = self.inputs[1]
      local gradOutput = (1 == self.step-1) and self.gradOutputs[1] or self._gradOutputs[1]
      local scale = self.scales[1]
      self.initialModule:accGradParameters(input, gradOutput, scale)
   end
   
   self.gradParametersAccumulated = true
   return gradInput
end

function Recurrent:accUpdateGradParametersThroughInitialModule(lr, rho)
   if self.initialModule:size() ~= 3 then
      error("only works with Recurrent:buildInitialModule(). "..
      "Reimplement this method to work with your subclass."..
      "Or use accGradParametersThroughTime instead of accUpdateGrad...")
   end
   
   -- backward propagate through first step
   local input = self.inputs[1]
   local gradOutput = (1 == self.step-1) and self.gradOutputs[1] or self._gradOutputs[1]
   local scale = self.scales[1]
   local inputModule = self.initialModule:get(1)
   local startModule = self.initialModule:get(2)
   local transferModule = self.initialModule:get(3)
   inputModule:accUpdateGradParameters(input, self.startModule.gradInput, lr*scale)
   startModule:accUpdateGradParameters(inputModule.output, transferModule.gradInput, lr*scale)
   transferModule:accUpdateGradParameters(startModule.output, gradOutput, lr*scale)
end

function Recurrent:accUpdateGradParametersThroughTime(lr, timeStep, rho)
   timeStep = timeStep or self.step
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho
   for step=timeStep-1,math.max(stop,2),-1 do
      local recurrentModule = self:getStepModule(step)
      
      -- backward propagate through this step
      local input = self.inputs[step]
      local output = self.outputs[step-1]
      local gradOutput = (step == self.step-1) and self.gradOutputs[step] or self._gradOutputs[step]

      local scale = self.scales[step]
      recurrentModule:accUpdateGradParameters({input, output}, gradOutput, lr*scale)
   end
   
   if stop <= 1 then      
      self:accUpdateGradParametersThroughInitialModule(lr, rho)
   end
   
   return gradInput
end

function Recurrent:recycle()
   return parent.recycle(self, 1)
end

function Recurrent:forget()
   return parent.forget(self, 1)
end

function Recurrent:includingSharedClones(f)
   local modules = self.modules
   self.modules = {}
   local sharedClones = self.sharedClones
   self.sharedClones = nil
   local initModule = self.initialModule
   self.initialModule = nil
   for i,modules in ipairs{modules, sharedClones, {initModule}} do
      for j, module in pairs(modules) do
         table.insert(self.modules, module)
      end
   end
   local r = f()
   self.modules = modules
   self.sharedClones = sharedClones
   self.initialModule = initModule 
   return r
end

function Recurrent:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = torch.type(self)
   str = str .. ' {' .. line .. tab .. '[{input(t), output(t-1)}'
   for i=1,3 do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output(t)]'
   
   local tab = '  '
   local line = '\n  '
   local next = '  |`-> '
   local ext = '  |    '
   local last = '   ... -> '
   str = str .. line ..  '(1): ' .. ' {' .. line .. tab .. 'input(t)'
   str = str .. line .. tab .. next .. '(t==0): ' .. tostring(self.startModule):gsub('\n', '\n' .. tab .. ext)
   str = str .. line .. tab .. next .. '(t~=0): ' .. tostring(self.inputModule):gsub('\n', '\n' .. tab .. ext)
   str = str .. line .. tab .. 'output(t-1)'
   str = str .. line .. tab .. next .. tostring(self.feedbackModule):gsub('\n', line .. tab .. ext)
   str = str .. line .. "}"
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   str = str .. line .. tab .. '(' .. 2 .. '): ' .. tostring(self.mergeModule):gsub(line, line .. tab)
   str = str .. line .. tab .. '(' .. 3 .. '): ' .. tostring(self.transferModule):gsub(line, line .. tab)
   str = str .. line .. '}'
   return str
end
