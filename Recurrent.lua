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
local Recurrent, parent = torch.class('nn.Recurrent', 'nn.AbstractRecurrent')

function Recurrent:__init(start, input, feedback, transfer, rho, merge)
   parent.__init(self, rho or 5)
   
   local ts = torch.type(start)
   if ts == 'torch.LongTensor' or ts == 'number' then
      start = nn.Add(start)
   end
   
   self.startModule = start
   self.inputModule = input
   self.feedbackModule = feedback
   self.transferModule = transfer or nn.Sigmoid()
   self.mergeModule = merge or nn.CAddTable()
   
   self.modules = {self.startModule, self.inputModule, self.feedbackModule, self.transferModule, self.mergeModule}
   
   self:buildInitialModule()
   self:buildRecurrentModule()
   
   self.initialOutputs = {}
   self.initialGradInputs = {}
end

-- build module used for the first step (steps == 1)
function Recurrent:buildInitialModule()
   self.initialModule = nn.Sequential()
   self.initialModule:add(self.inputModule)
   self.initialModule:add(self.startModule)
   self.initialModule:add(self.transferModule)
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
      -- set/save the output states
      local modules = self.initialModule:listModules()
      for i,modula in ipairs(modules) do
         local output_ = self.recursiveResizeAs(self.initialOutputs[i], modula.output)
         modula.output = output_
      end
      output = self.initialModule:updateOutput(input)
      for i,modula in ipairs(modules) do
         self.initialOutputs[i]  = modula.output
      end
   else
      if self.train ~= false then
         -- set/save the output states
         local modules = self.recurrentModule:listModules()
         self:recycle()
         local recurrentOutputs = self.recurrentOutputs[self.step]
         if not recurrentOutputs then
            recurrentOutputs = {}
            self.recurrentOutputs[self.step] = recurrentOutputs
         end
         for i,modula in ipairs(modules) do
            local output_ = self.recursiveResizeAs(recurrentOutputs[i], modula.output)
            modula.output = output_
         end
          -- self.output is the previous output of this module
         output = self.recurrentModule:updateOutput{input, self.output}
         for i,modula in ipairs(modules) do
            recurrentOutputs[i]  = modula.output
         end
      else
         -- self.output is the previous output of this module
         output = self.recurrentModule:updateOutput{input, self.output}
      end
   end
   
   if self.train ~= false then
      local input_ = self.inputs[self.step]
      self.inputs[self.step] = self.copyInputs 
         and self.recursiveCopy(input_, input) 
         or self.recursiveSet(input_, input)     
   end
   
   self.outputs[self.step] = output
   self.output = output
   self.step = self.step + 1
   self.gradParametersAccumulated = false
   return self.output
end

-- not to be confused with the hit movie Back to the Future
function Recurrent:backwardThroughTime()
   assert(self.step > 1, "expecting at least one updateOutput")
   local rho = math.min(self.rho, self.step-1)
   local stop = self.step - rho
   if self.fastBackward then
      self.gradInputs = {}
      local gradInput, gradPrevOutput
      for step=self.step-1,math.max(stop, 2),-1 do
         -- set the output/gradOutput states of current Module
         local modules = self.recurrentModule:listModules()
         local recurrentOutputs = self.recurrentOutputs[step]
         local recurrentGradInputs = self.recurrentGradInputs[step]
         if not recurrentGradInputs then
            recurrentGradInputs = {}
            self.recurrentGradInputs[step] = recurrentGradInputs
         end
         for i,modula in ipairs(modules) do
            local output, gradInput = modula.output, modula.gradInput
            assert(gradInput, "missing gradInput")
            local output_ = recurrentOutputs[i]
            assert(output_, "backwardThroughTime should be preceded by updateOutput")
            modula.output = output_
            modula.gradInput = self.recursiveResizeAs(recurrentGradInputs[i], gradInput)
         end
         
         -- backward propagate through this step
         local input = self.inputs[step]
         local output = self.outputs[step-1]
         local gradOutput = self.gradOutputs[step] 
         if gradPrevOutput then
            self.recursiveAdd(gradOutput, gradPrevOutput)   
         end
         local scale = self.scales[step]
         
         gradInput, gradPrevOutput = unpack(self.recurrentModule:backward({input, output}, gradOutput, scale/rho))
         table.insert(self.gradInputs, 1, gradInput)
         
         for i,modula in ipairs(modules) do
            recurrentGradInputs[i] = modula.gradInput
         end
      end
      
      if stop <= 1 then
         -- set the output/gradOutput states of initialModule
         local modules = self.initialModule:listModules()
         for i,modula in ipairs(modules) do
            modula.output = self.initialOutputs[i]
            modula.gradInput = self.recursiveResizeAs(self.initialGradInputs[i], modula.gradInput)
         end
         
         -- backward propagate through first step
         local input = self.inputs[1]
         local gradOutput = self.gradOutputs[1]
         if gradPrevOutput then
            self.recursiveAdd(gradOutput, gradPrevOutput)
         end
         local scale = self.scales[1]
         gradInput = self.initialModule:backward(input, gradOutput, scale/rho)
         table.insert(self.gradInputs, 1, gradInput)
         
         for i,modula in ipairs(modules) do
            self.initialGradInputs[i] = modula.gradInput
         end
         
         -- startModule's gradParams shouldn't be step-averaged
         -- as it is used only once. So un-step-average it
         local params, gradParams = self.startModule:parameters()
         if gradParams then
            for i,gradParam in ipairs(gradParams) do
               gradParam:mul(rho)
            end
         end
         
         self.gradParametersAccumulated = true
         return gradInput
      end
   else
      local gradInput = self:updateGradInputThroughTime()
      self:accGradParametersThroughTime()
      return gradInput
   end
end

function Recurrent:updateGradInputThroughTime()
   assert(self.step > 1, "expecting at least one updateOutput")
   self.gradInputs = {}
   local gradInput, gradPrevOutput
   local rho = math.min(self.rho, self.step-1)
   local stop = self.step - rho
   for step=self.step-1,math.max(stop,2),-1 do
      -- set the output/gradOutput states of current Module
      local modules = self.recurrentModule:listModules()
      local recurrentOutputs = self.recurrentOutputs[step]
      local recurrentGradInputs = self.recurrentGradInputs[step]
      if not recurrentGradInputs then
         recurrentGradInputs = {}
         self.recurrentGradInputs[step] = recurrentGradInputs
      end
      for i,modula in ipairs(modules) do
         local output, gradInput = modula.output, modula.gradInput
         local output_ = recurrentOutputs[i]
         assert(output_, "updateGradInputThroughTime should be preceded by updateOutput")
         modula.output = output_
         modula.gradInput = self.recursiveResizeAs(recurrentGradInputs[i], gradInput)
      end
      
      -- backward propagate through this step
      local input = self.inputs[step]
      local output = self.outputs[step-1]
      local gradOutput = self.gradOutputs[step]
      if gradPrevOutput then
         self.recursiveAdd(gradOutput, gradPrevOutput) 
      end
      
      gradInput, gradPrevOutput = unpack(self.recurrentModule:updateGradInput({input, output}, gradOutput))
      table.insert(self.gradInputs, 1, gradInput)
      
      for i,modula in ipairs(modules) do
         recurrentGradInputs[i] = modula.gradInput
      end
   end
   
   if stop <= 1 then
      -- set the output/gradOutput states of initialModule
      local modules = self.initialModule:listModules()
      for i,modula in ipairs(modules) do
         modula.output = self.initialOutputs[i]
         modula.gradInput = self.recursiveResizeAs(self.initialGradInputs[i], modula.gradInput)
      end
      
      -- backward propagate through first step
      local input = self.inputs[1]
      local gradOutput = self.gradOutputs[1]
      if gradPrevOutput then
         self.recursiveAdd(gradOutput, gradPrevOutput) 
      end
      gradInput = self.initialModule:updateGradInput(input, gradOutput)
      table.insert(self.gradInputs, 1, gradInput)
      
      for i,modula in ipairs(modules) do
         self.initialGradInputs[i] = modula.gradInput
      end
   end
   
   return gradInput
end

function Recurrent:accGradParametersThroughTime()
   local rho = math.min(self.rho, self.step-1)
   local stop = self.step - rho
   for step=self.step-1,math.max(stop,2),-1 do
      -- set the output/gradOutput states of current Module
      local modules = self.recurrentModule:listModules()
      local recurrentOutputs = self.recurrentOutputs[step]
      local recurrentGradInputs = self.recurrentGradInputs[step]
      
      for i,modula in ipairs(modules) do
         local output, gradInput = modula.output, modula.gradInput
         local output_ = recurrentOutputs[i]
         local gradInput_ = recurrentGradInputs[i]
         assert(output_, "accGradParametersThroughTime should be preceded by updateOutput")
         assert(gradInput_, "accGradParametersThroughTime should be preceded by updateGradInputThroughTime")
         modula.output = output_
         modula.gradInput = gradInput_
      end
      
      -- backward propagate through this step
      local input = self.inputs[step]
      local output = self.outputs[step-1]
      local gradOutput = self.gradOutputs[step]

      local scale = self.scales[step]
      self.recurrentModule:accGradParameters({input, output}, gradOutput, scale/rho)
   end
   
   if stop <= 1 then
      -- set the output/gradOutput states of initialModule
      local modules = self.initialModule:listModules()
      for i,modula in ipairs(modules) do
         local output, gradInput = modula.output, modula.gradInput
         local output_ = self.initialOutputs[i]
         local gradInput_ = self.initialGradInputs[i] 
         modula.output = output_
         modula.gradInput = gradInput_
      end
         
      -- backward propagate through first step
      local input = self.inputs[1]
      local gradOutput = self.gradOutputs[1]
      local scale = self.scales[1]
      self.initialModule:accGradParameters(input, gradOutput, scale/rho)
      
      -- startModule's gradParams shouldn't be step-averaged
      -- as it is used only once. So un-step-average it
      local params, gradParams = self.startModule:parameters()
      if gradParams then
         for i,gradParam in ipairs(gradParams) do
            gradParam:mul(rho)
         end
      end
   end
   
   self.gradParametersAccumulated = true
   return gradInput
end

function Recurrent:accUpdateGradParametersThroughTime(lr)
   local rho = math.min(self.rho, self.step-1)
   local stop = self.step - rho
   for step=self.step-1,math.max(stop,2),-1 do
      -- set the output/gradOutput states of current Module
      local modules = self.recurrentModule:listModules()
      local recurrentOutputs = self.recurrentOutputs[step]
      local recurrentGradInputs = self.recurrentGradInputs[step]
      
      for i,modula in ipairs(modules) do
         local output, gradInput = modula.output, modula.gradInput
         local output_ = recurrentOutputs[i]
         local gradInput_ = recurrentGradInputs[i]
         assert(output_, "accGradParametersThroughTime should be preceded by updateOutput")
         assert(gradInput_, "accGradParametersThroughTime should be preceded by updateGradInputThroughTime")
         modula.output = output_
         modula.gradInput = gradInput_
      end
      
      -- backward propagate through this step
      local input = self.inputs[step]
      local output = self.outputs[step-1]
      local gradOutput = self.gradOutputs[step]

      local scale = self.scales[step]
      self.recurrentModule:accUpdateGradParameters({input, output}, gradOutput, lr*scale/rho)
   end
   
   if stop <= 1 then
      -- set the output/gradOutput states of initialModule
      local modules = self.initialModule:listModules()
      for i,modula in ipairs(modules) do
         local output, gradInput = modula.output, modula.gradInput
         local output_ = self.initialOutputs[i]
         local gradInput_ = self.initialGradInputs[i] 
         modula.output = output_
         modula.gradInput = gradInput_
      end
      
      -- backward propagate through first step
      local input = self.inputs[1]
      local gradOutput = self.gradOutputs[1]
      local scale = self.scales[1]
      self.inputModule:accUpdateGradParameters(input, self.startModule.gradInput, lr*scale/rho)
      -- startModule's gradParams shouldn't be step-averaged as it is used only once.
      self.startModule:accUpdateGradParameters(self.inputModule.output, self.transferModule.gradInput, lr*scale)
   end
   
   return gradInput
end

function Recurrent:forget()
   parent.forget(self, 1)
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
