------------------------------------------------------------------------
--[[ Sequencer ]]--
-- Encapsulates a Module. 
-- Input is a sequence (a table) of tensors.
-- Output is a sequence (a table) of tensors of the same length.
-- Applies the module to each element in the sequence.
-- Handles both recurrent modules and non-recurrent modules.
-- The sequences in a batch must have the same size.
-- But the sequence length of each batch can vary.
------------------------------------------------------------------------
local Sequencer, parent = torch.class("nn.Sequencer", "nn.Container")

function Sequencer:__init(module)
   parent.__init(self)
   self.module = module
   self.isRecurrent = module.backwardThroughTime ~= nil
   self.modules[1] = module
   self.sequenceOutputs = {}
   self.output = {}
   self.step = 1
end

local function recursiveResizeAs(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = recursiveResizeAs(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = t1 or t2.new()
      t1:resizeAs(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end


function Sequencer:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table', "expecting input table")
   self.output = {}
   if self.isRecurrent then
      self.module:forget()
      for step, input in ipairs(inputTable) do
         self.output[step] = self.module:updateOutput(input)
      end
   else
      for step, input in ipairs(inputTable) do
         -- set output states for this step
         local modules = self.module:listModules()
         local sequenceOutputs = self.sequenceOutputs[step]
         if not sequenceOutputs then
            sequenceOutputs = {}
            self.sequenceOutputs[step] = sequenceOutputs
         end
         for i,modula in ipairs(modules) do
            local output_ = recursiveResizeAs(sequenceOutputs[i], modula.output)
            modula.output = output_
         end
         
         -- forward propagate this step
         self.output[step] = self.module:updateOutput(input)
         
         -- save output state of this step
         for i,modula in ipairs(modules) do
            sequenceOutputs[i] = modula.output
         end
      end
   end
   return self.output
end

function Sequencer:updateGradInput(inputTable, gradOutputTable)
   self.gradInput = {}
   if self.isRecurrent then
      assert(torch.type(gradOutputTable) == 'table', "expecting gradOutput table")
      assert(#gradOutputTable == #inputTable, "gradOutput should have as many elements as input")
      for step, input in ipairs(inputTable) do
         self.module.step = step + 1
         self.module:updateGradInput(input, gradOutputTable[step])
      end
      -- back-propagate through time (BPTT)
      self.module:updateGradInputThroughTime()
      assert(self.module.gradInputs, "recurrent module did not fill gradInputs")
      for step=1,#inputTable do
         self.gradInput[step] = self.module.gradInputs[step]
      end
      assert(#self.gradInput == #inputTable, "missing gradInputs")
   else
      for step, input in ipairs(inputTable) do
         -- set the output/gradOutput states for this step
         local modules = self.module:listModules()
         local sequenceOutputs = self.sequenceOutputs[step]
         local sequenceGradInputs = self.sequenceGradInputs[step]
         if not sequenceGradInputs then
            sequenceGradInputs = {}
            self.sequenceGradInputs[step] = sequenceGradInputs
         end
         for i,modula in ipairs(modules) do
            local output, gradInput = modula.output, modula.gradInput
            local output_ = sequenceOutputs[i]
            assert(output_, "updateGradInputThroughTime should be preceded by updateOutput")
            modula.output = output_
            modula.gradInput = recursiveResizeAs(sequenceGradInputs[i], gradInput)
         end
         
         -- backward propagate this step
         self.gradInput[step] = self.module:updateGradInput(input, gradOutputTable[step])
         
         -- save the output/gradOutput states of this step
         for i,modula in ipairs(modules) do
            sequenceGradInputs[i] = modula.gradInput
         end
      end
   end
   return self.gradInput
end

function Sequencer:accGradParameters(inputTable, gradOutputTable, scale)
   if self.isRecurrent then
      assert(torch.type(gradOutputTable) == 'table', "expecting gradOutput table")
      assert(#gradOutputTable == #inputTable, "gradOutput should have as many elements as input")
      for step, input in ipairs(inputTable) do
         self.module.step = step + 1
         self.module:accGradParameters(input, gradOutputTable[step], scale)
      end
      -- back-propagate through time (BPTT)
      self.module:accGradParametersThroughTime()
   else
      for step, input in ipairs(inputTable) do
         -- set the output/gradOutput states for this step
         local modules = self.module:listModules()
         local sequenceOutputs = self.sequenceOutputs[step]
         local sequenceGradInputs = self.sequenceGradInputs[step]
         if not sequenceGradInputs then
            sequenceGradInputs = {}
            self.sequenceGradInputs[step] = sequenceGradInputs
         end
         for i,modula in ipairs(modules) do
            local output, gradInput = modula.output, modula.gradInput
            local output_ = sequenceOutputs[i]
            modula.output = output_
            modula.gradInput = recursiveResizeAs(sequenceGradInputs[i], gradInput)
         end
         
         -- accumulate parameters for this step
         self.module:accGradParameters(input, gradOutputTable[step], scale)
      end
   end
end

function Sequencer:accUpdateGradParameters(input, gradOutput, lr)
   if self.isRecurrent then
      assert(torch.type(gradOutputTable) == 'table', "expecting gradOutput table")
      assert(#gradOutputTable == #inputTable, "gradOutput should have as many elements as input")
      for step, input in ipairs(inputTable) do
         self.module.step = step + 1
         self.module:accGradParameters(input, gradOutputTable[step], 1)
      end
      -- back-propagate through time (BPTT)
      self.module:accUpdateGradParametersThroughTime(lr)
   else
      for step, input in ipairs(inputTable) do
         -- set the output/gradOutput states for this step
         local modules = self.module:listModules()
         local sequenceOutputs = self.sequenceOutputs[step]
         local sequenceGradInputs = self.sequenceGradInputs[step]
         if not sequenceGradInputs then
            sequenceGradInputs = {}
            self.sequenceGradInputs[step] = sequenceGradInputs
         end
         for i,modula in ipairs(modules) do
            local output, gradInput = modula.output, modula.gradInput
            local output_ = sequenceOutputs[i]
            modula.output = output_
            modula.gradInput = recursiveResizeAs(sequenceGradInputs[i], gradInput)
         end
         
         -- accumulate parameters for this step
         self.module:accUpdateGradParameters(input, gradOutputTable[step], lr)
      end
   end
end

function Sequencer:__tostring__()
   local tab = '  '
   local line = '\n'
   local str = torch.type(self) .. ' {' .. line
   str = str .. tab .. '[input(1), input(2), ..., input(T)]'.. line
   str = str .. tab .. '   V           V            V      '.. line
   str = str .. tab .. tostring(self.modules[1]):gsub(line, line .. tab) .. line
   str = str .. tab .. '   V           V            V      '.. line
   str = str .. tab .. '[output(1),output(2),...,output(T)]' .. line
   str = str .. '}'
   return str
end
