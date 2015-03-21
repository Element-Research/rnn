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
local Sequencer, parent
if nn.Sequencer then -- prevent name conflicts with nnx
   Sequencer, parent = nn.Sequencer, nn.Container
else
   Sequencer, parent = torch.class('nn.Sequencer', 'nn.Container')
end

function Sequencer:__init(module)
   parent.__init(self)
   if not torch.isTypeOf(module, 'nn.Module') then
      error"Sequencer: expecting nn.Module instance at arg 1"
   end
   self.module = module
   self.isRecurrent = module.backwardThroughTime ~= nil
   self.modules[1] = module
   self.sharedClones = {}
   if not self.isRecurrent then
      self.sharedClones[1] = self.module
      -- test that it doesn't contain a recurrent module :
      local err = false
      for i,modula in ipairs(module:listModules()) do
         if modula.backwardThroughTime then
            err = modula
            break
         end
      end
      
      if err then
         error("Sequencer: non-recurrent Module should not contain a "..
         "nested recurrent Modules. Recurrent module is "..torch.type(err)..
         ". Use a Sequencer instance for each recurrent module. "..
         "And encapsulate the rest of the non-recurrent modules into "..
         "one or many Sequencers. Yes you can encapsulate many non-recurrent"..
         " modules in a single Sequencer (as long as they don't include recurrent modules.") 
      end
   end
   self.output = {}
   self.step = 1
end

local recursiveResizeAs = nn.AbstractRecurrent.recursiveResizeAs
local recursiveCopy = nn.AbstractRecurrent.recursiveCopy

function Sequencer:getStepModule(step)
   assert(step, "expecting step at arg 1")
   local module = self.sharedClones[step]
   if not module then
      module = self.module:sharedClone()
      self.sharedClones[step] = module
   end
   return module
end


function Sequencer:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table', "expecting input table")
   self.output = {}
   if self.isRecurrent then
      self.module:forget()
      for step, input in ipairs(inputTable) do
         self.output[step] = recursiveCopy(
            self.output[step], 
            self.module:updateOutput(input)
         )
      end
   else
      for step, input in ipairs(inputTable) do
         -- set output states for this step
         local module = self:getStepModule(step)
         
         -- forward propagate this step
         self.output[step] = recursiveCopy(
            self.output[step], 
            module:updateOutput(input)
         )
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
      assert(#self.gradInput == #inputTable, "missing gradInputs (rho is too low?)")
   else
      for step, input in ipairs(inputTable) do
         -- set the output/gradOutput states for this step
         local module = self:getStepModule(step)
         
         -- backward propagate this step
         self.gradInput[step] = recursiveCopy(
            self.gradInput[step], 
            self.module:updateGradInput(input, gradOutputTable[step])
         )
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
         local module = self:getStepModule(step)
         
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
         local module = self:getStepModule(step)
         
         -- accumulate parameters for this step
         self.module:accUpdateGradParameters(input, gradOutputTable[step], lr)
      end
   end
end

function Sequencer:sharedType(type, castmap)
   local modules = self.modules
   self.modules = {}
   for i,modules in ipairs{modules, self.sharedClones} do
      for j, module in pairs(modules) do
         table.insert(self.modules, module)
      end
   end
   parent.sharedType(self, type, castmap)
   self.modules = modules
   return self
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
