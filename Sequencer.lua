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
   else
      self.module.copyInputs = false
      self.module.copyGradOutputs = false
   end
   self.output = {}
   
   -- table of buffers used for evaluation
   self._output = {}
   -- so that these buffers aren't serialized :
   self.dpnn_mediumEmpty = _.clone(self.dpnn_mediumEmpty)
   table.insert(self.dpnn_mediumEmpty, '_output')
   -- default is to forget previous inputs before each forward()
   self._remember = 'neither'
end

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
   if self.isRecurrent then
      -- Note that the Sequencer hijacks the rho attribute of the rnn
      self.module.rho = #inputTable
      if self.train ~= false then -- training
         if not (self._remember == 'train' or self._remember == 'both') then
            self.module:forget()
         end
         self.output = {}
         for step, input in ipairs(inputTable) do
            self.output[step] = self.module:updateOutput(input)
         end
      else -- evaluation
         if not (self._remember == 'eval' or self._remember == 'both') then
            self.module:forget()
         end
         -- during evaluation, recurrent modules reuse memory (i.e. outputs)
         -- so we need to copy each output into our own
         for step, input in ipairs(inputTable) do
            self.output[step] = nn.rnn.recursiveCopy(
               self.output[step] or table.remove(self._output, 1), 
               self.module:updateOutput(input)
            )
         end
         -- remove extra output tensors (save for later)
         for i=#inputTable+1,#self.output do
            table.insert(self._output, self.output[i])
            self.output[i] = nil
         end
      end
   else
      self.output = {}
      for step, input in ipairs(inputTable) do
         -- set output states for this step
         local module = self:getStepModule(step)
         
         -- forward propagate this step
         self.output[step] = module:updateOutput(input)
      end
   end
   return self.output
end

function Sequencer:updateGradInput(inputTable, gradOutputTable)
   self.gradInput = {}
   if self.isRecurrent then
      assert(torch.type(gradOutputTable) == 'table', "expecting gradOutput table")
      assert(#gradOutputTable == #inputTable, "gradOutput should have as many elements as input")
      local i = 1
      for step=self.module.step-#inputTable+1,self.module.step do
         self.module.step = step
         self.module:updateGradInput(inputTable[i], gradOutputTable[i])
         i = i + 1
      end
      -- back-propagate through time (BPTT)
      self.module:updateGradInputThroughTime()
      assert(self.module.gradInputs, "recurrent module did not fill gradInputs")
      assert(#inputTable == #self.module.gradInputs, #inputTable.." ~= "..#self.module.gradInputs)
      for i=1,#inputTable do
         self.gradInput[i] = self.module.gradInputs[i]
      end
      assert(#self.gradInput == #inputTable, "missing gradInputs (rho is too low?)")
   else
      for step, input in ipairs(inputTable) do
         -- set the output/gradOutput states for this step
         local module = self:getStepModule(step)
         
         -- backward propagate this step
         self.gradInput[step] = module:updateGradInput(input, gradOutputTable[step])
      end
   end
   return self.gradInput
end

function Sequencer:accGradParameters(inputTable, gradOutputTable, scale)
   if self.isRecurrent then
      assert(torch.type(gradOutputTable) == 'table', "expecting gradOutput table")
      assert(#gradOutputTable == #inputTable, "gradOutput should have as many elements as input")
      local i = 1
      for step=self.module.step-#inputTable+1,self.module.step do
         self.module.step = step
         self.module:accGradParameters(inputTable[i], gradOutputTable[i], scale)
         i = i + 1
      end
      -- back-propagate through time (BPTT)
      self.module:accGradParametersThroughTime()
   else
      for step, input in ipairs(inputTable) do
         -- set the output/gradOutput states for this step
         local module = self:getStepModule(step)
         
         -- accumulate parameters for this step
         module:accGradParameters(input, gradOutputTable[step], scale)
      end
   end
end

function Sequencer:accUpdateGradParameters(inputTable, gradOutputTable, lr)
   if self.isRecurrent then
      assert(torch.type(gradOutputTable) == 'table', "expecting gradOutput table")
      assert(#gradOutputTable == #inputTable, "gradOutput should have as many elements as input")
      local i = 1
      for step=self.module.step-#inputTable+1,self.module.step do
         self.module.step = step
         self.module:accGradUpdateParameters(inputTable[i], gradOutputTable[i], lr)
         i = i + 1
      end
      -- back-propagate through time (BPTT)
      self.module:accUpdateGradParametersThroughTime(lr)
   else
      for step, input in ipairs(inputTable) do
         -- set the output/gradOutput states for this step
         local module = self:getStepModule(step)
         
         -- accumulate parameters for this step
         module:accUpdateGradParameters(input, gradOutputTable[step], lr)
      end
   end
end

-- Toggle to feed long sequences using multiple forwards.
-- 'eval' only affects evaluation (recommended for RNNs)
-- 'train' only affects training
-- 'neither' affects neither training nor evaluation
-- 'both' affects both training and evaluation (recommended for LSTMs)
-- Essentially, forget() isn't called on rnn module when remember is on
function Sequencer:remember(remember)
   self._remember = (remember == nil) and 'both' or remember
   assert(_.contains({'both','eval','train','neither'}, self._remember), 
      "Sequencer : unrecognized value for remember : "..self._remember)
   return self
end

function Sequencer:type(type)
   local modules = self.modules
   self.modules = {}
   for i,modules in ipairs{modules, self.sharedClones} do
      for j, module in pairs(modules) do
         table.insert(self.modules, module)
      end
   end
   parent.type(self, type)
   self.modules = modules
   return self
end

function Sequencer:training()
   if self.isRecurrent and self.train == false then
      -- empty output table (tensor mem was managed by seq)
      for i,output in ipairs(self.output) do
         table.insert(self._output, output)
         self.output[i] = nil
      end
      -- forget at the start of each training
      self:forget()
   end
   parent.training(self)
end

function Sequencer:evaluate()
   if self.isRecurrent and self.train ~= false then
      -- empty output table (tensor mem was managed by rnn)
      self.output = {}
      -- forget at the start of each evaluation
      self:forget()
   end
   parent.evaluate(self)
   assert(self.train == false)
end

Sequencer.__tostring__ = nn.Decorator.__tostring__
