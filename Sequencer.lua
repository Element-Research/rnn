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
assert(not nn.Sequencer, "update nnx package : luarocks install nnx")
local Sequencer, parent = torch.class('nn.Sequencer', 'nn.AbstractSequencer')

function Sequencer:__init(module)
   parent.__init(self)
   if not torch.isTypeOf(module, 'nn.Module') then
      error"Sequencer: expecting nn.Module instance at arg 1"
   end
   
   -- we can decorate the module with a Recursor to make it AbstractRecurrent
   self.module = (not torch.isTypeOf(module, 'nn.AbstractRecurrent')) and nn.Recursor(module) or module
   -- backprop through time (BPTT) will be done online (in reverse order of forward)
   self.modules = {self.module}
   
   self.output = {}
   
   -- table of buffers used for evaluation
   self._output = {}
   -- so that these buffers aren't serialized :
   local _ = require 'moses'
   self.dpnn_mediumEmpty = _.clone(self.dpnn_mediumEmpty)
   table.insert(self.dpnn_mediumEmpty, '_output')
   -- default is to forget previous inputs before each forward()
   self._remember = 'neither'
end

function Sequencer:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table', "expecting input table")

   -- Note that the Sequencer hijacks the rho attribute of the rnn
   self.module:maxBPTTstep(#inputTable)
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
      -- so we need to copy each output into our own table
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
   
   return self.output
end

function Sequencer:updateGradInput(inputTable, gradOutputTable)
   assert(torch.type(gradOutputTable) == 'table', "expecting gradOutput table")
   assert(#gradOutputTable == #inputTable, "gradOutput should have as many elements as input")
   
   -- back-propagate through time (BPTT)
   self.gradInput = {}
   for step=#gradOutputTable,1,-1 do
      self.gradInput[step] = self.module:updateGradInput(inputTable[step], gradOutputTable[step])
   end
   
   assert(#inputTable == #self.gradInput, #inputTable.." ~= "..#self.gradInput)

   return self.gradInput
end

function Sequencer:accGradParameters(inputTable, gradOutputTable, scale)
   assert(torch.type(gradOutputTable) == 'table', "expecting gradOutput table")
   assert(#gradOutputTable == #inputTable, "gradOutput should have as many elements as input")
   
   -- back-propagate through time (BPTT)
   for step=#gradOutputTable,1,-1 do
      self.module:accGradParameters(inputTable[step], gradOutputTable[step], scale)
   end   
end

function Sequencer:accUpdateGradParameters(inputTable, gradOutputTable, lr)
   assert(torch.type(gradOutputTable) == 'table', "expecting gradOutput table")
   assert(#gradOutputTable == #inputTable, "gradOutput should have as many elements as input")
   
   -- back-propagate through time (BPTT)
   for step=#gradOutputTable,1,-1 do
      self.module:accUpdateGradParameters(inputTable[step], gradOutputTable[step], lr)
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
   local _ = require 'moses'
   assert(_.contains({'both','eval','train','neither'}, self._remember), 
      "Sequencer : unrecognized value for remember : "..self._remember)
   return self
end

function Sequencer:training()
   if self.train == false then
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
   if self.train ~= false then
      -- empty output table (tensor mem was managed by rnn)
      self.output = {}
      -- forget at the start of each evaluation
      self:forget()
   end
   parent.evaluate(self)
   assert(self.train == false)
end

Sequencer.__tostring__ = nn.Decorator.__tostring__
