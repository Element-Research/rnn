------------------------------------------------------------------------
--[[ BiSequencerLM ]]--
-- Encapsulates forward, backward and merge modules. 
-- Input is a sequence (a table) of tensors.
-- Output is a sequence (a table) of tensors of the same length.
-- Applies a `fwd` rnn instance to the first `N-1` elements in the 
-- sequence in forward order.
-- Applies the `bwd` rnn in reverse order to the last `N-1` elements 
-- (from second-to-last element to first element).
-- Note : you shouldn't stack these for language modeling. 
-- Instead, stack each fwd/bwd seqs and encapsulate these.
------------------------------------------------------------------------
local BiSequencerLM, parent = torch.class('nn.BiSequencerLM', 'nn.AbstractSequencer')

function BiSequencerLM:__init(forward, backward, merge)
   
   if not torch.isTypeOf(forward, 'nn.Module') then
      error"BiSequencerLM: expecting nn.Module instance at arg 1"
   end
   self.forwardModule = forward
   
   self.backwardModule = backward
   if not self.backwardModule then
      self.backwardModule = forward:clone()
      self.backwardModule:reset()
   end
   if not torch.isTypeOf(self.backwardModule, 'nn.Module') then
      error"BiSequencerLM: expecting nn.Module instance at arg 2"
   end
   
   if torch.type(merge) == 'number' then
      self.mergeModule = nn.JoinTable(1, merge)
   elseif merge == nil then
      self.mergeModule = nn.JoinTable(1, 1)
   elseif torch.isTypeOf(merge, 'nn.Module') then
      self.mergeModule = merge
   else
      error"BiSequencerLM: expecting nn.Module or number instance at arg 3"
   end
   
   if torch.isTypeOf(self.forwardModule, 'nn.AbstractRecurrent') then
      self.fwdSeq = nn.Sequencer(self.forwardModule)
   else -- assumes a nn.Sequencer or stack thereof
      self.fwdSeq = self.forwardModule
   end
   
   if torch.isTypeOf(self.backwardModule, 'nn.AbstractRecurrent') then
      self.bwdSeq = nn.Sequencer(self.backwardModule)
   else
      self.bwdSeq = self.backwardModule
   end
   self.mergeSeq = nn.Sequencer(self.mergeModule)
   
   self._fwd = self.fwdSeq
   
   self._bwd = nn.Sequential()
   self._bwd:add(nn.ReverseTable())
   self._bwd:add(self.bwdSeq)
   self._bwd:add(nn.ReverseTable())
   
   self._merge = nn.Sequential()
   self._merge:add(nn.ZipTable())
   self._merge:add(self.mergeSeq)
   
   
   parent.__init(self)
   
   self.modules = {self._fwd, self._bwd, self._merge}
   
   self.output = {}
   self.gradInput = {}
end

function BiSequencerLM:updateOutput(input)
   assert(torch.type(input) == 'table', 'Expecting table at arg 1')
   local nStep = #input
   assert(nStep > 1, "Expecting at least 2 elements in table")
   
   -- forward through fwd and bwd rnn in fwd and reverse order
   self._fwdOutput = self._fwd:updateOutput(_.first(input, nStep - 1))
   self._bwdOutput = self._bwd:updateOutput(_.last(input, nStep - 1))
   
   -- empty outputs
   for k,v in ipairs(self.output) do self.output[k] = nil end
   
   -- padding for first and last elements of fwd and bwd outputs, respectively
   self._firstStep = nn.rnn.recursiveResizeAs(self._firstStep, self._fwdOutput[1])
   nn.rnn.recursiveFill(self._firstStep, 0)
   self._lastStep = nn.rnn.recursiveResizeAs(self._lastStep, self._bwdOutput[1])
   nn.rnn.recursiveFill(self._lastStep, 0)
   
   -- { { zeros, fwd1, fwd2, ..., fwdN}, {bwd1, bwd2, ..., bwdN, zeros} }
   self._mergeInput = {_.clone(self._fwdOutput), _.clone(self._bwdOutput)}
   table.insert(self._mergeInput[1], 1, self._firstStep)
   table.insert(self._mergeInput[2], self._lastStep)
   assert(#self._mergeInput[1] == #self._mergeInput[2])
   
   self.output = self._merge:updateOutput(self._mergeInput)
   
   return self.output
end

function BiSequencerLM:updateGradInput(input, gradOutput)
   local nStep = #input
   
   self._mergeGradInput = self._merge:updateGradInput(self._mergeInput, gradOutput)
   self._fwdGradInput = self._fwd:updateGradInput(_.first(input, nStep - 1), _.last(self._mergeGradInput[1], nStep - 1))
   self._bwdGradInput = self._bwd:updateGradInput(_.last(input, nStep - 1), _.first(self._mergeGradInput[2], nStep - 1))
   
   -- add fwd rnn gradInputs to bwd rnn gradInputs
   for i=1,nStep do
      if i == 1 then
         self.gradInput[1] = self._fwdGradInput[1]
      elseif i == nStep then
         self.gradInput[nStep] = self._bwdGradInput[nStep-1]
      else
         self.gradInput[i] = nn.rnn.recursiveCopy(self.gradInput[i], self._fwdGradInput[i])
         nn.rnn.recursiveAdd(self.gradInput[i], self._bwdGradInput[i-1])
      end
   end
   
   return self.gradInput
end

function BiSequencerLM:accGradParameters(input, gradOutput, scale)
   local nStep = #input
   
   self._merge:accGradParameters(self._mergeInput, gradOutput, scale)
   self._fwd:accGradParameters(_.first(input, nStep - 1), _.last(self._mergeGradInput[1], nStep - 1), scale)
   self._bwd:accGradParameters(_.last(input, nStep - 1), _.first(self._mergeGradInput[2], nStep - 1), scale)
end

function BiSequencerLM:accUpdateGradParameters(input, gradOutput, lr)
   local nStep = #input
   
   self._merge:accUpdateGradParameters(self._mergeInput, gradOutput, lr)
   self._fwd:accUpdateGradParameters(_.first(input, nStep - 1), _.last(self._mergeGradInput[1], nStep - 1), lr)
   self._bwd:accUpdateGradParameters(_.last(input, nStep - 1), _.first(self._mergeGradInput[2], nStep - 1), lr)
end

function BiSequencerLM:__tostring__()
   local tab = '  '
   local line = '\n'
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {'
   str = str .. line .. tab .. '(  fwd  ): ' .. tostring(self._fwd):gsub(line, line .. tab .. ext)
   str = str .. line .. tab .. '(  bwd  ): ' .. tostring(self._bwd):gsub(line, line .. tab .. ext)
   str = str .. line .. tab .. '( merge ): ' .. tostring(self._merge):gsub(line, line .. tab .. ext)
   str = str .. line .. '}'
   return str
end
