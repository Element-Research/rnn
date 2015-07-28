------------------------------------------------------------------------
--[[ SequencerCriterion ]]--
-- Applies a criterion to each of the inputs and targets in the 
-- corresponding input and target Tables.
-- Useful for nn.Repeater and nn.Sequencer.
-- WARNING : assumes that the decorated criterion is stateless, i.e. 
-- the backward doesn't need to be preceded by a commensurate forward.
------------------------------------------------------------------------
local SequencerCriterion, parent = torch.class('nn.SequencerCriterion', 'nn.Criterion')

function SequencerCriterion:__init(criterion)
   parent.__init(self)
   self.criterion = criterion
   if torch.isTypeOf(criterion, 'nn.ModuleCriterion') then
      error("SequencerCriterion shouldn't decorate a ModuleCriterion. "..
         "Instead, try the other way around : "..
         "ModuleCriterion decorates a SequencerCriterion. "..
         "Its modules can also be similarly decorated with a Sequencer.")
   end
   self.gradInput = {}
end

function SequencerCriterion:updateOutput(inputTable, targetTable)
   self.output = 0
   for i,input in ipairs(inputTable) do
      self.output = self.output + self.criterion:forward(input, targetTable[i])
   end
   return self.output
end

function SequencerCriterion:updateGradInput(inputTable, targetTable)
   for i,input in ipairs(inputTable) do
      self.gradInput[i] = nn.rnn.recursiveCopy(self.gradInput[i], self.criterion:backward(input, targetTable[i]))
   end
   
   if #inputTable >= 3 and not self.isStateless then
      -- make sure the criterion is stateless
      local gradInput
      for i = 1,3 do
         self.criterion:forward(inputTable[i], targetTable[i])
         gradInput = self.criterion:backward(inputTable[i], targetTable[i])
         nn.utils.recursiveAdd(gradInput -1, self.gradInput[i])
         if math.abs(nn.rnn.recursiveSum(gradInput)) < 0.0001 then
            error("SequencerCriterion only decorates stateless criterions : "..tostring(self.criterion))
         end
      end
      self.isStateless = true -- test should only be run once
   end
   return self.gradInput
end
