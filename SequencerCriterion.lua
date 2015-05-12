------------------------------------------------------------------------
--[[ SequencerCriterion ]]--
-- Applies a criterion to each of the inputs and targets in the 
-- corresponding input and target Tables.
-- Useful for nn.Repeater and nn.Sequencer.
------------------------------------------------------------------------
local SequencerCriterion, parent = torch.class('nn.SequencerCriterion', 'nn.Criterion')

function SequencerCriterion:__init(criterion)
   parent.__init(self)
   self.criterion = criterion
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
   return self.gradInput
end

