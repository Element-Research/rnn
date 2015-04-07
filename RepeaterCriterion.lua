------------------------------------------------------------------------
--[[ RepeaterCriterion ]]--
-- Applies a criterion to each of the inputs in a Table using the 
-- same target (the target is repeated). 
-- Useful for nn.Repeater and nn.Sequencer.
------------------------------------------------------------------------
local RepeaterCriterion, parent
if nn.RepeaterCriterion then -- prevent name conflicts with nnx
   RepeaterCriterion, parent = nn.RepeaterCriterion, nn.Criterion
else
   RepeaterCriterion, parent = torch.class('nn.RepeaterCriterion', 'nn.Criterion')
end

function RepeaterCriterion:__init(criterion)
   parent.__init(self)
   self.criterion = criterion
   self.gradInput = {}
end

function RepeaterCriterion:forward(inputTable, target)
   self.output = 0
   for i,input in ipairs(inputTable) do
      self.output = self.output + self.criterion:forward(input, target)
   end
   return self.output
end

function RepeaterCriterion:backward(inputTable, target)
   for i,input in ipairs(inputTable) do
      self.gradInput[i] = rnn.recusiveCopy(self.gradInput[i], self.criterion:backward(input, target))
   end
   return self.gradInput
end

function RepeaterCriterion:type(type)
   self.gradInput = rnn.recursiveType(self.gradInput)
   return self.criterion:type(type)
end
