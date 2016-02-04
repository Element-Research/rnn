------------------------------------------------------------------------
--[[ RepeaterCriterion ]]--
-- Applies a criterion to each of the inputs in a Table using the
-- same target (the target is repeated).
-- Useful for nn.Repeater and nn.Sequencer.
------------------------------------------------------------------------
assert(not nn.RepeaterCriterion, "update nnx package : luarocks install nnx")
local RepeaterCriterion, parent = torch.class('nn.RepeaterCriterion', 'nn.Criterion')

function RepeaterCriterion:__init(criterion)
   parent.__init(self)
   self.criterion = criterion
   self.gradInput = {}
   self.clones = {}
end

RepeaterCriterion.getStepCriterion = nn.SequencerCriterion.getStepCriterion

function RepeaterCriterion:forward(inputTable, target)
   self.output = 0
   
   for i,input in ipairs(inputTable) do
      local criterion = self:getStepCriterion(i)
      self.output = self.output + criterion:forward(input, target)
   end
   
   return self.output
end

function RepeaterCriterion:backward(inputTable, target)
   self.gradInput = {}
   
   for i,input in ipairs(inputTable) do
      local criterion = self:getStepCriterion(i)
      self.gradInput[i] = criterion:backward(input, target)
   end
   
   return self.gradInput
end
