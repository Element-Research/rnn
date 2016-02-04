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
   self.clones = {}
   self.gradInput = {}
end

function SequencerCriterion:getStepCriterion(step)
   assert(step, "expecting step at arg 1")
   local criterion = self.clones[step]
   if not criterion then
      criterion = self.criterion:clone()
      self.clones[step] = criterion
   end
   return criterion
end

function SequencerCriterion:updateOutput(inputTable, targetTable)
   self.output = 0
   
   for i,input in ipairs(inputTable) do
      local criterion = self:getStepCriterion(i)
      self.output = self.output + criterion:forward(input, targetTable[i])
   end
   
   return self.output
end

function SequencerCriterion:updateGradInput(inputTable, targetTable)
   self.gradInput = {}
   
   for i,input in ipairs(inputTable) do
      local criterion = self:getStepCriterion(i)
      self.gradInput[i] = criterion:backward(input, targetTable[i])
   end
   
   return self.gradInput
end
