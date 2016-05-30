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

function RepeaterCriterion:forward(input, target)
   self.output = 0
   local nStep
   if torch.isTensor(input) then
      nStep = input:size(1)
   else
      nStep = #input
   end

   
   for i=1,nStep do
      local criterion = self:getStepCriterion(i)
      self.output = self.output + criterion:forward(input[i], target)
   end
   
   return self.output
end

function RepeaterCriterion:backward(input, target)
   self.gradInput = {}
   if torch.isTensor(input) then
      nStep = input:size(1)
   else
      nStep = #input
   end
   
   local tableGradInput = {}
   for i=1,nStep do
      local criterion = self:getStepCriterion(i)
      tableGradInput[i] = criterion:backward(input[i], target)
   end
   
   if torch.isTensor(input) then
      self.gradInput = tableGradInput[1].new()
      self.gradInput:resize(nStep, unpack(tableGradInput[1]:size():totable()))
      for step=1,nStep do
         self.gradInput[step]:copy(tableGradInput[step])
      end
   else
      self.gradInput = tableGradInput
   end
   
   return self.gradInput
end
