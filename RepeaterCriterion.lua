------------------------------------------------------------------------
--[[ RepeaterCriterion ]]--
-- Applies a criterion to each of the inputs in a Table using the 
-- same target (the target is repeated). 
-- Useful for nn.Repeater and nn.Sequencer.
------------------------------------------------------------------------
local RepeaterCriterion, parent = torch.class("nn.RepeaterCriterion", "nn.Criterion")

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
      local gradInput = self.criterion:backward(input, target)
      self.gradInput[i] = self.gradInput[i] or gradInput.new()
      self.gradInput[i]:resizeAs(gradInput):copy(gradInput)
   end
   return self.gradInput
end

local function recursiveType(param, type_str)
   if torch.type(param) == 'table' then
      for i = 1, #param do
         param[i] = recursiveType(param[i], type_str)
      end
   else
      if torch.typename(param) and 
        torch.typename(param):find('torch%..+Tensor') then
         param = param:type(type_str)
      end
   end
   return param
end

function RepeaterCriterion:type(type)
   self.gradInput = recursiveType(self.gradInput)
   return self.criterion:type(type)
end
