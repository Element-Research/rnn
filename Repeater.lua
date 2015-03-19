------------------------------------------------------------------------
--[[ Repeater ]]--
-- Encapsulates an AbstractRecurrent instance (rnn) which is repeatedly 
-- presented with the same input for nStep time steps.
-- The output is a table of nStep outputs of the rnn.
------------------------------------------------------------------------
local Repeater, parent = torch.class("nn.Repeater", "nn.Container")

function Repeater:__init(rnn, nStep)
   parent.__init(self)
   assert(torch.type(nStep) == 'number', "expecting number value for arg 2")
   self.nStep = nStep
   self.rnn = rnn
   assert(rnn.backwardThroughTime, "expecting AbstractRecurrent instance for arg 1")
   self.modules[1] = rnn
   self.output = {}
end

function Repeater:updateOutput(input)
   self.rnn:forget()
   for step=1,self.nStep do
      self.output[step] = self.rnn:updateOutput(input)
   end
   return self.output
end

local recursiveAdd = nn.AbstractRecurrent.recursiveAdd
local recursiveCopy = nn.AbstractRecurrent.recursiveCopy

function Repeater:updateGradInput(input, gradOutput)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
   for step=1,self.nStep do
      self.rnn.step = step + 1
      self.rnn:updateGradInput(input, gradOutput[step])
   end
   -- back-propagate through time (BPTT)
   self.rnn:updateGradInputThroughTime()
   
   for i,currentGradInput in ipairs(self.rnn.gradInputs) do
      if i == 1 then
         self.gradInput = recursiveCopy(self.gradInput, currentGradInput)
      else
         recursiveAdd(self.gradInput, currentGradInput)
      end
   end
   
   return self.gradInput
end

function Repeater:accGradParameters(input, gradOutput, scale)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
   for step=1,self.nStep do
      self.rnn.step = step + 1
      self.rnn:accGradParameters(input, gradOutput[step], scale)
   end
   -- back-propagate through time (BPTT)
   self.rnn:accGradParametersThroughTime()
end

function Repeater:accUpdateGradParameters(input, gradOutput, lr)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
   for step=1,self.nStep do
      self.rnn.step = step + 1
      self.rnn:accGradParameters(input, gradOutput[step], 1)
   end
   -- back-propagate through time (BPTT)
   self.rnn:accUpdateGradParametersThroughTime(lr)
end

function Repeater:__tostring__()
   local tab = '  '
   local line = '\n'
   local str = torch.type(self) .. ' {' .. line
   str = str .. tab .. '[  input,    input,  ...,  input  ]'.. line
   str = str .. tab .. '     V         V             V     '.. line
   str = str .. tab .. tostring(self.modules[1]):gsub(line, line .. tab) .. line
   str = str .. tab .. '     V         V             V     '.. line
   str = str .. tab .. '[output(1),output(2),...,output('..self.nStep..')]' .. line
   str = str .. '}'
   return str
end
