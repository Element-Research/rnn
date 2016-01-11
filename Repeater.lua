------------------------------------------------------------------------
--[[ Repeater ]]--
-- Encapsulates an AbstractRecurrent instance (rnn) which is repeatedly 
-- presented with the same input for rho time steps.
-- The output is a table of rho outputs of the rnn.
------------------------------------------------------------------------
assert(not nn.Repeater, "update nnx package : luarocks install nnx")
local Repeater, parent = torch.class('nn.Repeater', 'nn.AbstractSequencer')

function Repeater:__init(module, rho)
   parent.__init(self)
   assert(torch.type(rho) == 'number', "expecting number value for arg 2")
   self.rho = rho
   self.module = (not torch.isTypeOf(rnn, 'nn.AbstractRecurrent')) and nn.Recursor(module) or module
   
   self.module:maxBPTTstep(rho) -- hijack rho (max number of time-steps for backprop)
   
   self.modules[1] = self.module
   self.output = {}
end

function Repeater:updateOutput(input)
   self.module = self.module or self.rnn -- backwards compatibility

   self.module:forget()
   -- TODO make copy outputs optional
   for step=1,self.rho do
      self.output[step] = nn.rnn.recursiveCopy(self.output[step], self.module:updateOutput(input))
   end
   return self.output
end

function Repeater:updateGradInput(input, gradOutput)
   assert(self.module.step - 1 == self.rho, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.rho, "gradOutput should have rho elements")
   
   -- back-propagate through time (BPTT)
   for step=self.rho,1,-1 do
      local gradInput = self.module:updateGradInput(input, gradOutput[step])
      if step == self.rho then
         self.gradInput = nn.rnn.recursiveCopy(self.gradInput, gradInput)
      else
         nn.rnn.recursiveAdd(self.gradInput, gradInput)
      end
   end

   return self.gradInput
end

function Repeater:accGradParameters(input, gradOutput, scale)
   assert(self.module.step - 1 == self.rho, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.rho, "gradOutput should have rho elements")
   
   -- back-propagate through time (BPTT)
   for step=self.rho,1,-1 do
      self.module:accGradParameters(input, gradOutput[step], scale)
   end
   
end

function Repeater:maxBPTTstep(rho)
   self.rho = rho
   self.module:maxBPTTstep(rho)
end

function Repeater:accUpdateGradParameters(input, gradOutput, lr)
   assert(self.module.step - 1 == self.rho, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.rho, "gradOutput should have rho elements")
   
   -- back-propagate through time (BPTT)
   for step=self.rho,1,-1 do
      self.module:accUpdateGradParameters(input, gradOutput[step], lr)
   end
end

function Repeater:__tostring__()
   local tab = '  '
   local line = '\n'
   local str = torch.type(self) .. ' {' .. line
   str = str .. tab .. '[  input,    input,  ...,  input  ]'.. line
   str = str .. tab .. '     V         V             V     '.. line
   str = str .. tab .. tostring(self.modules[1]):gsub(line, line .. tab) .. line
   str = str .. tab .. '     V         V             V     '.. line
   str = str .. tab .. '[output(1),output(2),...,output('..self.rho..')]' .. line
   str = str .. '}'
   return str
end
