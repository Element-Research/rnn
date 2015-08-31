------------------------------------------------------------------------
--[[ RecurrentAttention ]]-- 
-- Ref. A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf
-- module which takes an RNN as argument with other 
-- hyper-parameters such as the maximum number of steps, 
-- action (actions sampling module like ReinforceNormal) and 
------------------------------------------------------------------------
local RecurrentAttention, parent = torch.class("nn.RecurrentAttention", "nn.Container")

function RecurrentAttention:__init(rnn, action, nStep, hiddenSize)
   parent.__init(self)
   require 'image'
   assert(torch.isTypeOf(rnn, 'nn.AbstractRecurrent'))
   assert(torch.isTypeOf(action, 'nn.Module'))
   assert(torch.type(nStep) == 'number')
   assert(torch.type(hiddenSize) == 'table')
   assert(torch.type(hiddenSize[1]) == 'number', "Does not support table hidden layers" )
   
   self.rnn = rnn
   self.rnn.copyInputs = true
   self.action = action -- samples an x,y actions for each example
   self.hiddenSize = hiddenSize
   self.nStep = nStep
   
   self.modules = {self.rnn, self.action}
   self.sharedClones = {self.action} -- action clones
   
   self.output = {} -- rnn output
   self.actions = {} -- action output
   
   self.forwardActions = false
   
   self.gradHidden = {}
end

function RecurrentAttention:getStepModule(step)
   assert(step, "expecting step at arg 1")
   local module = self.sharedClones[step]
   if not module then
      module = self.sharedClones[1]:sharedClone()
      self.sharedClones[step] = module
   end
   -- return main, action 
   return module
end

function RecurrentAttention:updateOutput(input)
   self.rnn:forget()
   local nDim = input:dim()
   
   for step=1,self.nStep do
      -- we maintain a copy of action (with shared params) for each time-step
      local action = self:getStepModule(step)
      
      if step == 1 then
         -- sample an initial starting actions by forwarding zeros through the action
         self._initInput = self._initInput or input.new()
         self._initInput:resize(input:size(1),table.unpack(self.hiddenSize)):zero()
         self.actions[1] = action:updateOutput(self._initInput)
      else
         -- sample actions from previous hidden activation (rnn output)
         self.actions[step] = action:updateOutput(self.output[step-1])
      end
      
      -- rnn handles the recurrence internally
      local output = self.rnn:updateOutput{input, self.actions[step]}
      self.output[step] = self.forwardActions and {output, self.actions[step]} or output
   end
   
   return self.output
end

function RecurrentAttention:updateGradInput(input, gradOutput)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
   -- backward through the action
   for step=self.nStep,1,-1 do
      local action = self:getStepModule(step)
      
      local gradOutput_, gradAction_ = gradOutput[step]
      if self.forwardActions then
         gradOutput_, gradActions = unpack(gradOutput[step])
      end
      
      if step == self.nStep then
         self.gradHidden[step] = nn.rnn.recursiveCopy(self.gradHidden[step], gradOutput_)
      else
         -- gradHidden = gradOutput + gradAction
         nn.rnn.recursiveAdd(self.gradHidden[step], gradOutput_)
      end
      
      if step == 1 then
         -- backward through initial starting actions
         action:updateGradInput(self._initInput, gradAction_ or action.output)
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give action.output as a dummy variable
         local gradAction = action:updateGradInput(self.output[step-1], gradAction_ or action.output)
         self.gradHidden[step-1] = nn.rnn.recursiveCopy(self.gradHidden[step-1], gradAction)
      end
   end
   
   -- backward through the rnn layer
   for step=1,self.nStep do
      self.rnn.step = step + 1
      self.rnn:updateGradInput(input, self.gradHidden[step])
   end
   -- back-propagate through time (BPTT)
   self.rnn:updateGradInputThroughTime()
   
   for step=self.nStep,1,-1 do
      local gradInput = self.rnn.gradInputs[step][1]
      if step == self.nStep then
         self.gradInput:resizeAs(gradInput):copy(gradInput)
      else
         self.gradInput:add(gradInput)
      end
   end

   return self.gradInput
end

function RecurrentAttention:accGradParameters(input, gradOutput, scale)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
   -- backward through the action layers
   for step=self.nStep,1,-1 do
      local action = self:getStepModule(step)
      local gradAction_ = self.forwardActions and gradOutput[step][2] or nil
            
      if step == 1 then
         -- backward through initial starting actions
         action:accGradParameters(self._initInput, gradAction_ or action.output, scale)
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give action.output as a dummy variable
         action:accGradParameters(self.output[step-1], gradAction_ or action.output, scale)
      end
   end
   
   -- backward through the rnn layer
   for step=1,self.nStep do
      self.rnn.step = step + 1
      self.rnn:accGradParameters(input, self.gradHidden[step], scale)
   end
   -- back-propagate through time (BPTT)
   self.rnn:accGradParametersThroughTime()
end

function RecurrentAttention:accUpdateGradParameters(input, gradOutput, lr)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
   -- backward through the action layers
   for step=self.nStep,1,-1 do
      local action = self:getStepModule(step)
      local gradAction_ = self.forwardActions and gradOutput[step][2] or nil
      
      if step == 1 then
         -- backward through initial starting actions
         action:accUpdateGradParameters(self._initInput, gradAction_ or action.output, lr)
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give action.output as a dummy variable
         action:accUpdateGradParameters(self.output[step-1], gradAction_ or action.output, lr)
      end
   end
   
   -- backward through the rnn layer
   for step=1,self.nStep do
      self.rnn.step = step + 1
      self.rnn:accUpdateGradParameters(input, self.gradHidden[step], lr)
   end
   -- back-propagate through time (BPTT)
   self.rnn:accUpdateGradParametersThroughTime()
end

function RecurrentAttention:training()
   for i,clone in pairs(self.sharedClones) do
      clone:training()
   end
   parent.training(self)
end

function RecurrentAttention:evaluate()
   for i,clone in pairs(self.sharedClones) do
      clone:evaluate()
   end
   parent.evaluate(self)
   assert(self.train == false)
end

function RecurrentAttention:reinforce(reward)
   if torch.type(reward) == 'table' then
      error"Sequencer Error : step-wise rewards not yet supported"
   end
   
   self.rnn:reinforce(reward)
   for step=1,self.nStep do
      local action = self:getStepModule(step)
      action:reinforce(reward)
   end 
   
   local modules = self.modules
   self.modules = nil
   local ret = parent.reinforce(self, reward)
   self.modules = modules
   return ret
end

function RecurrentAttention:type(type)
   self._input = nil
   self._actions = nil
   self._crop = nil
   self._pad = nil
   self._byte = nil
   return nn.Sequencer.type(self, type)
end

function RecurrentAttention:__tostring__()
   local tab = '  '
   local line = '\n'
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {'
   str = str .. line .. tab .. 'action : ' .. tostring(self.action):gsub(line, line .. tab .. ext)
   str = str .. line .. tab .. 'rnn     : ' .. tostring(self.rnn):gsub(line, line .. tab .. ext)
   str = str .. line .. '}'
   return str
end
