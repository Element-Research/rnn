------------------------------------------------------------------------
--[[ RecurrentVisualAttention ]]-- 
-- Ref. A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf
-- module which takes an RNN as argument with other 
-- hyper-parameters such as the maximum number of steps, 
-- locator (location sampling module like ReinforceNormal) and 
-- sensor (produces glimpses given image and location) module.
------------------------------------------------------------------------
local RVA, parent = torch.class("nn.RecurrentVisualAttention", "nn.Container")

function RVA:__init(rnn, sensor, locator, nStep, hiddenSize)
   parent.__init(self)
   require 'image'
   assert(torch.isTypeOf(rnn, 'nn.Module') and rnn.forget)
   assert(torch.isTypeOf(sensor, 'nn.Module'))
   assert(torch.isTypeOf(locator, 'nn.Module'))
   assert(torch.type(nStep) == 'number')
   assert(torch.type(hiddenSize) == 'table')
   assert(torch.type(hiddenSize[1]) == 'number', "Does not support table hidden layers" )
   
   self.rnn = rnn
   self.rnn.copyInputs = true
   self.sensor = sensor -- produces glimpse given image and location
   self.locator = locator -- samples an x,y location for each example
   self.hiddenSize = hiddenSize
   self.nStep = nStep
   
   self.modules = {self.rnn, self.sensor, self.locator}
   -- we combine these into a Container to simplify self:getSetModule()
   local outputModule = nn.Container():add(self.sensor):add(self.locator)
   self.sharedClones = {outputModule} -- action clones
   
   self.output = {} -- rnn output
   self.glimpse = {} -- sensor output (called glimpses in original paper)
   self.location = {} -- locator output
   
   self.gradHidden = {}
end

function RVA:getStepModule(step)
   assert(step, "expecting step at arg 1")
   local module = self.sharedClones[step]
   if not module then
      module = self.sharedClones[1]:sharedClone()
      self.sharedClones[step] = module
   end
   -- return main, locator 
   return module:get(1), module:get(2)
end

function RVA:updateOutput(input)
   self.rnn:forget()
   local nDim = input:dim()
   assert(nDim == 4, "only works with batch of images")
   
   for step=1,self.nStep do
      -- we maintain a copy of sensor and locator (with shared params) for each time-step
      local sensor, locator = self:getStepModule(step)
      
      if step == 1 then
         -- sample an initial starting location by forwarding zeros through the locator
         self._initInput = self._initInput or input.new()
         self._initInput:resize(input:size(1),table.unpack(self.hiddenSize)):zero()
         self.location[1] = locator:updateOutput(self._initInput)
      else
         -- sample location from previous hidden activation (rnn output)
         self.location[step] = locator:updateOutput(self.output[step-1])
      end
      local location = self.location[step]
      
      -- sensor generates transformation of image given input and location
      self.glimpse[step] = sensor:updateOutput{input, location}
      
      -- rnn handles the recurrence internally
      self.output[step] = self.rnn:updateOutput{location, self.glimpse[step]}
   end
   
   return self.output
end

function RVA:updateGradInput(input, gradOutput)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
   -- backward through the locator
   for step=self.nStep,1,-1 do
      local sensor, locator = self:getStepModule(step)
      
      if step == self.nStep then
         self.gradHidden[step] = nn.rnn.recursiveCopy(self.gradHidden[step], gradOutput[step])
      else
         -- gradHidden = gradOutput + gradLocator
         nn.rnn.recursiveAdd(self.gradHidden[step], gradOutput[step])
      end
      
      if step == 1 then
         -- backward through initial starting location
         locator:updateGradInput(self._initInput, locator.output)
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give locator.output as a dummy variable
         local gradLocator = locator:updateGradInput(self.output[step-1], locator.output)
         self.gradHidden[step-1] = nn.rnn.recursiveCopy(self.gradHidden[step-1], gradLocator)
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
      local sensor = self:getStepModule(step)
      local gradInput = sensor:updateGradInput({input, self.location[step]}, self.rnn.gradInputs[step][2])[1]
      if step == self.nStep then
         self.gradInput:resizeAs(gradInput):copy(gradInput)
      else
         self.gradInput:add(gradInput)
      end
   end

   return self.gradInput
end

function RVA:accGradParameters(input, gradOutput, scale)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
   -- backward through the locator layers
   for step=self.nStep,1,-1 do
      local sensor, locator = self:getStepModule(step)
            
      if step == 1 then
         -- backward through initial starting location
         locator:accGradParameters(self._initInput, locator.output, scale)
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give locator.output as a dummy variable
         locator:accGradParameters(self.output[step-1], locator.output, scale)
      end
   end
   
   -- backward through the rnn layer
   for step=1,self.nStep do
      self.rnn.step = step + 1
      self.rnn:accGradParameters(input, self.gradHidden[step], scale)
   end
   -- back-propagate through time (BPTT)
   self.rnn:accGradParametersThroughTime()
   
   for step=self.nStep,1,-1 do
      local sensor = self:getStepModule(step)
      sensor:accGradParameters({input, self.location[step]}, self.rnn.gradInputs[step][2], scale)
   end
end

function RVA:accUpdateGradParameters(input, gradOutput, lr)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
   -- backward through the locator layers
   for step=self.nStep,1,-1 do
      local sensor, locator = self:getStepModule(step)
      
      if step == 1 then
         -- backward through initial starting location
         locator:accUpdateGradParameters(self._initInput, locator.output, lr)
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give locator.output as a dummy variable
         locator:accUpdateGradParameters(self.output[step-1], locator.output, lr)
      end
   end
   
   -- backward through the rnn layer
   for step=1,self.nStep do
      self.rnn.step = step + 1
      self.rnn:accUpdateGradParameters(input, self.gradHidden[step], lr)
   end
   -- back-propagate through time (BPTT)
   self.rnn:accUpdateGradParametersThroughTime()
   
   for step=self.nStep,1,-1 do
      local sensor = self:getStepModule(step)
      sensor:accUpdateGradParameters({input, self.location[step]}, self.rnn.gradInputs[step][2], lr)
   end
end

-- annotates image with path taken by attention
function RVA:annotate(input)
   
end

function RVA:reinforce(reward)
   if torch.type(reward) == 'table' then
      error"Sequencer Error : step-wise rewards not yet supported"
   end
   
   self.rnn:reinforce(reward)
   for step=1,self.nStep do
      local sensor, locator = self:getStepModule(step)
      sensor:reinforce(reward)
      locator:reinforce(reward)
   end 
   
   local modules = self.modules
   self.modules = nil
   local ret = parent.reinforce(self, reward)
   self.modules = modules
   return ret
end

function RVA:type(type)
   self._input = nil
   self._location = nil
   self._glimpse = nil
   self._crop = nil
   self._pad = nil
   self._byte = nil
   return nn.Sequencer.type(self, type)
end

function RVA:__tostring__()
   local tab = '  '
   local line = '\n'
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {'
   str = str .. line .. tab .. 'locator : ' .. tostring(self.locator):gsub(line, line .. tab .. ext)
   str = str .. line .. tab .. 'sensor  : ' .. tostring(self.sensor):gsub(line, line .. tab .. ext)
   str = str .. line .. tab .. 'rnn     : ' .. tostring(self.rnn):gsub(line, line .. tab .. ext)
   str = str .. line .. '}'
   return str
end
