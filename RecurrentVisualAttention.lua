------------------------------------------------------------------------
--[[ RecurrentVisualAttention ]]-- 
-- Ref. A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf
-- module which takes an RNN as argument with other 
-- hyper-parameters such as the maximum number of steps, 
-- locator (location sampling module like ReinforceNormal) and 
-- main (classification, segmentation, etc) module.
------------------------------------------------------------------------
local RVA, parent = torch.class("nn.RecurrentVisualAttention", "nn.Container")

function RVA:__init(rnn, main, locator, nStep, hiddenSize)
   parent.__init(self)
   require 'image'
   assert(torch.isTypeOf(rnn, 'nn.Module') and rnn.forget)
   assert(torch.isTypeOf(main, 'nn.Module'))
   assert(torch.isTypeOf(locator, 'nn.Module'))
   assert(torch.type(nStep) == 'number')
   assert(torch.type(hiddenSize) == 'table')
   assert(torch.type(hiddenSize[1]) == 'number', "Does not support table hidden layers" )
   
   self.rnn = rnn
   self.rnn.copyInputs = true
   self.main = main -- main task (classification, segmentation, etc.)
   self.locator = locator -- samples an x,y location for each example
   self.hiddenSize = hiddenSize
   self.nStep = nStep
   
   self.modules = {self.rnn, self.main, self.locator}
   -- we combine these into a Container to simplify self:getSetModule()
   local outputModule = nn.Container():add(self.main):add(self.locator)
   self.sharedClones = {outputModule} -- action clones
   
   self.output = {}
   self.glimpse = {}
   self.hidden = {}
   self.gradHidden = {}
   self.location = {}
end

function RVA:initGlimpseSensor(glimpseSize, glimpseDepth, glimpseScale)
   self.glimpseSize = glimpseSize -- height == width
   self.glimpseDepth = glimpseDepth or 3
   self.glimpseScale = glimplseScale or 2
   
   assert(torch.type(self.glimpseSize) == 'number')
   assert(torch.type(self.glimpseDepth) == 'number')
   assert(torch.type(self.glimpseScale) == 'number')
end

-- a bandwidth limited sensor which focuses on a location
-- locations index the x,y coord of the top-left corner
function RVA:glimpseSensor(glimpse, input, location)
   assert(self.glimpseSize, "glimpseSensor not initialize")
   glimpse:resize(input:size(1), self.glimpseDepth, input:size(2), self.glimpseSize, self.glimpseSize)
   
   -- handle cuda inputs
   local cuglimpse
   if torch.type(glimpse) == 'torch.CudaTensor' then
      cuglimpse = glimpse
      self._glimpse = self._glimpse or torch.FloatTensor()
      self._glimpse:resize(cuglimpse:size()):copy(cuglimpse)
      glimpse = self._glimpse
   end
   
   if torch.type(input) == 'torch.CudaTensor' then
      self._input = self._input or torch.FloatTensor()
      self._input:resize(input:size()):copy(input)
      input = self._input
   end
   
   local culocation
   if torch.type(location) == 'torch.CudaTensor' then
      self._location = self._location or torch.FloatTensor()
      self._location:resize(location:size()):copy(location)
      location = self._location
   end
   
   self._crop = self._crop or glimpse.new()
   self._pad = self._pad or input.new()
   
   for sampleIdx=1,glimpse:size(1) do
      local glimpseSample = glimpse[sampleIdx]
      local inputSample = input[sampleIdx]
      local xy = location[sampleIdx]
      
      for depth=1,self.glimpseDepth do
         local dst = glimpseSample[depth]
         if depth == 1 then
            -- add 1 because image.crop is zero-indexed
            image.crop(dst, inputSample, xy:select(1,1)-1, xy:select(1,2)-1)
         else
            -- pad, crop, downscale
            local glimpseSize = self.glimpseSize*(self.glimpseScale^(depth-1))
            local padSize = (glimpseSize-self.glimpseSize)/2
            self._pad:resize(input:size(2), input:size(3)+padSize*2, input:size(4)+padSize*2):zero()
            local center = self._pad:narrow(2,padSize,input:size(3)):narrow(3,padSize,input:size(4))
            center:copy(inputSample)
            self._crop:resize(input:size(2), glimpseSize, glimpseSize)
            image.crop(self._crop, self._pad, xy:select(1,1)-1, xy:select(1,2)-1)
            image.scale(dst, self._crop)
         end
      end
   end
   
   if cuglimpse then
      cuglimpse:copy(glimpse)
      glimpse = cuglimpse
   end
   glimpse:resize(input:size(1), self.glimpseDepth*input:size(2), self.glimpseSize, self.glimpseSize)
   return glimpse
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
      -- we maintain a copy of main and locator (with shared params) for each time-step
      local main, locator = self:getStepModule(step)
      
      if step == 1 then
         -- sample an initial starting location by forwarding zeros through the locator
         self._initInput = self._initInput or input.new()
         self._initInput:resize(input:size(1),table.unpack(self.hiddenSize)):zero()
         self.location[1] = locator:updateOutput(self._initInput)
      else
         -- sample location from previous hidden activation (rnn output)
         self.location[step] = locator:updateOutput(self.hidden[step-1])
      end
      local location = self.location[step]
      
      -- glimpse is the concatenation of down-scaled cropped images of increasing scale around a location
      self.glimpse[step] = self:glimpseSensor(self.glimpse[step] or input.new(), input, location)
      
      -- rnn handles the recurrence internally
      self.hidden[step] = self.rnn:updateOutput{location, self.glimpse[step]}
      
      -- main tasks are the only things that will be forwarded.
      -- the locator is updated via the REINFORCE rule
      self.output[step] = main:updateOutput(self.hidden[step])
   end
   return self.output
end

function RVA:updateGradInput(input, gradOutput)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
   -- backward through the main and locator layers
   for step=self.nStep,1,-1 do
      local main, locator = self:getStepModule(step)
      local gradMain = main:udpateGradInput(self.hidden[step], gradOutput[step])
      
      if step == self.nStep then
         self.gradHidden[step] = nn.rnn.recursiveCopy(self.gradHidden[step], gradMain)
      else
         -- gradHidden = gradMain + gradLocator
         nn.rnn.recursiveAdd(self.gradHidden[step], gradMain)
      end
      
      if step == 1 then
         -- backward through initial starting location
         locator:updateGradInput(self._initInput, locator.output)
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give locator.output as a dummy variable
         local gradLocator = locator:updateGradInput(self.hidden[step-1], locator.output) 
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
   
   -- for now, we assume self is at the input of the graph (no gradInput)
   -- so we use a dummy gradInput pointing to input
   self.gradInput:set(input)

   return self.gradInput
end

function RVA:accGradParameters(input, gradOutput, scale)
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

function RVA:accUpdateGradParameters(input, gradOutput, lr)
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

-- annotates image with path taken by attention
function RVA:annotate(input)
   
end

function RVA:reinforce(reward)
   parent.reinforce(self, reward)
   error"TODO : dont forget to reward sharedClones"
end

function RVA:type(type)
   self._input = nil
   self._location = nil
   self._glimpse = nil
   self._crop = nil
   self._pad = nil
   self._byte = nil
   return parent.type(self, type)
end

function RVA:__tostring__()
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
