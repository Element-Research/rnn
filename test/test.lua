
local rnntest = {}
local precision = 1e-5
local mytester
local benchmark = false

local makeOldRecurrent_isdone = false
local function makeOldRecurrent()
   
   if makeOldRecurrent_isdone then
      return
   end
   makeOldRecurrent_isdone = true
   -- I am making major modifications to nn.Recurrent.
   -- So I want to make sure the new version matches the old
   local AbstractRecurrent, parent = torch.class('nn.ARTest', 'nn.Container')

   function AbstractRecurrent:__init(rho)
      parent.__init(self)
      
      self.rho = rho --the maximum number of time steps to BPTT
      
      self.fastBackward = true
      self.copyInputs = true
      self.copyGradOutputs = true
      
      self.inputs = {}
      self.outputs = {}
      self._gradOutputs = {}
      self.gradOutputs = {}
      self.scales = {}
      
      self.gradParametersAccumulated = false
      self.onlineBackward = false
      self.step = 1
      
      -- stores internal states of Modules at different time-steps
      self.sharedClones = {}
      
      self:reset()
   end

   function AbstractRecurrent:getStepModule(step)
      assert(step, "expecting step at arg 1")
      local recurrentModule = self.sharedClones[step]
      if not recurrentModule then
         recurrentModule = self.recurrentModule:stepClone()
         self.sharedClones[step] = recurrentModule
      end
      return recurrentModule
   end

   function AbstractRecurrent:maskZero(nInputDim)
      self.recurrentModule = nn.MaskZero(self.recurrentModule, nInputDim)
      return self
   end

   function AbstractRecurrent:updateGradInput(input, gradOutput)      
      if self.onlineBackward then
         -- updateGradInput will be called in reverse order of time
         self.updateGradInputStep = self.updateGradInputStep or self.step
         if self.copyGradOutputs then
            self.gradOutputs[self.updateGradInputStep-1] = nn.rnn.recursiveCopy(self.gradOutputs[self.updateGradInputStep-1] , gradOutput)
         else
            self.gradOutputs[self.updateGradInputStep-1] = self.gradOutputs[self.updateGradInputStep-1] or nn.rnn.recursiveNew(gradOutput)
            nn.rnn.recursiveSet(self.gradOutputs[self.updateGradInputStep-1], gradOutput)
         end
         
         -- BPTT for one time-step (rho = 1)
         self.gradInput = self:updateGradInputThroughTime(self.updateGradInputStep, 1)
         
         self.updateGradInputStep = self.updateGradInputStep - 1
         assert(self.gradInput, "Missing gradInput")
         return self.gradInput
      else
         -- Back-Propagate Through Time (BPTT) happens in updateParameters()
         -- for now we just keep a list of the gradOutputs
         if self.copyGradOutputs then
            self.gradOutputs[self.step-1] = nn.rnn.recursiveCopy(self.gradOutputs[self.step-1] , gradOutput)
         else
            self.gradOutputs[self.step-1] = self.gradOutputs[self.step-1] or nn.rnn.recursiveNew(gradOutput)
            nn.rnn.recursiveSet(self.gradOutputs[self.step-1], gradOutput)
         end
      end
   end

   function AbstractRecurrent:accGradParameters(input, gradOutput, scale)
      if self.onlineBackward then
         -- accGradParameters will be called in reverse order of time
         assert(self.updateGradInputStep < self.step, "Missing updateGradInput")
         self.accGradParametersStep = self.accGradParametersStep or self.step
         self.scales[self.accGradParametersStep-1] = scale or 1
         
         -- BPTT for one time-step (rho = 1)
         self:accGradParametersThroughTime(self.accGradParametersStep, 1)
         
         self.accGradParametersStep = self.accGradParametersStep - 1
      else
         -- Back-Propagate Through Time (BPTT) happens in updateParameters()
         -- for now we just keep a list of the scales
         self.scales[self.step-1] = scale or 1
      end
   end

   function AbstractRecurrent:backwardUpdateThroughTime(learningRate)
      local gradInput = self:updateGradInputThroughTime()
      self:accUpdateGradParametersThroughTime(learningRate)
      return gradInput
   end

   -- this is only useful when calling updateParameters directly on the rnn
   -- Note that a call to updateParameters on an rnn container DOES NOT call this method
   function AbstractRecurrent:updateParameters(learningRate)
      if self.gradParametersAccumulated then
         for i=1,#self.modules do
            self.modules[i]:updateParameters(learningRate)
         end
      else
         self:backwardUpdateThroughTime(learningRate)
      end
   end

   -- goes hand in hand with the next method : forget()
   -- this methods brings the oldest memory to the current step
   function AbstractRecurrent:recycle(offset)
      -- offset can be used to skip initialModule (if any)
      offset = offset or 0
      -- pad rho with one extra time-step of memory (helps for Sequencer:remember()).
      -- also, rho could have been manually increased or decreased
      local rho = math.max(self.rho+1, _.size(self.sharedClones) or 0)
      if self.step > rho + offset then
         assert(self.sharedClones[self.step] == nil)
         self.sharedClones[self.step] = self.sharedClones[self.step-rho]
         self.sharedClones[self.step-rho] = nil
      end
      
      rho = math.max(self.rho+1, _.size(self.outputs) or 0)
      if self.step > rho + offset then
         -- need to keep rho+1 of these
         assert(self.outputs[self.step] == nil)
         self.outputs[self.step] = self.outputs[self.step-rho-1] 
         self.outputs[self.step-rho-1] = nil
      end
      
      rho = math.max(self.rho+1, _.size(self.inputs) or 0)
      if self.step > rho then
         assert(self.inputs[self.step] == nil)
         assert(self.gradOutputs[self.step] == nil)
         assert(self._gradOutputs[self.step] == nil)
         self.inputs[self.step] = self.inputs[self.step-rho]
         self.inputs[self.step-rho] = nil      
         self.gradOutputs[self.step] = self.gradOutputs[self.step-rho] 
         self._gradOutputs[self.step] = self._gradOutputs[self.step-rho]
         self.gradOutputs[self.step-rho] = nil
         self._gradOutputs[self.step-rho] = nil
         self.scales[self.step-rho] = nil
      end
      
      return self
   end

   -- this method brings all the memory back to the start
   function AbstractRecurrent:forget(offset)
      offset = offset or 0
      
       -- bring all states back to the start of the sequence buffers
      if self.train ~= false then
         self.outputs = _.compact(self.outputs)
         self.sharedClones = _.compact(self.sharedClones)
         self.inputs = _.compact(self.inputs)
         
         self.scales = {}
         self.gradOutputs = _.compact(self.gradOutputs)
         self._gradOutputs = _.compact(self._gradOutputs)
      end
      
      -- forget the past inputs; restart from first step
      self.step = 1
      return self
   end

   function AbstractRecurrent:includingSharedClones(f)
      local modules = self.modules
      local sharedClones = self.sharedClones
      self.sharedClones = nil
      self.modules = {}
      for i,modules in ipairs{modules, sharedClones} do
         for j, module in pairs(modules) do
            table.insert(self.modules, module)
         end
      end
      local r = f()
      self.modules = modules
      self.sharedClones = sharedClones
      return r
   end

   function AbstractRecurrent:type(type)
      return self:includingSharedClones(function()
         return parent.type(self, type)
      end)
   end

   function AbstractRecurrent:training()
      return self:includingSharedClones(function()
         return parent.training(self)
      end)
   end

   function AbstractRecurrent:evaluate()
      return self:includingSharedClones(function()
         return parent.evaluate(self)
      end)
   end

   function AbstractRecurrent:reinforce(reward)
      return self:includingSharedClones(function()
         return parent.reinforce(self, reward)
      end)
   end

   function AbstractRecurrent:sharedClone(shareParams, shareGradParams, clones, pointers, stepClone)
      if stepClone then 
         return self 
      else
         return parent.sharedClone(self, shareParams, shareGradParams, clones, pointers, stepClone)
      end
   end

   function AbstractRecurrent:backwardOnline(online)
      self.onlineBackward = (online == nil) and true or online
   end

   function AbstractRecurrent:maxBPTTstep(rho)
      self.rho = rho
   end

   -- backwards compatibility
   AbstractRecurrent.recursiveResizeAs = rnn.recursiveResizeAs
   AbstractRecurrent.recursiveSet = rnn.recursiveSet
   AbstractRecurrent.recursiveCopy = rnn.recursiveCopy
   AbstractRecurrent.recursiveAdd = rnn.recursiveAdd
   AbstractRecurrent.recursiveTensorEq = rnn.recursiveTensorEq
   AbstractRecurrent.recursiveNormal = rnn.recursiveNormal
   
   local Recurrent, parent = torch.class('nn.ReTest', 'nn.ARTest')

   function Recurrent:__init(start, input, feedback, transfer, rho, merge)
      parent.__init(self, rho or 5)
      
      local ts = torch.type(start)
      if ts == 'torch.LongStorage' or ts == 'number' then
         start = nn.Add(start)
      elseif ts == 'table' then
         start = nn.Add(torch.LongStorage(start))
      elseif not torch.isTypeOf(start, 'nn.Module') then
         error"Recurrent : expecting arg 1 of type nn.Module, torch.LongStorage, number or table"
      end
      
      self.startModule = start
      self.inputModule = input
      self.feedbackModule = feedback
      self.transferModule = transfer or nn.Sigmoid()
      self.mergeModule = merge or nn.CAddTable()
      
      self.modules = {self.startModule, self.inputModule, self.feedbackModule, self.transferModule, self.mergeModule}
      
      self:buildInitialModule()
      self:buildRecurrentModule()
      self.sharedClones[2] = self.recurrentModule 
   end

   -- build module used for the first step (steps == 1)
   function Recurrent:buildInitialModule()
      self.initialModule = nn.Sequential()
      self.initialModule:add(self.inputModule:sharedClone())
      self.initialModule:add(self.startModule)
      self.initialModule:add(self.transferModule:sharedClone())
   end

   -- build module used for the other steps (steps > 1)
   function Recurrent:buildRecurrentModule()
      local parallelModule = nn.ParallelTable()
      parallelModule:add(self.inputModule)
      parallelModule:add(self.feedbackModule)
      self.recurrentModule = nn.Sequential()
      self.recurrentModule:add(parallelModule)
      self.recurrentModule:add(self.mergeModule)
      self.recurrentModule:add(self.transferModule)
   end

   function Recurrent:updateOutput(input)
      -- output(t) = transfer(feedback(output_(t-1)) + input(input_(t)))
      local output
      if self.step == 1 then
         output = self.initialModule:updateOutput(input)
      else
         if self.train ~= false then
            -- set/save the output states
            self:recycle()
            local recurrentModule = self:getStepModule(self.step)
             -- self.output is the previous output of this module
            output = recurrentModule:updateOutput{input, self.output}
         else
            -- self.output is the previous output of this module
            output = self.recurrentModule:updateOutput{input, self.output}
         end
      end
      
      if self.train ~= false then
         local input_ = self.inputs[self.step]
         self.inputs[self.step] = self.copyInputs 
            and nn.rnn.recursiveCopy(input_, input) 
            or nn.rnn.recursiveSet(input_, input)     
      end
      
      self.outputs[self.step] = output
      self.output = output
      self.step = self.step + 1
      self.gradPrevOutput = nil
      self.updateGradInputStep = nil
      self.accGradParametersStep = nil
      self.gradParametersAccumulated = false
      return self.output
   end

   -- not to be confused with the hit movie Back to the Future
   function Recurrent:backwardThroughTime(timeStep, timeRho)
      timeStep = timeStep or self.step
      local rho = math.min(timeRho or self.rho, timeStep-1)
      local stop = timeStep - rho
      local gradInput
      if self.fastBackward then
         self.gradInputs = {}
         for step=timeStep-1,math.max(stop, 2),-1 do
            local recurrentModule = self:getStepModule(step)
            
            -- backward propagate through this step
            local input = self.inputs[step]
            local output = self.outputs[step-1]
            local gradOutput = self.gradOutputs[step] 
            if self.gradPrevOutput then
               self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
               nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
               gradOutput = self._gradOutputs[step]
            end
            local scale = self.scales[step]
            
            gradInput, self.gradPrevOutput = unpack(recurrentModule:backward({input, output}, gradOutput, scale))
            
            table.insert(self.gradInputs, 1, gradInput)
         end
         
         if stop <= 1 then
            -- backward propagate through first step
            local input = self.inputs[1]
            local gradOutput = self.gradOutputs[1]
            if self.gradPrevOutput then
               self._gradOutputs[1] = nn.rnn.recursiveCopy(self._gradOutputs[1], self.gradPrevOutput)
               nn.rnn.recursiveAdd(self._gradOutputs[1], gradOutput)
               gradOutput = self._gradOutputs[1]
            end
            local scale = self.scales[1]
            gradInput = self.initialModule:backward(input, gradOutput, scale)
            table.insert(self.gradInputs, 1, gradInput)
         end
         self.gradParametersAccumulated = true
      else
         gradInput = self:updateGradInputThroughTime(timeStep, timeRho)
         self:accGradParametersThroughTime(timeStep, timeRho)
      end
      return gradInput
   end

   function Recurrent:updateGradInputThroughTime(timeStep, rho)
      assert(self.step > 1, "expecting at least one updateOutput")
      timeStep = timeStep or self.step
      self.gradInputs = {}
      local gradInput
      local rho = math.min(rho or self.rho, timeStep-1)
      local stop = timeStep - rho
      for step=timeStep-1,math.max(stop,2),-1 do
         local recurrentModule = self:getStepModule(step)
         
         -- backward propagate through this step
         local input = self.inputs[step]
         local output = self.outputs[step-1]
         local gradOutput = self.gradOutputs[step]
         if self.gradPrevOutput then
            self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
            nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
            gradOutput = self._gradOutputs[step]
         end

         gradInput, self.gradPrevOutput = unpack(recurrentModule:updateGradInput({input, output}, gradOutput))
         table.insert(self.gradInputs, 1, gradInput)
      end
      
      if stop <= 1 then      
         -- backward propagate through first step
         local input = self.inputs[1]
         local gradOutput = self.gradOutputs[1]
         if self.gradPrevOutput then
            self._gradOutputs[1] = nn.rnn.recursiveCopy(self._gradOutputs[1], self.gradPrevOutput)
            nn.rnn.recursiveAdd(self._gradOutputs[1], gradOutput)
            gradOutput = self._gradOutputs[1]
         end
         gradInput = self.initialModule:updateGradInput(input, gradOutput)
         table.insert(self.gradInputs, 1, gradInput)
      end
      
      return gradInput
   end

   function Recurrent:accGradParametersThroughTime(timeStep, rho)
      timeStep = timeStep or self.step
      local rho = math.min(rho or self.rho, timeStep-1)
      local stop = timeStep - rho
      for step=timeStep-1,math.max(stop,2),-1 do
         local recurrentModule = self:getStepModule(step)
         
         -- backward propagate through this step
         local input = self.inputs[step]
         local output = self.outputs[step-1]
         local gradOutput = (step == self.step-1) and self.gradOutputs[step] or self._gradOutputs[step]

         local scale = self.scales[step]
         recurrentModule:accGradParameters({input, output}, gradOutput, scale)
      end
      
      if stop <= 1 then
         -- backward propagate through first step
         local input = self.inputs[1]
         local gradOutput = (1 == self.step-1) and self.gradOutputs[1] or self._gradOutputs[1]
         local scale = self.scales[1]
         self.initialModule:accGradParameters(input, gradOutput, scale)
      end
      
      self.gradParametersAccumulated = true
      return gradInput
   end

   function Recurrent:accUpdateGradParametersThroughInitialModule(lr, rho)
      if self.initialModule:size() ~= 3 then
         error("only works with Recurrent:buildInitialModule(). "..
         "Reimplement this method to work with your subclass."..
         "Or use accGradParametersThroughTime instead of accUpdateGrad...")
      end
      
      -- backward propagate through first step
      local input = self.inputs[1]
      local gradOutput = (1 == self.step-1) and self.gradOutputs[1] or self._gradOutputs[1]
      local scale = self.scales[1]
      local inputModule = self.initialModule:get(1)
      local startModule = self.initialModule:get(2)
      local transferModule = self.initialModule:get(3)
      inputModule:accUpdateGradParameters(input, self.startModule.gradInput, lr*scale)
      startModule:accUpdateGradParameters(inputModule.output, transferModule.gradInput, lr*scale)
      transferModule:accUpdateGradParameters(startModule.output, gradOutput, lr*scale)
   end

   function Recurrent:accUpdateGradParametersThroughTime(lr, timeStep, rho)
      timeStep = timeStep or self.step
      local rho = math.min(rho or self.rho, timeStep-1)
      local stop = timeStep - rho
      for step=timeStep-1,math.max(stop,2),-1 do
         local recurrentModule = self:getStepModule(step)
         
         -- backward propagate through this step
         local input = self.inputs[step]
         local output = self.outputs[step-1]
         local gradOutput = (step == self.step-1) and self.gradOutputs[step] or self._gradOutputs[step]

         local scale = self.scales[step]
         recurrentModule:accUpdateGradParameters({input, output}, gradOutput, lr*scale)
      end
      
      if stop <= 1 then      
         self:accUpdateGradParametersThroughInitialModule(lr, rho)
      end
      
      return gradInput
   end

   function Recurrent:recycle()
      return parent.recycle(self, 1)
   end

   function Recurrent:forget()
      return parent.forget(self, 1)
   end

   function Recurrent:includingSharedClones(f)
      local modules = self.modules
      self.modules = {}
      local sharedClones = self.sharedClones
      self.sharedClones = nil
      local initModule = self.initialModule
      self.initialModule = nil
      for i,modules in ipairs{modules, sharedClones, {initModule}} do
         for j, module in pairs(modules) do
            table.insert(self.modules, module)
         end
      end
      local r = f()
      self.modules = modules
      self.sharedClones = sharedClones
      self.initialModule = initModule 
      return r
   end
end

function rnntest.Recurrent_old()
   -- make sure the new version is still as good as the last version
   makeOldRecurrent()
   
   local batchSize = 2
   local hiddenSize = 10
   local nStep = 3
   
   -- recurrent neural network
   local rnn = nn.Recurrent(
      hiddenSize, 
      nn.Linear(hiddenSize, hiddenSize),
      nn.Linear(hiddenSize, hiddenSize), 
      nn.ReLU(), 99999
   )
   
   local rnn2 = nn.ReTest(
      rnn.startModule:clone(),
      rnn.inputModule:clone(),
      rnn.feedbackModule:clone(),
      nn.ReLU(), 99999
   )
   
   local inputs, gradOutputs = {}, {}
   local inputs2, gradOutputs2 = {}, {}
   for i=1,nStep do
      inputs[i] = torch.randn(batchSize, hiddenSize)
      gradOutputs[i] = torch.randn(batchSize, hiddenSize)
      inputs2[i] = inputs[i]:clone()
      gradOutputs2[i] = gradOutputs[i]:clone()
   end
   
   local params, gradParams = rnn:getParameters()
   local params2, gradParams2 = rnn2:getParameters()
   
   for j=1,3 do
   
      rnn:forget()
      rnn2:forget()
   
      rnn:zeroGradParameters()
      rnn2:zeroGradParameters()
   
      -- forward
      for i=1,nStep do
         local output = rnn:forward(inputs[i])
         local output2 = rnn2:forward(inputs2[i])
         mytester:assertTensorEq(output, output2, 0.000001, "Recurrent_old output err "..i)
         rnn2:backward(inputs[i], gradOutputs2[i])
      end
      
      -- backward
      rnn2:backwardThroughTime()
      for i=nStep,1,-1 do
         local gradInput = rnn:backward(inputs[i], gradOutputs[i])
         mytester:assertTensorEq(gradInput, rnn2.gradInputs[i], 0.000001, "Recurrent_old gradInput err "..i)
      end
      
      local p1, gp1 = rnn:parameters()
      local p2, gp2 = rnn2:parameters()
      
      for i=1,#gp1 do
         mytester:assertTensorEq(gp1[i], gp2[i], 0.00000001, "Recurrent_old gradParams err "..i)
      end
      
      mytester:assertTensorEq(gradParams, gradParams2, 0.00000001, "Recurrent_old gradParams error")
      
      rnn2:updateParameters(0.1)
      rnn:updateParameters(0.1)
   
   end
   
   if not pcall(function() require 'optim' end) then return end

   local hiddenSize = 2
   local rnn = nn.Recurrent(hiddenSize, nn.Linear(hiddenSize, hiddenSize), nn.Linear(hiddenSize, hiddenSize))

   local criterion = nn.MSECriterion()
   local sequence = torch.randn(4,2)
   local s = sequence:clone()
   local parameters, grads = rnn:getParameters()
   
   function f(x)
      parameters:copy(x)
      -- Do the forward prop
      rnn:zeroGradParameters()
      assert(grads:sum() == 0)
      local err = 0
      local outputs = {}
      for i = 1, sequence:size(1) - 1 do
         local output = rnn:forward(sequence[i])
         outputs[i] = output
         err = err + criterion:forward(output, sequence[i + 1])
      end
      for i=sequence:size(1)-1,1,-1 do
         criterion:forward(outputs[i], sequence[i + 1])
         local gradOutput = criterion:backward(outputs[i], sequence[i + 1])
         rnn:backward(sequence[i], gradOutput)
      end
      rnn:forget()
      return err, grads
   end
   
   function optim.checkgrad(opfunc, x, eps)
       -- compute true gradient:
       local _,dC = opfunc(x)
       dC:resize(x:size())
       
       -- compute numeric approximations to gradient:
       local eps = eps or 1e-7
       local dC_est = torch.DoubleTensor(dC:size())
       for i = 1,dC:size(1) do
         x[i] = x[i] + eps
         local C1 = opfunc(x)
         x[i] = x[i] - 2 * eps
         local C2 = opfunc(x)
         x[i] = x[i] + eps
         dC_est[i] = (C1 - C2) / (2 * eps)
       end

       -- estimate error of gradient:
       local diff = torch.norm(dC - dC_est) / torch.norm(dC + dC_est)
       return diff,dC,dC_est
   end

   local err = optim.checkgrad(f, parameters:clone())
   mytester:assert(err < 0.0001, "Recurrent optim.checkgrad error")
end

function rnntest.Recurrent()
   local batchSize = 4
   local dictSize = 100
   local hiddenSize = 12
   local outputSize = 7
   local nStep = 5 
   local inputModule = nn.LookupTable(dictSize, outputSize)
   local transferModule = nn.Sigmoid()
   -- test MLP feedback Module (because of Module:representations())
   local feedbackModule = nn.Sequential()
   feedbackModule:add(nn.Linear(outputSize, hiddenSize))
   feedbackModule:add(nn.Sigmoid())
   feedbackModule:add(nn.Linear(hiddenSize, outputSize))
   -- rho = nStep
   local mlp = nn.Recurrent(outputSize, inputModule, feedbackModule, transferModule:clone(), nStep)
 
   local gradOutputs, outputs = {}, {}
   -- inputs = {inputN, {inputN-1, {inputN-2, ...}}}}}
   local inputs
   local startModule = mlp.startModule:clone()
   inputModule = mlp.inputModule:clone()
   feedbackModule = mlp.feedbackModule:clone()
   
   local mlp6 = mlp:clone()
   mlp6:evaluate()
   
   mlp:zeroGradParameters()
   local mlp7 = mlp:clone()
   mlp7.rho = nStep - 1
   local inputSequence, gradOutputSequence = {}, {}
   for step=1,nStep do
      local input = torch.IntTensor(batchSize):random(1,dictSize)
      inputSequence[step] = input
      local gradOutput
      if step ~= nStep then
         -- for the sake of keeping this unit test simple,
         gradOutput = torch.zeros(batchSize, outputSize)
      else
         -- only the last step will get a gradient from the output
         gradOutput = torch.randn(batchSize, outputSize)
      end
      gradOutputSequence[step] = gradOutput
      
      local output = mlp:forward(input)
      
      local output6 = mlp6:forward(input)
      mytester:assertTensorEq(output, output6, 0.000001, "evaluation error "..step)
      
      local output7 = mlp7:forward(input)
      mytester:assertTensorEq(output, output7, 0.000001, "rho = nStep-1 forward error "..step)

      table.insert(gradOutputs, gradOutput)
      table.insert(outputs, output:clone())
      
      if inputs then
         inputs = {input, inputs}
      else
         inputs = input
      end
   end

   local mlp5 = mlp:clone()
   
   -- backward propagate through time (BPTT)
   local gradInputs1 = {}
   local gradInputs7 = {}
   for step=nStep,1,-1 do
      table.insert(gradInputs1, mlp:backward(inputSequence[step], gradOutputSequence[step]))
      if step > 1 then -- rho = nStep - 1 : shouldn't update startModule
         table.insert(gradInputs7, mlp7:backward(inputSequence[step], gradOutputSequence[step]))
      end
   end
   
   
   local gradInput = gradInputs1[1]:clone()
   mlp:forget() -- test ability to forget
   mlp:zeroGradParameters()
   local foutputs = {}
   for step=1,nStep do
      foutputs[step] = mlp:forward(inputSequence[step])
      mytester:assertTensorEq(foutputs[step], outputs[step], 0.00001, "Recurrent forget output error "..step)
   end
   
   local fgradInput
   for step=nStep,1,-1 do
      fgradInput = mlp:backward(inputSequence[step], gradOutputs[step])
   end
   fgradInput = fgradInput:clone()
   mytester:assertTensorEq(gradInput, fgradInput, 0.00001, "Recurrent forget gradInput error")
   
   local mlp10 = mlp7:clone()
   mlp10:forget()
   mytester:assert(#mlp10.outputs == 0, 'forget outputs error')
   local i = 0
   for k,v in pairs(mlp10.sharedClones) do
      i = i + 1
   end
   mytester:assert(i == 4, 'forget recurrentOutputs error')
   
   local mlp2 -- this one will simulate rho = nStep
   local outputModules = {}
   for step=1,nStep do
      local inputModule_ = inputModule:sharedClone()
      local outputModule = transferModule:clone()
      table.insert(outputModules, outputModule)
      if step == 1 then
         local initialModule = nn.Sequential()
         initialModule:add(inputModule_)
         initialModule:add(startModule)
         initialModule:add(outputModule)
         mlp2 = initialModule
      else
         local parallelModule = nn.ParallelTable()
         parallelModule:add(inputModule_)
         local pastModule = nn.Sequential()
         pastModule:add(mlp2)
         local feedbackModule_ = feedbackModule:sharedClone()
         pastModule:add(feedbackModule_)
         parallelModule:add(pastModule)
         local recurrentModule = nn.Sequential()
         recurrentModule:add(parallelModule)
         recurrentModule:add(nn.CAddTable())
         recurrentModule:add(outputModule)
         mlp2 = recurrentModule
      end
   end
   
   
   local output2 = mlp2:forward(inputs)
   mlp2:zeroGradParameters()
   
   -- unlike mlp2, mlp8 will simulate rho = nStep -1
   local mlp8 = mlp2:clone() 
   local inputModule8 = mlp8.modules[1].modules[1]
   local m = mlp8.modules[1].modules[2].modules[1].modules[1].modules[2]
   m = m.modules[1].modules[1].modules[2].modules[1].modules[1].modules[2]
   local feedbackModule8 = m.modules[2]
   local startModule8 = m.modules[1].modules[2] -- before clone
   -- unshare the intialModule:
   m.modules[1] = m.modules[1]:clone()
   m.modules[2] = m.modules[2]:clone()
   mlp8:backward(inputs, gradOutputs[#gradOutputs])
   
   local gradInput2 = mlp2:backward(inputs, gradOutputs[#gradOutputs])
   for step=1,nStep-1 do
      gradInput2 = gradInput2[2]
   end   
   
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001, "recurrent gradInput")
   mytester:assertTensorEq(outputs[#outputs], output2, 0.000001, "recurrent output")
   for step=1,nStep do
      local output, outputModule = outputs[step], outputModules[step]
      mytester:assertTensorEq(output, outputModule.output, 0.000001, "recurrent output step="..step)
   end
   
   local mlp3 = nn.Sequential()
   -- contains params and grads of mlp2 (the MLP version of the Recurrent)
   mlp3:add(startModule):add(inputModule):add(feedbackModule)
   
   local params2, gradParams2 = mlp3:parameters()
   local params, gradParams = mlp:parameters()
   
   mytester:assert(_.size(params2) == _.size(params), 'missing parameters')
   mytester:assert(_.size(gradParams) == _.size(params), 'missing gradParameters')
   mytester:assert(_.size(gradParams2) == _.size(params), 'missing gradParameters2')
   
   for i,v in pairs(params) do
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.000001, 'gradParameter error ' .. i)
   end
   
   local mlp9 = nn.Sequential()
   -- contains params and grads of mlp8
   mlp9:add(startModule8):add(inputModule8):add(feedbackModule8)
   local params9, gradParams9 = mlp9:parameters()
   local params7, gradParams7 = mlp7:parameters()
   mytester:assert(#_.keys(params9) == #_.keys(params7), 'missing parameters')
   mytester:assert(#_.keys(gradParams7) == #_.keys(params7), 'missing gradParameters')
   for i,v in pairs(params7) do
      mytester:assertTensorEq(gradParams7[i], gradParams9[i], 0.00001, 'gradParameter error ' .. i)
   end
   
   mlp:updateParameters(0.1) 
   
   local params5 = mlp5:sparseParameters()
   local params = mlp:sparseParameters()
   for k,v in pairs(params) do
      if params5[k] then
         mytester:assertTensorNe(params[k], params5[k], 0.0000000001, 'backwardThroughTime error ' .. i)
      end
   end
end

function rnntest.Recurrent_oneElement()
   -- test sequence of one element
   local x = torch.rand(200)
   local target = torch.rand(2)

   local rho = 5
   local hiddenSize = 100
   -- RNN
   local r = nn.Recurrent(
     hiddenSize, nn.Linear(200,hiddenSize), 
     nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
     rho
   )

   local seq = nn.Sequential()
   seq:add(r)
   seq:add(nn.Linear(hiddenSize, 2))

   local criterion = nn.MSECriterion()

   local output = seq:forward(x)
   local err = criterion:forward(output,target)
   local gradOutput = criterion:backward(output,target)
   
   seq:backward(x,gradOutput)
   seq:updateParameters(0.01)
end

function rnntest.Recurrent_TestTable()
   -- Set up RNN where internal state is a table.
   -- Trivial example is same RNN from rnntest.Recurrent test
   -- but all layers are duplicated
   local batchSize = 4
   local inputSize = 10
   local hiddenSize = 12
   local outputSize = 7
   local nStep = 10
   local inputModule = nn.Linear(inputSize, outputSize)
   local transferModule = nn.Sigmoid()
   local learningRate = 0.1
   -- test MLP feedback Module
   local feedbackModule = nn.Sequential()
   feedbackModule:add(nn.Linear(outputSize, hiddenSize))
   feedbackModule:add(nn.Sigmoid())
   feedbackModule:add(nn.Linear(hiddenSize, outputSize))
   -- rho = nStep
   local mlp = nn.Recurrent(
      nn.ParallelTable()
         :add(nn.Add(outputSize))
         :add(nn.Add(outputSize)),
      nn.ParallelTable()
         :add(inputModule:clone())
         :add(inputModule:clone()),
      nn.ParallelTable()
         :add(feedbackModule:clone())
         :add(feedbackModule:clone()),
      nn.ParallelTable()
         :add(transferModule:clone())
         :add(transferModule:clone()),
      nStep,
      nn.ParallelTable()
         :add(nn.CAddTable())
         :add(nn.CAddTable())
   )

   local input = torch.randn(batchSize, inputSize)
   local err = torch.randn(batchSize, outputSize)
   for i=1,nStep do
      mlp:forward{input, input:clone()}
   end
   for i=nStep,1,-1 do
      mlp:backward({input, input:clone()}, {err, err:clone()})
   end
end

function rnntest.LSTM()
   local batchSize = math.random(1,2)
   local inputSize = math.random(3,4)
   local outputSize = math.random(5,6)
   local nStep = 3
   local input = {}
   local gradOutput = {}
   for step=1,nStep do
      input[step] = torch.randn(batchSize, inputSize)
      if step == nStep then
         -- for the sake of keeping this unit test simple,
         gradOutput[step] = torch.randn(batchSize, outputSize)
      else
         -- only the last step will get a gradient from the output
         gradOutput[step] = torch.zeros(batchSize, outputSize)
      end
   end
   local lstm = nn.LSTM(inputSize, outputSize)
   
   -- we will use this to build an LSTM step by step (with shared params)
   local lstmStep = lstm.recurrentModule:clone()
   
   -- forward/backward through LSTM
   local output = {}
   lstm:zeroGradParameters()
   for step=1,nStep do
      output[step] = lstm:forward(input[step])
      assert(torch.isTensor(input[step]))
   end   
   
   local gradInputs = {}
   for step=nStep,1,-1 do
      gradInputs[step] = lstm:backward(input[step], gradOutput[step], 1)
   end
   
   local gradInput = gradInputs[1]
   
   local mlp2 -- this one will simulate rho = nStep
   local inputs
   for step=1,nStep do
      -- iteratively build an LSTM out of non-recurrent components
      local lstm = lstmStep:clone()
      lstm:share(lstmStep, 'weight', 'gradWeight', 'bias', 'gradBias')
      if step == 1 then
         mlp2 = lstm
      else
         local rnn = nn.Sequential()
         local para = nn.ParallelTable()
         para:add(nn.Identity()):add(mlp2)
         rnn:add(para)
         rnn:add(nn.FlattenTable())
         rnn:add(lstm)
         mlp2 = rnn
      end
      
      -- prepare inputs for mlp2
      if inputs then
         inputs = {input[step], inputs}
      else
         inputs = {input[step], torch.zeros(batchSize, outputSize), torch.zeros(batchSize, outputSize)}
      end
   end
   mlp2:add(nn.SelectTable(1)) --just output the output (not cell)
   local output2 = mlp2:forward(inputs)
   
   mlp2:zeroGradParameters()
   local gradInput2 = mlp2:backward(inputs, gradOutput[nStep], 1) --/nStep)
   mytester:assertTensorEq(gradInput2[2][2][1], gradInput, 0.00001, "LSTM gradInput error")
   mytester:assertTensorEq(output[nStep], output2, 0.00001, "LSTM output error")
   
   local params, gradParams = lstm:parameters()
   local params2, gradParams2 = lstmStep:parameters()
   mytester:assert(#params == #params2, "LSTM parameters error "..#params.." ~= "..#params2)
   for i, gradParam in ipairs(gradParams) do
      local gradParam2 = gradParams2[i]
      mytester:assertTensorEq(gradParam, gradParam2, 0.000001, 
         "LSTM gradParam "..i.." error "..tostring(gradParam).." "..tostring(gradParam2))
   end
   
   gradParams = lstm.recursiveCopy(nil, gradParams)
   gradInput = gradInput:clone()
   mytester:assert(lstm.zeroTensor:sum() == 0, "zeroTensor error")
   lstm:forget()
   output = lstm.recursiveCopy(nil, output)
   local output3 = {}
   lstm:zeroGradParameters()
   for step=1,nStep do
      output3[step] = lstm:forward(input[step])
   end   
   
   local gradInputs3 = {}
   for step=nStep,1,-1 do
      gradInputs3[step] = lstm:updateGradInput(input[step], gradOutput[step])
      lstm:accGradParameters(input[step], gradOutput[step], 1)
   end
   local gradInput3 = gradInputs[1]
   
   mytester:assert(#output == #output3, "LSTM output size error")
   for i,output in ipairs(output) do
      mytester:assertTensorEq(output, output3[i], 0.00001, "LSTM forget (updateOutput) error "..i)
   end
   
   mytester:assertTensorEq(gradInput, gradInput3, 0.00001, "LSTM updateGradInput error")
   
   local params3, gradParams3 = lstm:parameters()
   mytester:assert(#params == #params3, "LSTM parameters error "..#params.." ~= "..#params3)
   for i, gradParam in ipairs(gradParams) do
      local gradParam3 = gradParams3[i]
      mytester:assertTensorEq(gradParam, gradParam3, 0.000001, 
         "LSTM gradParam "..i.." error "..tostring(gradParam).." "..tostring(gradParam3))
   end
end

function rnntest.FastLSTM()
   local inputSize = 100
   local batchSize = 40
   local nStep = 3
   
   local input = {}
   local gradOutput = {}
   for step=1,nStep do
      input[step] = torch.randn(batchSize, inputSize)
      gradOutput[step] = torch.randn(batchSize, inputSize)
   end
   local gradOutputClone = gradOutput[1]:clone()
   local lstm1 = nn.LSTM(inputSize, inputSize, nil, false)
   local lstm2 = nn.FastLSTM(inputSize, inputSize, nil)
   local seq1 = nn.Sequencer(lstm1)
   local seq2 = nn.Sequencer(lstm2)
   
   local output1 = seq1:forward(input)
   local gradInput1 = seq1:backward(input, gradOutput)
   mytester:assertTensorEq(gradOutput[1], gradOutputClone, 0.00001, "LSTM modified gradOutput")
   seq1:zeroGradParameters()
   seq2:zeroGradParameters()
   
   -- make them have same params
   local ig = lstm1.inputGate:parameters()
   local hg = lstm1.hiddenLayer:parameters()
   local fg = lstm1.forgetGate:parameters()
   local og = lstm1.outputGate:parameters()
   
   local i2g = lstm2.i2g:parameters()
   local o2g = lstm2.o2g:parameters()
   
   ig[1]:copy(i2g[1]:narrow(1,1,inputSize))
   ig[2]:copy(i2g[2]:narrow(1,1,inputSize))
   ig[3]:copy(o2g[1]:narrow(1,1,inputSize))
   hg[1]:copy(i2g[1]:narrow(1,inputSize+1,inputSize))
   hg[2]:copy(i2g[2]:narrow(1,inputSize+1,inputSize))
   hg[3]:copy(o2g[1]:narrow(1,inputSize+1,inputSize))
   fg[1]:copy(i2g[1]:narrow(1,inputSize*2+1,inputSize))
   fg[2]:copy(i2g[2]:narrow(1,inputSize*2+1,inputSize))
   fg[3]:copy(o2g[1]:narrow(1,inputSize*2+1,inputSize))
   og[1]:copy(i2g[1]:narrow(1,inputSize*3+1,inputSize))
   og[2]:copy(i2g[2]:narrow(1,inputSize*3+1,inputSize))
   og[3]:copy(o2g[1]:narrow(1,inputSize*3+1,inputSize))
   
   local output1 = seq1:forward(input)
   local gradInput1 = seq1:backward(input, gradOutput)
   local output2 = seq2:forward(input)
   local gradInput2 = seq2:backward(input, gradOutput)
   
   mytester:assert(#output1 == #output2 and #output1 == nStep)
   mytester:assert(#gradInput1 == #gradInput2 and #gradInput1 == nStep)
   for i=1,#output1 do
      mytester:assertTensorEq(output1[i], output2[i], 0.000001, "FastLSTM output error "..i)
      mytester:assertTensorEq(gradInput1[i], gradInput2[i], 0.000001, "FastLSTM gradInput error "..i)
   end
end
   
function rnntest.FastLSTM_nngraph()
   -- test the nngraph version of FastLSTM
   if not pcall(function() require 'nngraph' end) then
      return
   end
   
   local lstmSize = 10
   local batchSize = 4
   local nStep = 3
   
   local lstm1 = nn.FastLSTM(lstmSize) -- without nngraph
   local params1, gradParams1 = lstm1:getParameters()
   assert(torch.type(lstm1.recurrentModule) ~= 'nn.gModule')
   nn.FastLSTM.usenngraph = true
   local lstm2 = nn.FastLSTM(lstmSize) -- with nngraph
   nn.FastLSTM.usenngraph = false
   local params2, gradParams2 = lstm2:getParameters()
   assert(torch.type(lstm2.recurrentModule) == 'nn.gModule')
   
   lstm2.i2g.weight:copy(lstm1.i2g.weight)
   lstm2.i2g.bias:copy(lstm1.i2g.bias)
   lstm2.o2g.weight:copy(lstm1.o2g.weight)
   
   mytester:assertTensorEq(params1, params2, 0.00000001, "FastLSTM nngraph params init err")
   
   lstm1:zeroGradParameters()
   lstm2:zeroGradParameters()
   mytester:assertTensorEq(gradParams1, gradParams2, 0.000001, "FastLSTM nngraph zeroGradParameters err")
   
   local seq1 = nn.Sequencer(lstm1)
   local seq2 = nn.Sequencer(lstm2)
   
   local input = {}
   local gradOutput = {}
   for step=1,nStep do
      input[step] = torch.randn(batchSize, lstmSize)
      gradOutput[step] = torch.randn(batchSize, lstmSize)
   end
   
   local rm1 = lstm1.recurrentModule
   local rm2 = lstm2.recurrentModule
   
   local input_ = {input[1], torch.randn(batchSize, lstmSize), torch.randn(batchSize, lstmSize)}
   local gradOutput_ = {gradOutput[1], torch.randn(batchSize, lstmSize)}
   local output1 = rm1:forward(input_)
   local output2 = rm2:forward(input_)
   rm1:zeroGradParameters()
   rm2:zeroGradParameters()
   local gradInput1 = rm1:backward(input_, gradOutput_)
   local gradInput2 = rm2:backward(input_, gradOutput_)
   
   mytester:assertTensorEq(output1[1], output2[1], 0.0000001, "FastLSTM.recurrentModule forward 1 error")
   mytester:assertTensorEq(output1[2], output2[2], 0.0000001, "FastLSTM.recurrentModule forward 2 error")
   for i=1,3 do
      mytester:assertTensorEq(gradInput1[i], gradInput2[i], 0.0000001, "FastLSTM.recurrentModule backward err "..i)
   end
   
   mytester:assertTensorEq(gradParams1, gradParams2, 0.000001, "FastLSTM.recurrenModule nngraph gradParams err")
   
   -- again, with sharedClone
   local rm3 = lstm1.recurrentModule:sharedClone()
   local rm4 = lstm2.recurrentModule:clone()
   
   local output1 = rm3:forward(input_)
   local output2 = rm4:forward(input_)
   local gradInput1 = rm3:backward(input_, gradOutput_)
   local gradInput2 = rm4:backward(input_, gradOutput_)
   
   mytester:assertTensorEq(output1[1], output2[1], 0.0000001, "FastLSTM.recurrentModule forward 1 error")
   mytester:assertTensorEq(output1[2], output2[2], 0.0000001, "FastLSTM.recurrentModule forward 2 error")
   for i=1,3 do
      mytester:assertTensorEq(gradInput1[i], gradInput2[i], 0.0000001, "FastLSTM.recurrentModule backward err "..i)
   end
   
   local p1, gp1 = rm3:parameters()
   local p2, gp2 = rm4:parameters()
   
   for i=1,#p1 do
      mytester:assertTensorEq(gp1[i], gp2[i], 0.000001, "FastLSTM nngraph gradParam err "..i)
   end
   
   seq1:zeroGradParameters()
   seq2:zeroGradParameters()
   mytester:assertTensorEq(gradParams1, gradParams2, 0.000001, "FastLSTM nngraph zeroGradParameters err")
   mytester:assert(gradParams1:sum() == 0)
   
   local input_ = _.map(input, function(k, x) return x:clone() end)
   local gradOutput_ = _.map(gradOutput, function(k, x) return x:clone() end)
   
   -- forward/backward
   local output1 = seq1:forward(input)
   local gradInput1 = seq1:backward(input, gradOutput)
   local output2 = seq2:forward(input)
   local gradInput2 = seq2:backward(input, gradOutput)
   
   for i=1,#input do
      mytester:assertTensorEq(input[i], input_[i], 0.000001)
      mytester:assertTensorEq(gradOutput[i], gradOutput_[i], 0.000001)
   end
   
   for i=1,#output1 do
      mytester:assertTensorEq(output1[i], output2[i], 0.000001, "FastLSTM nngraph output error "..i)
      mytester:assertTensorEq(gradInput1[i], gradInput2[i], 0.000001, "FastLSTM nngraph gradInput error "..i)
   end
   
   local p1, gp1 = lstm2:parameters()
   local p2, gp2 = lstm2.sharedClones[2]:parameters()
   
   for i=1,#p1 do
      mytester:assertTensorEq(p1[i], p2[i], 0.000001, "FastLSTM nngraph param err "..i)
      mytester:assertTensorEq(gp1[i], gp2[i], 0.000001, "FastLSTM nngraph gradParam err "..i)
   end
   
   mytester:assertTensorEq(gradParams1, gradParams2, 0.000001, "FastLSTM nngraph gradParams err")
   
   if benchmark and pcall(function() require 'cunn' end ) then
      local lstmSize = 128
      local batchSize = 50
      local nStep = 50
   
      local input = {}
      local gradOutput = {}
      for step=1,nStep do
         input[step] = torch.randn(batchSize, lstmSize):cuda()
         gradOutput[step] = torch.randn(batchSize, lstmSize):cuda()
      end
      
      nn.FastLSTM.usenngraph = false
      local lstm1 = nn.Sequencer(nn.FastLSTM(lstmSize)):cuda()
      nn.FastLSTM.usenngraph = true
      local lstm2 = nn.Sequencer(nn.FastLSTM(lstmSize)):cuda()
      nn.FastLSTM.usenngraph = false
      -- nn
      
      local output = lstm1:forward(input)
      cutorch.synchronize()
      local a = torch.Timer()
      for i=1,10 do
         lstm1:forward(input)
      end
      cutorch.synchronize()
      local nntime = a:time().real
      
      -- nngraph
      
      local output = lstm2:forward(input)
      cutorch.synchronize()
      local a = torch.Timer()
      for i=1,10 do
         lstm2:forward(input)
      end
      cutorch.synchronize()
      local nngraphtime = a:time().real
      
      print("Benchmark: nn vs nngraph time", nntime, nngraphtime)
   end
end

function rnntest.GRU()
   local batchSize = math.random(1,2)
   local inputSize = math.random(3,4)
   local outputSize = math.random(5,6)
   local nStep = 3
   local input = {}
   local gradOutput = {}
   for step=1,nStep do
      input[step] = torch.randn(batchSize, inputSize)
      if step == nStep then
         -- for the sake of keeping this unit test simple,
         gradOutput[step] = torch.randn(batchSize, outputSize)
      else
         -- only the last step will get a gradient from the output
         gradOutput[step] = torch.zeros(batchSize, outputSize)
      end
   end
   local gru = nn.GRU(inputSize, outputSize):maskZero(1) -- issue 145
   
   -- we will use this to build an GRU step by step (with shared params)
   local gruStep = gru.recurrentModule:clone()
   
   -- forward/backward through GRU
   local output = {}
   gru:zeroGradParameters()
   for step=1,nStep do
      output[step] = gru:forward(input[step])
      assert(torch.isTensor(input[step]))
   end  
   local gradInput 
   for step=nStep,1,-1 do
      gradInput = gru:backward(input[step], gradOutput[step], 1)
   end
   
   local mlp2 -- this one will simulate rho = nStep
   local inputs
   for step=1,nStep do
      -- iteratively build an GRU out of non-recurrent components
      local gru = gruStep:clone()
      gru:share(gruStep, 'weight', 'gradWeight', 'bias', 'gradBias')
      if step == 1 then
         mlp2 = gru
      else
         local rnn = nn.Sequential()
         local para = nn.ParallelTable()
         para:add(nn.Identity()):add(mlp2)
         rnn:add(para)
         rnn:add(nn.FlattenTable())
         rnn:add(gru)
         mlp2 = rnn
      end
      
      -- prepare inputs for mlp2
      if inputs then
         inputs = {input[step], inputs}
      else
         inputs = {input[step], torch.zeros(batchSize, outputSize)}
      end
   end
   local output2 = mlp2:forward(inputs)
   
   mlp2:zeroGradParameters()
   local gradInput2 = mlp2:backward(inputs, gradOutput[nStep], 1) --/nStep)
   mytester:assertTensorEq(gradInput2[2][2][1], gradInput, 0.00001, "GRU gradInput error")
   mytester:assertTensorEq(output[nStep], output2, 0.00001, "GRU output error")
   
   local params, gradParams = gru:parameters()
   local params2, gradParams2 = gruStep:parameters()
   mytester:assert(#params == #params2, "GRU parameters error "..#params.." ~= "..#params2)
   for i, gradParam in ipairs(gradParams) do
      local gradParam2 = gradParams2[i]
      mytester:assertTensorEq(gradParam, gradParam2, 0.000001, 
         "gru gradParam "..i.." error "..tostring(gradParam).." "..tostring(gradParam2))
   end
   
   gradParams = gru.recursiveCopy(nil, gradParams)
   gradInput = gradInput:clone()
   mytester:assert(gru.zeroTensor:sum() == 0, "zeroTensor error")
   gru:forget()
   output = gru.recursiveCopy(nil, output)
   local output3 = {}
   gru:zeroGradParameters()
   for step=1,nStep do
      output3[step] = gru:forward(input[step])
   end   
   
   local gradInput3
   for step=nStep,1,-1 do
      gradInput3 = gru:backward(input[step], gradOutput[step], 1)
   end
   
   mytester:assert(#output == #output3, "GRU output size error")
   for i,output in ipairs(output) do
      mytester:assertTensorEq(output, output3[i], 0.00001, "GRU forget (updateOutput) error "..i)
   end
   
   mytester:assertTensorEq(gradInput, gradInput3, 0.00001, "GRU updateGradInput error")
   
   local params3, gradParams3 = gru:parameters()
   mytester:assert(#params == #params3, "GRU parameters error "..#params.." ~= "..#params3)
   for i, gradParam in ipairs(gradParams) do
      local gradParam3 = gradParams3[i]
      mytester:assertTensorEq(gradParam, gradParam3, 0.000001, 
         "GRU gradParam "..i.." error "..tostring(gradParam).." "..tostring(gradParam3))
   end
end

function rnntest.Sequencer()
   local batchSize = 4
   local inputSize = 3
   local outputSize = 7
   local nStep = 5 
   
   -- test with recurrent module
   local inputModule = nn.Linear(inputSize, outputSize)
   local transferModule = nn.Sigmoid()
   -- test MLP feedback Module (because of Module:representations())
   local feedbackModule = nn.Euclidean(outputSize, outputSize)
   -- rho = nStep
   local rnn = nn.Recurrent(outputSize, inputModule, feedbackModule, transferModule, nStep)
   rnn:zeroGradParameters()
   local rnn2 = rnn:clone()
   
   local inputs, outputs, gradOutputs = {}, {}, {}
   for step=1,nStep do
      inputs[step] = torch.randn(batchSize, inputSize)
      outputs[step] = rnn:forward(inputs[step]):clone()
      gradOutputs[step] = torch.randn(batchSize, outputSize)
   end
   
   local gradInputs = {}
   for step=nStep,1,-1 do
      gradInputs[step] = rnn:backward(inputs[step], gradOutputs[step])
   end
   
   local gradOutput1 = gradOutputs[1]:clone()
   local rnn3 = nn.Sequencer(rnn2)
   local outputs3 = rnn3:forward(inputs)
   mytester:assert(#outputs3 == #outputs, "Sequencer output size err")
   for step,output in ipairs(outputs) do
      mytester:assertTensorEq(outputs3[step], output, 0.00001, "Sequencer output "..step)
   end
   local gradInputs3 = rnn3:backward(inputs, gradOutputs)
  
   mytester:assert(#gradInputs3 == #gradInputs, "Sequencer gradInputs size err")
   mytester:assert(gradInputs3[1]:nElement() ~= 0) 
   
   for step,output in ipairs(outputs) do
      mytester:assertTensorEq(gradInputs3[step], gradInputs[step], 0.00001, "Sequencer gradInputs "..step)
   end
   mytester:assertTensorEq(gradOutputs[1], gradOutput1, 0.00001, "Sequencer rnn gradOutput modified error")
   
   local nStep7 = torch.Tensor{5,4,5,3,7,3,3,3}
   local function testRemember(rnn)
      rnn:zeroGradParameters()
      -- test remember for training mode (with variable length)
      local rnn7 = rnn:clone()
      rnn7:zeroGradParameters()
      local rnn8 = rnn7:clone()
      local rnn9 = rnn7:clone()
      local rnn10 = nn.Recursor(rnn7:clone())
      
      local inputs7, outputs9 = {}, {}
      for step=1,nStep7:sum() do
         inputs7[step] = torch.randn(batchSize, outputSize)
         outputs9[step] = rnn9:forward(inputs7[step]):clone()
      end
      
      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember forward err "..step)
            step = step + 1
         end
      end
      
      rnn7:forget()
      
      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember forward2 err "..step)
            step = step + 1
         end
      end
      
      rnn7:forget()
      
      local step = 1
      local outputs7, gradOutputs7, gradInputs7 = {}, {}, {}
      for i=1,nStep7:size(1) do
         -- forward
         for j=1,nStep7[i] do
            outputs7[step] = rnn7:forward(inputs7[step]):clone()
            gradOutputs7[step] = torch.randn(batchSize, outputSize)
            step = step + 1
         end
         -- backward
         rnn7:maxBPTTstep(nStep7[i])
         for _step=step-1,step-nStep7[i],-1 do
            gradInputs7[_step] = rnn7:backward(inputs7[_step], gradOutputs7[_step]):clone()
         end
         -- update
         rnn7:updateParameters(1)
         rnn7:zeroGradParameters()
      end
      
      -- nn.Recursor 
      
      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn10:forward(inputs7[step]), 0.000001, "Recursor "..torch.type(rnn10).." remember forward err "..step)
            step = step + 1
         end
      end
      
      rnn10:forget()
      
      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn10:forward(inputs7[step]), 0.000001, "Recursor "..torch.type(rnn10).." remember forward2 err "..step)
            step = step + 1
         end
      end
      
      rnn10:forget()
      
      local step = 1
      local outputs10, gradOutputs10, gradInputs10 = {}, {}, {}
      for i=1,nStep7:size(1) do
         local start = step
         local nStep = 0
         for j=1,nStep7[i] do
            outputs10[step] = rnn10:forward(inputs7[step]):clone()
            step = step + 1
            nStep = nStep + 1
         end
         rnn10:maxBPTTstep(nStep7[i])
         local nStep2 = 0
         for s=step-1,start,-1 do
            gradInputs10[s] = rnn10:backward(inputs7[s], gradOutputs7[s]):clone()
            nStep2 = nStep2 + 1
         end
         assert(nStep == nStep2)
         rnn10:updateParameters(1)
         rnn10:zeroGradParameters()
      end
      
      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(gradInputs10[step], gradInputs7[step], 0.0000001, "Recursor "..torch.type(rnn7).." remember variable backward err "..i.." "..j)
            mytester:assertTensorEq(outputs10[step], outputs7[step], 0.0000001, "Recursor "..torch.type(rnn7).." remember variable forward err "..i.." "..j)
            step = step + 1
         end
      end
      
      -- nn.Sequencer
      
      local seq = nn.Sequencer(rnn8)
      seq:remember('both')
      local outputs8, gradInputs8 = {}, {}
      local step = 1
      for i=1,nStep7:size(1) do
         local inputs8 = _.slice(inputs7,step,step+nStep7[i]-1)
         local gradOutputs8 = _.slice(gradOutputs7,step,step+nStep7[i]-1)
         outputs8[i] = _.map(seq:forward(inputs8), function(k,v) return v:clone() end)
         gradInputs8[i] = _.map(seq:backward(inputs8, gradOutputs8), function(k,v) return v:clone() end)
         seq:updateParameters(1)
         seq:zeroGradParameters()
         step = step + nStep7[i]
      end
      
      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(gradInputs8[i][j], gradInputs7[step], 0.0000001, "Sequencer "..torch.type(rnn7).." remember variable backward err "..i.." "..j)
            mytester:assertTensorEq(outputs8[i][j], outputs7[step], 0.0000001, "Sequencer "..torch.type(rnn7).." remember variable forward err "..i.." "..j)
            step = step + 1
         end
      end
      
      local params7 = rnn7:parameters()
      local params8 = rnn8:parameters()
      for i=1,#params7 do
         mytester:assertTensorEq(params7[i], params8[i], 0.0000001, "Sequencer "..torch.type(rnn7).." remember params err "..i)
      end
      
      -- test in evaluation mode with remember and variable rho
      local rnn7 = rnn:clone() -- a fresh copy (no hidden states)
      local params7 = rnn7:parameters()
      local params9 = rnn9:parameters() -- not a fresh copy
      for i,param in ipairs(rnn8:parameters()) do
         params7[i]:copy(param)
         params9[i]:copy(param)
      end
      
      rnn7:evaluate()
      rnn9:evaluate()
      rnn9:forget()
      
      local inputs7, outputs9 = {}, {}
      for step=1,nStep7:sum() do
         inputs7[step] = torch.randn(batchSize, outputSize)
         outputs9[step] = rnn9:forward(inputs7[step]):clone()
      end
      
      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember eval forward err "..step)
            step = step + 1
         end
      end
      
      rnn7:forget()
      
      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember eval forward2 err "..step)
            step = step + 1
         end
      end
      
      rnn7:forget()
      
      local step = 1
      local outputs7 = {}
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            outputs7[step] = rnn7:forward(inputs7[step]):clone()
            step = step + 1
         end
      end
      
      seq:remember('both')
      local outputs8 = {}
      local step = 1
      for i=1,nStep7:size(1) do
         seq:evaluate()
         local inputs8 = _.slice(inputs7,step,step+nStep7[i]-1)
         local gradOutputs8 = _.slice(gradOutputs7,step,step+nStep7[i]-1)
         outputs8[i] = _.map(seq:forward(inputs8), function(k,v) return v:clone() end)
         step = step + nStep7[i]
      end
      
      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs8[i][j], outputs7[step], 0.0000001, "Sequencer "..torch.type(rnn7).." remember variable eval forward err "..i.." "..j)
            step = step + 1
         end
      end
      
      -- test remember for training mode (with variable length) (from evaluation to training)
      
      rnn7:forget()
      rnn9:forget()
      
      rnn7:training()
      rnn9:training()
      
      rnn7:zeroGradParameters()
      seq:zeroGradParameters()
      rnn9:zeroGradParameters()
      
      local inputs7, outputs9 = {}, {}
      for step=1,nStep7:sum() do
         inputs7[step] = torch.randn(batchSize, outputSize)
         outputs9[step] = rnn9:forward(inputs7[step]):clone()
      end
      
      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember forward err "..step)
            step = step + 1
         end
      end
      
      rnn7:forget()
      
      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(outputs9[step], rnn7:forward(inputs7[step]), 0.000001, "Sequencer "..torch.type(rnn7).." remember forward2 err "..step)
            step = step + 1
         end
      end
      
      rnn7:forget()
      
      local step = 1
      local outputs7, gradOutputs7, gradInputs7 = {}, {}, {}      
      for i=1,nStep7:size(1) do
         -- forward
         for j=1,nStep7[i] do
            outputs7[step] = rnn7:forward(inputs7[step]):clone()
            gradOutputs7[step] = torch.randn(batchSize, outputSize)
            step = step + 1
         end
         -- backward
         rnn7:maxBPTTstep(nStep7[i])
         for _step=step-1,step-nStep7[i],-1 do
            gradInputs7[_step] = rnn7:backward(inputs7[_step], gradOutputs7[_step]):clone()
         end
         -- update
         rnn7:updateParameters(1)
         rnn7:zeroGradParameters()
      end
      
      seq:remember('both')
      local outputs8, gradInputs8 = {}, {}
      local step = 1
      for i=1,nStep7:size(1) do
         seq:training()
         local inputs8 = _.slice(inputs7,step,step+nStep7[i]-1)
         local gradOutputs8 = _.slice(gradOutputs7,step,step+nStep7[i]-1)
         outputs8[i] = _.map(seq:forward(inputs8), function(k,v) return v:clone() end)
         gradInputs8[i] = _.map(seq:backward(inputs8, gradOutputs8), function(k,v) return v:clone() end)
         seq:updateParameters(1)
         seq:zeroGradParameters()
         step = step + nStep7[i]
      end
      
      local step = 1
      for i=1,nStep7:size(1) do
         for j=1,nStep7[i] do
            mytester:assertTensorEq(gradInputs8[i][j], gradInputs7[step], 0.0000001, "Sequencer "..torch.type(rnn7).." remember variable backward err "..i.." "..j)
            mytester:assertTensorEq(outputs8[i][j], outputs7[step], 0.0000001, "Sequencer "..torch.type(rnn7).." remember variable forward err "..i.." "..j)
            step = step + 1
         end
      end
      
      local params7 = rnn7:parameters()
      local params8 = rnn8:parameters()
      for i=1,#params7 do
         mytester:assertTensorEq(params7[i], params8[i], 0.0000001, "Sequencer "..torch.type(rnn7).." remember params err "..i)
      end
   end
   testRemember(nn.Recurrent(outputSize, nn.Linear(outputSize, outputSize), feedbackModule:clone(), transferModule:clone(), nStep7:max()))
   testRemember(nn.LSTM(outputSize, outputSize, nStep7:max()))

   -- test in evaluation mode
   rnn3:evaluate()
   local outputs4 = rnn3:forward(inputs)
   local outputs4_ = _.map(outputs4, function(k,v) return v:clone() end)
   mytester:assert(#outputs4 == #outputs, "Sequencer evaluate output size err")
   for step,output in ipairs(outputs) do
      mytester:assertTensorEq(outputs4[step], output, 0.00001, "Sequencer evaluate output "..step)
   end
   local inputs5 = _.clone(inputs)
   table.remove(inputs5, nStep) -- remove last input
   local outputs5 = rnn3:forward(inputs5)
   mytester:assert(#outputs5 == #outputs - 1, "Sequencer evaluate -1 output size err")
   for step,output in ipairs(outputs5) do
      mytester:assertTensorEq(outputs[step], output, 0.00001, "Sequencer evaluate -1 output "..step)
   end
   
   -- test evaluation with remember 
   rnn3:remember()
   rnn3:evaluate()
   rnn3:forget()
   local inputsA, inputsB = {inputs[1],inputs[2],inputs[3]}, {inputs[4],inputs[5]}
   local outputsA = _.map(rnn3:forward(inputsA), function(k,v) return v:clone() end)
   local outputsB = rnn3:forward(inputsB)
   mytester:assert(#outputsA == 3, "Sequencer evaluate-remember output size err A")
   mytester:assert(#outputsB == 2, "Sequencer evaluate-remember output size err B")
   local outputsAB = {unpack(outputsA)}
   outputsAB[4], outputsAB[5] = unpack(outputsB)
   for step,output in ipairs(outputs4_) do
      mytester:assertTensorEq(outputsAB[step], output, 0.00001, "Sequencer evaluate-remember output "..step)
   end
   
   -- test with non-recurrent module
   local inputSize = 10
   local inputs = {}
   for step=1,nStep do
      inputs[step] = torch.randn(batchSize, inputSize)
   end
   local linear = nn.Euclidean(inputSize, outputSize)
   local seq, outputs, gradInputs
   for k=1,3 do
      outputs, gradInputs = {}, {}
      linear:zeroGradParameters()
      local clone = linear:clone()
      for step=1,nStep do
         outputs[step] = linear:forward(inputs[step]):clone()
         gradInputs[step] = linear:backward(inputs[step], gradOutputs[step]):clone()
      end
      
      seq = nn.Sequencer(clone)
      local outputs2 = seq:forward(inputs)
      local gradInputs2 = seq:backward(inputs, gradOutputs)
      
      mytester:assert(#outputs2 == #outputs, "Sequencer output size err")
      mytester:assert(#gradInputs2 == #gradInputs, "Sequencer gradInputs size err")
      for step,output in ipairs(outputs) do
         mytester:assertTensorEq(outputs2[step], output, 0.00001, "Sequencer output "..step)
         mytester:assertTensorEq(gradInputs2[step], gradInputs[step], 0.00001, "Sequencer gradInputs "..step)
      end
   end
   
   local inputs3, gradOutputs3 = {}, {}
   for i=1,#inputs do
      inputs3[i] = inputs[i]:float()
      gradOutputs3[i] = gradOutputs[i]:float()
   end
   local seq3 = seq:float()
   local outputs3 = seq:forward(inputs3)
   local gradInputs3 = seq:backward(inputs3, gradOutputs3)
   
   -- test for zeroGradParameters
   local seq = nn.Sequencer(nn.Linear(inputSize,outputSize))
   seq:zeroGradParameters()
   seq:forward(inputs)
   seq:backward(inputs, gradOutputs)
   local params, gradParams = seq:parameters()
   for i,gradParam in ipairs(gradParams) do
      mytester:assert(gradParam:sum() ~= 0, "Sequencer:backward err "..i)
   end
   local param, gradParam = seq:getParameters()
   seq:zeroGradParameters()
   mytester:assert(gradParam:sum() == 0, "Sequencer:getParameters err")
   local params, gradParams = seq:parameters()
   for i,gradParam in ipairs(gradParams) do
      mytester:assert(gradParam:sum() == 0, "Sequencer:zeroGradParameters err "..i)
   end
   
   -- test with LSTM
   local outputSize = inputSize
   local lstm = nn.LSTM(inputSize, outputSize, nil, false)
   lstm:zeroGradParameters()
   local lstm2 = lstm:clone()
   
   local inputs, outputs, gradOutputs = {}, {}, {}
   for step=1,nStep do
      inputs[step] = torch.randn(batchSize, inputSize)
      gradOutputs[step] = torch.randn(batchSize, outputSize)
   end
   local gradOutput1 = gradOutputs[2]:clone()
   for step=1,nStep do
      outputs[step] = lstm:forward(inputs[step])
   end
   
   local gradInputs72 = {}
   for step=nStep,1,-1 do
      gradInputs72[step] = lstm:backward(inputs[step], gradOutputs[step])
   end
   
   local lstm3 = nn.Sequencer(lstm2)
   lstm3:zeroGradParameters()
   local outputs3 = lstm3:forward(inputs)
   local gradInputs3 = lstm3:backward(inputs, gradOutputs)
   mytester:assert(#outputs3 == #outputs, "Sequencer LSTM output size err")
   mytester:assert(#gradInputs3 == #gradInputs72, "Sequencer LSTM gradInputs size err")
   for step,output in ipairs(outputs) do
      mytester:assertTensorEq(outputs3[step], output, 0.00001, "Sequencer LSTM output "..step)
      mytester:assertTensorEq(gradInputs3[step], gradInputs72[step], 0.00001, "Sequencer LSTM gradInputs "..step)
   end
   mytester:assertTensorEq(gradOutputs[2], gradOutput1, 0.00001, "Sequencer lstm gradOutput modified error")
   
   -- test remember modes : 'both', 'eval' for training(), evaluate(), training()
   local lstm = nn.LSTM(5,5)
   local seq = nn.Sequencer(lstm)
   local inputTrain = {torch.randn(5), torch.randn(5), torch.randn(5)}
   local inputEval = {torch.randn(5)}

   -- this shouldn't fail
   local modes = {'both', 'eval'}
   for i, mode in ipairs(modes) do
     seq:remember(mode)

     -- do one epoch of training
     seq:training()
     seq:forward(inputTrain)
     seq:backward(inputTrain, inputTrain)

     -- evaluate
     seq:evaluate()
     seq:forward(inputEval)

     -- do another epoch of training
     seq:training()
     seq:forward(inputTrain)
     seq:backward(inputTrain, inputTrain)
   end
end

function rnntest.BiSequencer()
   local hiddenSize = 8
   local batchSize = 4
   local nStep = 3
   local fwd = nn.LSTM(hiddenSize, hiddenSize)
   local bwd = nn.LSTM(hiddenSize, hiddenSize)
   fwd:zeroGradParameters()
   bwd:zeroGradParameters()
   local brnn = nn.BiSequencer(fwd:clone(), bwd:clone())
   local inputs, gradOutputs = {}, {}
   for i=1,nStep do
      inputs[i] = torch.randn(batchSize, hiddenSize)
      gradOutputs[i] = torch.randn(batchSize, hiddenSize*2)
   end
   local outputs = brnn:forward(inputs)
   local gradInputs = brnn:backward(inputs, gradOutputs)
   mytester:assert(#inputs == #outputs, "BiSequencer #outputs error")
   mytester:assert(#inputs == #gradInputs, "BiSequencer #outputs error")
   
   -- forward
   local fwdSeq = nn.Sequencer(fwd)
   local bwdSeq = nn.Sequencer(bwd)
   local zip, join = nn.ZipTable(), nn.Sequencer(nn.JoinTable(1,1))
   local fwdOutputs = fwdSeq:forward(inputs)
   local bwdOutputs = _.reverse(bwdSeq:forward(_.reverse(inputs)))
   local zipOutputs = zip:forward{fwdOutputs, bwdOutputs}
   local outputs2 = join:forward(zipOutputs)
   for i,output in ipairs(outputs) do
      mytester:assertTensorEq(output, outputs2[i], 0.000001, "BiSequencer output err "..i)
   end
   
   -- backward
   local joinGradInputs = join:backward(zipOutputs, gradOutputs)
   local zipGradInputs = zip:backward({fwdOutputs, bwdOutputs}, joinGradInputs)
   local bwdGradInputs = _.reverse(bwdSeq:backward(_.reverse(inputs), _.reverse(zipGradInputs[2])))
   local fwdGradInputs = fwdSeq:backward(inputs, zipGradInputs[1])
   local gradInputs2 = zip:forward{fwdGradInputs, bwdGradInputs}
   for i,gradInput in ipairs(gradInputs) do
      local gradInput2 = gradInputs2[i]
      gradInput2[1]:add(gradInput2[2])
      mytester:assertTensorEq(gradInput, gradInput2[1], 0.000001, "BiSequencer gradInput err "..i)
   end
   
   -- params
   local brnn2 = nn.Sequential():add(fwd):add(bwd)
   local params, gradParams = brnn:parameters()
   local params2, gradParams2 = brnn2:parameters()
   mytester:assert(#params == #params2, "BiSequencer #params err")
   mytester:assert(#params == #gradParams, "BiSequencer #gradParams err")
   for i,param in pairs(params) do
      mytester:assertTensorEq(param, params2[i], 0.000001, "BiSequencer params err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.000001, "BiSequencer gradParams err "..i)
   end
   
   -- updateParameters
   brnn:updateParameters(0.1)
   brnn2:updateParameters(0.1)
   brnn:zeroGradParameters()
   brnn2:zeroGradParameters()
   for i,param in pairs(params) do
      mytester:assertTensorEq(param, params2[i], 0.000001, "BiSequencer params update err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.000001, "BiSequencer gradParams zero err "..i)
   end
end

function rnntest.BiSequencerLM()
   local hiddenSize = 8
   local batchSize = 4
   local nStep = 3
   local fwd = nn.LSTM(hiddenSize, hiddenSize)
   local bwd = nn.LSTM(hiddenSize, hiddenSize)
   fwd:zeroGradParameters()
   bwd:zeroGradParameters()
   local brnn = nn.BiSequencerLM(fwd:clone(), bwd:clone())
   local inputs, gradOutputs = {}, {}
   for i=1,nStep do
      inputs[i] = torch.randn(batchSize, hiddenSize)
      gradOutputs[i] = torch.randn(batchSize, hiddenSize*2)
   end
   local outputs = brnn:forward(inputs)
   local gradInputs = brnn:backward(inputs, gradOutputs)
   mytester:assert(#inputs == #outputs, "BiSequencerLM #outputs error")
   mytester:assert(#inputs == #gradInputs, "BiSequencerLM #outputs error")
   
   -- forward
   local fwdSeq = nn.Sequencer(fwd)
   local bwdSeq = nn.Sequencer(bwd)
   local merge = nn.Sequential():add(nn.ZipTable()):add(nn.Sequencer(nn.JoinTable(1,1)))
   
   local fwdOutputs = fwdSeq:forward(_.first(inputs, #inputs-1))
   local bwdOutputs = _.reverse(bwdSeq:forward(_.reverse(_.last(inputs, #inputs-1))))
   
   local fwdMergeInputs = _.clone(fwdOutputs)
   table.insert(fwdMergeInputs, 1, fwdOutputs[1]:clone():zero())
   local bwdMergeInputs = _.clone(bwdOutputs)
   table.insert(bwdMergeInputs, bwdOutputs[1]:clone():zero())
   
   local outputs2 = merge:forward{fwdMergeInputs, bwdMergeInputs}
   
   for i,output in ipairs(outputs) do
      mytester:assertTensorEq(output, outputs2[i], 0.000001, "BiSequencerLM output err "..i)
   end
   
   -- backward
   local mergeGradInputs = merge:backward({fwdMergeInputs, bwdMergeInputs}, gradOutputs)
   
   local bwdGradInputs = _.reverse(bwdSeq:backward(_.reverse(_.last(inputs, #inputs-1)), _.reverse(_.first(mergeGradInputs[2], #inputs-1))))
   local fwdGradInputs = fwdSeq:backward(_.first(inputs, #inputs-1), _.last(mergeGradInputs[1], #inputs-1))
   
   for i,gradInput in ipairs(gradInputs) do
      local gradInput2
      if i == 1 then
         gradInput2 = fwdGradInputs[1]
      elseif i == #inputs then
         gradInput2 = bwdGradInputs[#inputs-1]
      else
         gradInput2 = fwdGradInputs[i]:clone()
         gradInput2:add(bwdGradInputs[i-1])
      end
      mytester:assertTensorEq(gradInput, gradInput2, 0.000001, "BiSequencerLM gradInput err "..i)
   end
   
   -- params
   local brnn2 = nn.Sequential():add(fwd):add(bwd)
   local params, gradParams = brnn:parameters()
   local params2, gradParams2 = brnn2:parameters()
   mytester:assert(#params == #params2, "BiSequencerLM #params err")
   mytester:assert(#params == #gradParams, "BiSequencerLM #gradParams err")
   for i,param in pairs(params) do
      mytester:assertTensorEq(param, params2[i], 0.000001, "BiSequencerLM params err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.000001, "BiSequencerLM gradParams err "..i)
   end
   
   -- updateParameters
   brnn:updateParameters(0.1)
   brnn2:updateParameters(0.1)
   brnn:zeroGradParameters()
   brnn2:zeroGradParameters()
   for i,param in pairs(params) do
      mytester:assertTensorEq(param, params2[i], 0.000001, "BiSequencerLM params update err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.000001, "BiSequencerLM gradParams zero err "..i)
   end
end

function rnntest.Repeater()
   local batchSize = 4
   local inputSize = 10
   local outputSize = 7
   local nStep = 5 
   local inputModule = nn.Linear(inputSize, outputSize)
   local transferModule = nn.Sigmoid()
   -- test MLP feedback Module (because of Module:representations())
   local feedbackModule = nn.Linear(outputSize, outputSize)
   -- rho = nStep
   local rnn = nn.Recurrent(outputSize, inputModule, feedbackModule, transferModule, nStep)
   local rnn2 = rnn:clone()
   
   local inputs, outputs, gradOutputs = {}, {}, {}
   local input = torch.randn(batchSize, inputSize)
   for step=1,nStep do
      outputs[step] = rnn:forward(input)
      gradOutputs[step] = torch.randn(batchSize, outputSize)
   end
   local gradInputs = {}
   for step=nStep,1,-1 do
      gradInputs[step] = rnn:backward(input, gradOutputs[step])
   end
   
   local rnn3 = nn.Repeater(rnn2, nStep)
   local outputs3 = rnn3:forward(input)
   local gradInput3 = rnn3:backward(input, gradOutputs)
   mytester:assert(#outputs3 == #outputs, "Repeater output size err")
   mytester:assert(#outputs3 == #gradInputs, "Repeater gradInputs size err")
   local gradInput = gradInputs[1]:clone():zero()
   for step,output in ipairs(outputs) do
      mytester:assertTensorEq(outputs3[step], output, 0.00001, "Repeater output "..step)
      gradInput:add(gradInputs[step])
   end
   mytester:assertTensorEq(gradInput3, gradInput, 0.00001, "Repeater gradInput err")
   
   -- test with Recursor
   
   local inputModule = nn.Linear(inputSize, outputSize)
   local transferModule = nn.Sigmoid()
   -- test MLP feedback Module (because of Module:representations())
   local feedbackModule = nn.Linear(outputSize, outputSize)
   -- rho = nStep
   local rnn = nn.Recurrent(outputSize, inputModule, feedbackModule, transferModule, nStep)
   local rnn2 = rnn:clone()
   
   local rnn3 = nn.Repeater(rnn, nStep)
   local rnn4 = nn.Repeater(nn.Sequential():add(nn.Identity()):add(rnn2), nStep)
   
   rnn3:zeroGradParameters()
   rnn4:zeroGradParameters()
   
   local outputs = rnn3:forward(input)
   local outputs2 = rnn4:forward(input)
   
   local gradInput = rnn3:backward(input, gradOutputs)
   local gradInput2 = rnn4:backward(input, gradOutputs)
   
   mytester:assert(#outputs == #outputs2, "Repeater output size err")
   for i=1,#outputs do
      mytester:assertTensorEq(outputs[i], outputs2[i], 0.0000001, "Repeater(Recursor) output err")
   end
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001, "Repeater(Recursor) gradInput err")
   
   rnn3:updateParameters(1)
   rnn4:updateParameters(1)
   
   local params, gradParams = rnn3:parameters()
   local params2, gradParams2 = rnn4:parameters()
   
   for i=1,#params do
      mytester:assertTensorEq(params[i], params2[i], 0.0000001, "Repeater(Recursor) param err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001, "Repeater(Recursor) gradParam err "..i)
   end
end

function rnntest.SequencerCriterion()
   local batchSize = 4
   local inputSize = 10
   local outputSize = 7
   local nStep = 5  
   -- https://github.com/Element-Research/rnn/issues/128
   local criterion = nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1)
   local sc = nn.SequencerCriterion(criterion:clone())
   local input = {}
   local target = {}
   local err2 = 0
   local gradInput2 = {}
   for i=1,nStep do
      input[i] = torch.randn(batchSize, inputSize)
      target[i] = torch.randperm(inputSize):narrow(1,1,batchSize)
      err2 = err2 + criterion:forward(input[i], target[i])
      gradInput2[i] = criterion:backward(input[i], target[i]):clone()
   end
   local err = sc:forward(input, target)
   mytester:assert(math.abs(err-err2) < 0.000001, "SequencerCriterion forward err")
   local gradInput = sc:backward(input, target)
   for i=1,nStep do
      mytester:assertTensorEq(gradInput[i], gradInput2[i], 0.000001, "SequencerCriterion backward err "..i)
   end
   
   -- test type()
   sc.gradInput = {}
   sc:float()
   
   for i=1,nStep do
      input[i] = input[i]:float()
      target[i] = target[i]:float()
   end
   
   local err3 = sc:forward(input, target)
   mytester:assert(math.abs(err - err3) < 0.000001, "SequencerCriterion forward type err") 
   local gradInput3 = sc:backward(input, target)
   for i=1,nStep do
      mytester:assertTensorEq(gradInput[i]:float(), gradInput3[i], 0.000001, "SequencerCriterion backward type err "..i)
   end
   
   if pcall(function() require 'cunn' end) then
      -- test cuda()
      sc.gradInput = {}
      sc:cuda()
   
      local gradInput4 = {}
      for i=1,nStep do
         input[i] = input[i]:cuda()
         target[i] = target[i]:cuda()
      end
      
      local err4 = sc:forward(input, target)
      mytester:assert(math.abs(err - err4) < 0.000001, "SequencerCriterion forward cuda err") 
      local gradInput4 = sc:backward(input, target)
      for i=1,nStep do
         mytester:assertTensorEq(gradInput4[i]:float(), gradInput3[i], 0.000001, "SequencerCriterion backward cuda err "..i)
      end
   end
end

function rnntest.RepeaterCriterion()
   local batchSize = 4
   local inputSize = 10
   local outputSize = 7
   local nStep = 5  
   local criterion = nn.ClassNLLCriterion()
   local sc = nn.RepeaterCriterion(criterion:clone())
   local input = {}
   local target = torch.randperm(inputSize):narrow(1,1,batchSize)
   local err2 = 0
   local gradInput2 = {}
   for i=1,nStep do
      input[i] = torch.randn(batchSize, inputSize)
      err2 = err2 + criterion:forward(input[i], target)
      gradInput2[i] = criterion:backward(input[i], target):clone()
   end
   local err = sc:forward(input, target)
   mytester:assert(math.abs(err-err2) < 0.000001, "RepeaterCriterion forward err") 
   local gradInput = sc:backward(input, target)
   for i=1,nStep do
      mytester:assertTensorEq(gradInput[i], gradInput2[i], 0.000001, "RepeaterCriterion backward err "..i)
   end
   
   -- test type()
   sc:float()
   
   local gradInput3 = {}
   target = target:float()
   for i=1,nStep do
      input[i] = input[i]:float()
   end
   
   local err3 = sc:forward(input, target)
   mytester:assert(math.abs(err - err3) < 0.000001, "RepeaterCriterion forward type err") 
   local gradInput3 = sc:backward(input, target)
   for i=1,nStep do
      mytester:assertTensorEq(gradInput[i]:float(), gradInput3[i], 0.000001, "RepeaterCriterion backward type err "..i)
   end
end

function rnntest.RecurrentAttention()
   -- so basically, I know that this works because I used it to 
   -- reproduce a paper's results. So all future RecurrentAttention
   -- versions should match the behavior of this RATest class.
   -- Yeah, its ugly, but it's a unit test, so kind of hidden :
   local RecurrentAttention, parent = torch.class("nn.RATest", "nn.AbstractSequencer")

   function RecurrentAttention:__init(rnn, action, nStep, hiddenSize)
      parent.__init(self)
      assert(torch.isTypeOf(rnn, 'nn.ARTest'))
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
      self.sharedClones = {self.action:sharedClone()} -- action clones
      
      self.output = {} -- rnn output
      self.actions = {} -- action output
      
      self.forwardActions = false
      
      self.gradHidden = {}
   end
   
   function RecurrentAttention:getStepModule(step)
      assert(self.sharedClones, "no sharedClones for type "..torch.type(self))
      assert(step, "expecting step at arg 1")
      local module = self.sharedClones[step]
      if not module then
         module = self.sharedClones[1]:sharedClone()
         self.sharedClones[step] = module
      end
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
         
         local gradOutput_, gradAction_ = gradOutput[step], action.output:clone():zero()
         if self.forwardActions then
            gradOutput_, gradAction_ = unpack(gradOutput[step])
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
            local gradAction = action:updateGradInput(self.output[step-1], gradAction_)
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
         local gradAction_ = self.forwardActions and gradOutput[step][2] or action.output:clone():zero()
               
         if step == 1 then
            -- backward through initial starting actions
            action:accGradParameters(self._initInput, gradAction_, scale)
         else
            -- Note : gradOutput is ignored by REINFORCE modules so we give action.output as a dummy variable
            action:accGradParameters(self.output[step-1], gradAction_, scale)
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
         local gradAction_ = self.forwardActions and gradOutput[step][2] or action.output:clone():zero()
         
         if step == 1 then
            -- backward through initial starting actions
            action:accUpdateGradParameters(self._initInput, gradAction_, lr)
         else
            -- Note : gradOutput is ignored by REINFORCE modules so we give action.output as a dummy variable
            action:accUpdateGradParameters(self.output[step-1], gradAction_, lr)
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

   function RecurrentAttention:type(type)
      self._input = nil
      self._actions = nil
      self._crop = nil
      self._pad = nil
      self._byte = nil
      return parent.type(self, type)
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
   
   makeOldRecurrent()

   if not pcall(function() require "image" end) then return end -- needs the image package
   
   local opt = {
      glimpseDepth = 3,
      glimpseHiddenSize = 20,
      glimpsePatchSize = 8,
      locatorHiddenSize = 20,
      imageHiddenSize = 20,
      hiddenSize = 20,
      rho = 5,
      locatorStd = 0.1,
      inputSize = 28,
      nClass = 10,
      batchSize = 4
   }
   
   -- glimpse network (rnn input layer) 
   local locationSensor = nn.Sequential()
   locationSensor:add(nn.SelectTable(2))
   locationSensor:add(nn.Linear(2, opt.locatorHiddenSize))
   locationSensor:add(nn.ReLU())

   local glimpseSensor = nn.Sequential()
   glimpseSensor:add(nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale))
   glimpseSensor:add(nn.Collapse(3))
   glimpseSensor:add(nn.Linear(1*(opt.glimpsePatchSize^2)*opt.glimpseDepth, opt.glimpseHiddenSize))
   glimpseSensor:add(nn.ReLU())

   local glimpse = nn.Sequential()
   --glimpse:add(nn.PrintSize("preglimpse"))
   glimpse:add(nn.ConcatTable():add(locationSensor):add(glimpseSensor))
   glimpse:add(nn.JoinTable(1,1))
   glimpse:add(nn.Linear(opt.glimpseHiddenSize+opt.locatorHiddenSize, opt.imageHiddenSize))
   glimpse:add(nn.ReLU())
   glimpse:add(nn.Linear(opt.imageHiddenSize, opt.hiddenSize))

   -- recurrent neural network
   local rnn = nn.Recurrent(
      opt.hiddenSize, 
      glimpse,
      nn.Linear(opt.hiddenSize, opt.hiddenSize), 
      nn.ReLU(), 99999
   )
   
   local rnn2 = nn.ReTest(
      rnn.startModule:clone(),
      glimpse:clone(),
      rnn.feedbackModule:clone(),
      nn.ReLU(), 99999
   )

   -- output layer (actions)
   local locator = nn.Sequential()
   locator:add(nn.Linear(opt.hiddenSize, 2))
   locator:add(nn.HardTanh()) -- bounds mean between -1 and 1
   local rn = nn.ReinforceNormal(2*opt.locatorStd)
   rn:evaluate() -- so we can match the output from sg to sg2 (i.e deterministic)
   locator:add(rn) -- sample from normal, uses REINFORCE learning rule
   locator:add(nn.HardTanh()) -- bounds sample between -1 and 1
   
   -- model is a reinforcement learning agent
   local rva2 = nn.RATest(rnn2:clone(), locator:clone(), opt.rho, {opt.hiddenSize})
   local rva = nn.RecurrentAttention(rnn:clone(), locator:clone(), opt.rho, {opt.hiddenSize})
   
   for i=1,3 do
   
      local input = torch.randn(opt.batchSize,1,opt.inputSize,opt.inputSize)
      local gradOutput = {}
      for step=1,opt.rho do
         table.insert(gradOutput, torch.randn(opt.batchSize, opt.hiddenSize))
      end
      
      -- now we compare to the nn.RATest class (which, we know, works)
      rva:zeroGradParameters()
      rva2:zeroGradParameters()
      
      local output = rva:forward(input)
      local output2 = rva2:forward(input)
      
      mytester:assert(#output == #output2, "RecurrentAttention #output err")
      for i=1,#output do
         mytester:assertTensorEq(output[i], output2[i], 0.0000001, "RecurrentAttention output err "..i)
      end
      
      local reward = torch.randn(opt.batchSize)
      rva:reinforce(reward)
      rva2:reinforce(reward:clone())
      local gradInput = rva:backward(input, gradOutput)
      local gradInput2 = rva2:backward(input, gradOutput)
      
      mytester:assertTensorEq(gradInput, gradInput2, 0.0000001, "RecurrentAttention gradInput err")
      
      rva:updateParameters(1)
      rva2:updateParameters(1)
      
      local params, gradParams = rva:parameters()
      local params2, gradParams2 = rva2:parameters()
      
      for i=1,#params do
         mytester:assertTensorEq(params[i], params2[i], 0.0000001, "RecurrentAttention, param err "..i)
         mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001, "RecurrentAttention, gradParam err "..i)
      end
   end
   
   -- test with explicit recursor
   
   -- model is a reinforcement learning agent
   local rva2 = nn.RATest(rnn2:clone(), locator:clone(), opt.rho, {opt.hiddenSize})
   local rva = nn.RecurrentAttention(nn.Recursor(rnn:clone()), locator:clone(), opt.rho, {opt.hiddenSize})
   
   for i=1,3 do
      local input = torch.randn(opt.batchSize,1,opt.inputSize,opt.inputSize)
      local gradOutput = {}
      for step=1,opt.rho do
         table.insert(gradOutput, torch.randn(opt.batchSize, opt.hiddenSize))
      end
      
      -- now we compare to the nn.RATest class (which, we know, works)
      rva:zeroGradParameters()
      rva2:zeroGradParameters()
      
      local output = rva:forward(input)
      local output2 = rva2:forward(input)
      
      mytester:assert(#output == #output2, "RecurrentAttention(Recursor) #output err")
      for i=1,#output do
         mytester:assertTensorEq(output[i], output2[i], 0.0000001, "RecurrentAttention(Recursor) output err "..i)
      end
      
      local reward = torch.randn(opt.batchSize)
      rva:reinforce(reward)
      rva2:reinforce(reward:clone())
      local gradInput = rva:backward(input, gradOutput)
      local gradInput2 = rva2:backward(input, gradOutput)
      
      mytester:assertTensorEq(gradInput, gradInput2, 0.0000001, "RecurrentAttention(Recursor) gradInput err")
      
      rva:updateParameters(1)
      rva2:updateParameters(1)
      
      local params, gradParams = rva:parameters()
      local params2, gradParams2 = rva2:parameters()
      
      for i=1,#params do
         mytester:assertTensorEq(params[i], params2[i], 0.0000001, "RecurrentAttention(Recursor), param err "..i)
         mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001, "RecurrentAttention(Recursor), gradParam err "..i)
      end
   end
end
   
function rnntest.LSTM_nn_vs_nngraph()
   local model = {}
   -- match the successful https://github.com/wojzaremba/lstm
   -- We want to make sure our LSTM matches theirs.
   -- Also, the ugliest unit test you have every seen.
   -- Resolved 2-3 annoying bugs with it.
   local success = pcall(function() require 'nngraph' end)
   if not success then
      return
   end
   
   local vocabSize = 100
   local inputSize = 30
   local batchSize = 4
   local nLayer = 2
   local dropout = 0
   local nStep = 10
   local lr = 1
   
   -- build nn equivalent of nngraph model
   local model2 = nn.Sequential()
   local container2 = nn.Container()
   container2:add(nn.LookupTable(vocabSize, inputSize))
   model2:add(container2:get(1))
   local dropout2 = nn.Dropout(dropout)
   model2:add(dropout2)
   local seq21 = nn.SplitTable(1,2)
   model2:add(seq21)
   container2:add(nn.FastLSTM(inputSize, inputSize))
   local seq22 = nn.Sequencer(container2:get(2))
   model2:add(seq22)
   local seq24 = nn.Sequencer(nn.Dropout(0))
   model2:add(seq24)
   container2:add(nn.FastLSTM(inputSize, inputSize))
   local seq23 = nn.Sequencer(container2:get(3))
   model2:add(seq23)
   local seq25 = nn.Sequencer(nn.Dropout(0))
   model2:add(seq25)
   container2:add(nn.Linear(inputSize, vocabSize))
   local mlp = nn.Sequential():add(container2:get(4)):add(nn.LogSoftMax()) -- test double encapsulation
   model2:add(nn.Sequencer(mlp))
   
   local criterion2 = nn.ModuleCriterion(nn.SequencerCriterion(nn.ClassNLLCriterion()), nil, nn.SplitTable(1,1))
   
   
   -- nngraph model 
   local container = nn.Container()
   local lstmId = 1
   local function lstm(x, prev_c, prev_h)
      -- Calculate all four gates in one go
      local i2h = nn.Linear(inputSize, 4*inputSize)
      local dummy = nn.Container()
      dummy:add(i2h)
      i2h = i2h(x)
      local h2h = nn.LinearNoBias(inputSize, 4*inputSize)
      dummy:add(h2h)
      h2h = h2h(prev_h)
      container:add(dummy)
      local gates = nn.CAddTable()({i2h, h2h})

      -- Reshape to (batch_size, n_gates, hid_size)
      -- Then slize the n_gates dimension, i.e dimension 2
      local reshaped_gates =  nn.Reshape(4,inputSize)(gates)
      local sliced_gates = nn.SplitTable(2)(reshaped_gates)

      -- Use select gate to fetch each gate and apply nonlinearity
      local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
      local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
      local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
      local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

      local next_c           = nn.CAddTable()({
         nn.CMulTable()({forget_gate, prev_c}),
         nn.CMulTable()({in_gate,     in_transform})
      })
      local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
      lstmId = lstmId + 1
      return next_c, next_h
   end
   local function create_network()
      local x                = nn.Identity()()
      local y                = nn.Identity()()
      local prev_s           = nn.Identity()()
      local lookup = nn.LookupTable(vocabSize, inputSize)
      container:add(lookup)
      local identity = nn.Identity()
      lookup = identity(lookup(x))
      local i                = {[0] = lookup}
      local next_s           = {}
      local split         = {prev_s:split(2 * nLayer)}
      for layer_idx = 1, nLayer do
         local prev_c         = split[2 * layer_idx - 1]
         local prev_h         = split[2 * layer_idx]
         local dropped        = nn.Dropout(dropout)(i[layer_idx - 1])
         local next_c, next_h = lstm(dropped, prev_c, prev_h)
         table.insert(next_s, next_c)
         table.insert(next_s, next_h)
         i[layer_idx] = next_h
      end
      
      local h2y              = nn.Linear(inputSize, vocabSize)
      container:add(h2y)
      local dropped          = nn.Dropout(dropout)(i[nLayer])
      local pred             = nn.LogSoftMax()(h2y(dropped))
      local err              = nn.ClassNLLCriterion()({pred, y})
      local module           = nn.gModule({x, y, prev_s}, {err, nn.Identity()(next_s)})
      module:getParameters():uniform(-0.1, 0.1)
      module._lookup = identity
      return module
   end
   
   local function g_cloneManyTimes(net, T)
      local clones = {}
      local params, gradParams = net:parameters()
      local mem = torch.MemoryFile("w"):binary()
      assert(net._lookup)
      mem:writeObject(net)
      for t = 1, T do
         local reader = torch.MemoryFile(mem:storage(), "r"):binary()
         local clone = reader:readObject()
         reader:close()
         local cloneParams, cloneGradParams = clone:parameters()
         for i = 1, #params do
            cloneParams[i]:set(params[i])
            cloneGradParams[i]:set(gradParams[i])
         end
         clones[t] = clone
         collectgarbage()
      end
      mem:close()
      return clones
   end
   
   local model = {}
   local paramx, paramdx
   local core_network = create_network()
   
   -- sync nn with nngraph model
   local params, gradParams = container:getParameters()
   local params2, gradParams2 = container2:getParameters()
   params2:copy(params)
   container:zeroGradParameters()
   container2:zeroGradParameters()
   paramx, paramdx = core_network:getParameters()
   
   model.s = {}
   model.ds = {}
   model.start_s = {}
   for j = 0, nStep do
      model.s[j] = {}
      for d = 1, 2 * nLayer do
         model.s[j][d] = torch.zeros(batchSize, inputSize)
      end
   end
   for d = 1, 2 * nLayer do
      model.start_s[d] = torch.zeros(batchSize, inputSize)
      model.ds[d] = torch.zeros(batchSize, inputSize)
   end
   model.core_network = core_network
   model.rnns = g_cloneManyTimes(core_network, nStep)
   model.norm_dw = 0
   model.err = torch.zeros(nStep)
   
   -- more functions for nngraph baseline
   local function g_replace_table(to, from)
     assert(#to == #from)
     for i = 1, #to do
       to[i]:copy(from[i])
     end
   end

   local function reset_ds()
     for d = 1, #model.ds do
       model.ds[d]:zero()
     end
   end
   
   local function reset_state(state)
     state.pos = 1
     if model ~= nil and model.start_s ~= nil then
       for d = 1, 2 * nLayer do
         model.start_s[d]:zero()
       end
     end
   end

   local function fp(state)
     g_replace_table(model.s[0], model.start_s)
     if state.pos + nStep > state.data:size(1) then
         error"Not Supposed to happen in this unit test"
     end
     for i = 1, nStep do
       local x = state.data[state.pos]
       local y = state.data[state.pos + 1]
       local s = model.s[i - 1]
       model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
       state.pos = state.pos + 1
     end
     g_replace_table(model.start_s, model.s[nStep])
     return model.err:mean()
   end

   model.dss = {}
   local function bp(state)
     paramdx:zero()
     local __, gradParams = core_network:parameters()
     for i=1,#gradParams do
        mytester:assert(gradParams[i]:sum() == 0)
     end
     reset_ds() -- backward of last step in each sequence is zero
     for i = nStep, 1, -1 do
       state.pos = state.pos - 1
       local x = state.data[state.pos]
       local y = state.data[state.pos + 1]
       local s = model.s[i - 1]
       local derr = torch.ones(1)
       local tmp = model.rnns[i]:backward({x, y, s}, {derr, model.ds,})[3]
       model.dss[i-1] = tmp
       g_replace_table(model.ds, tmp)
     end
     state.pos = state.pos + nStep
     paramx:add(-lr, paramdx)
   end
   
   -- inputs and targets (for nngraph implementation)
   local inputs = torch.Tensor(nStep*10, batchSize):random(1,vocabSize)

   -- is everything aligned between models?
   local params_, gradParams_ = container:parameters()
   local params2_, gradParams2_ = container2:parameters()

   for i=1,#params_ do
      mytester:assertTensorEq(params_[i], params2_[i], 0.00001, "nn vs nngraph unaligned params err "..i)
      mytester:assertTensorEq(gradParams_[i], gradParams2_[i], 0.00001, "nn vs nngraph unaligned gradParams err "..i)
   end
   
   -- forward 
   local state = {pos=1,data=inputs}
   local err = fp(state)
   
   local inputs2 = inputs:narrow(1,1,nStep):transpose(1,2)
   local targets2 = inputs:narrow(1,2,nStep):transpose(1,2)
   local outputs2 = model2:forward(inputs2)
   local err2 = criterion2:forward(outputs2, targets2)
   mytester:assert(math.abs(err - err2/nStep) < 0.0001, "nn vs nngraph err error")
   
   -- backward/update
   bp(state)
   
   local gradOutputs2 = criterion2:backward(outputs2, targets2)
   model2:backward(inputs2, gradOutputs2)
   model2:updateParameters(lr)
   model2:zeroGradParameters()
   
   for i=1,#gradParams2_ do
      mytester:assert(gradParams2_[i]:sum() == 0)
   end
   
   for i=1,#params_ do
      mytester:assertTensorEq(params_[i], params2_[i], 0.00001, "nn vs nngraph params err "..i)
   end
   
   for i=1,nStep do
      mytester:assertTensorEq(model.rnns[i]._lookup.output, dropout2.output:select(2,i), 0.0000001)
      mytester:assertTensorEq(model.rnns[i]._lookup.gradInput, dropout2.gradInput:select(2,i), 0.0000001)
   end
   
   -- next_c, next_h, next_c...
   for i=nStep-1,2,-1 do
      mytester:assertTensorEq(model.dss[i][1], container2:get(2).gradCells[i], 0.0000001, "gradCells1 err "..i)
      mytester:assertTensorEq(model.dss[i][2], container2:get(2)._gradOutputs[i] - seq24.gradInput[i], 0.0000001, "gradOutputs1 err "..i)
      mytester:assertTensorEq(model.dss[i][3], container2:get(3).gradCells[i], 0.0000001, "gradCells2 err "..i)
      mytester:assertTensorEq(model.dss[i][4], container2:get(3)._gradOutputs[i] - seq25.gradInput[i], 0.0000001, "gradOutputs2 err "..i)
   end
   
   for i=1,#params2_ do
      params2_[i]:copy(params_[i])
      gradParams_[i]:copy(gradParams2_[i])
   end
   
   local gradInputClone = dropout2.gradInput:select(2,1):clone()
   
   local start_s = _.map(model.start_s, function(k,v) return v:clone() end)
   mytester:assertTensorEq(start_s[1], container2:get(2).cells[nStep], 0.0000001)
   mytester:assertTensorEq(start_s[2], container2:get(2).outputs[nStep], 0.0000001)
   mytester:assertTensorEq(start_s[3], container2:get(3).cells[nStep], 0.0000001)
   mytester:assertTensorEq(start_s[4], container2:get(3).outputs[nStep], 0.0000001)
   
   -- and do it again
   -- forward 
   -- reset_state(state)
   
   local inputs2 = inputs:narrow(1,nStep+1,nStep):transpose(1,2)
   local targets2 = inputs:narrow(1,nStep+2,nStep):transpose(1,2)
   model2:remember()
   local outputs2 = model2:forward(inputs2)
   
   local inputsClone = seq21.output[nStep]:clone()
   local outputsClone = container2:get(2).outputs[nStep]:clone()
   local cellsClone = container2:get(2).cells[nStep]:clone()
   local err2 = criterion2:forward(outputs2, targets2)
   local state = {pos=nStep+1,data=inputs}
   local err = fp(state)
   mytester:assert(math.abs(err2/nStep - err) < 0.00001, "nn vs nngraph err error")
   -- backward/update
   bp(state)
   
   local gradOutputs2 = criterion2:backward(outputs2, targets2)
   model2:backward(inputs2, gradOutputs2)
   
   mytester:assertTensorEq(start_s[1], container2:get(2).cells[nStep], 0.0000001)
   mytester:assertTensorEq(start_s[2], container2:get(2).outputs[nStep], 0.0000001)
   mytester:assertTensorEq(start_s[3], container2:get(3).cells[nStep], 0.0000001)
   mytester:assertTensorEq(start_s[4], container2:get(3).outputs[nStep], 0.0000001)
   
   model2:updateParameters(lr)
   
   mytester:assertTensorEq(inputsClone, seq21.output[nStep], 0.000001)
   mytester:assertTensorEq(outputsClone, container2:get(2).outputs[nStep], 0.000001)
   mytester:assertTensorEq(cellsClone, container2:get(2).cells[nStep], 0.000001)
   
   -- next_c, next_h, next_c...
   for i=nStep-1,2,-1 do
      mytester:assertTensorEq(model.dss[i][1], container2:get(2).gradCells[i+nStep], 0.0000001, "gradCells1 err "..i)
      mytester:assertTensorEq(model.dss[i][2], container2:get(2)._gradOutputs[i+nStep] - seq24.gradInput[i], 0.0000001, "gradOutputs1 err "..i)
      mytester:assertTensorEq(model.dss[i][3], container2:get(3).gradCells[i+nStep], 0.0000001, "gradCells2 err "..i)
      mytester:assertTensorEq(model.dss[i][4], container2:get(3)._gradOutputs[i+nStep] - seq25.gradInput[i], 0.0000001, "gradOutputs2 err "..i)
   end
   
   mytester:assertTensorNe(gradInputClone, dropout2.gradInput:select(2,1), 0.0000001, "lookup table gradInput1 err")
   
   for i=1,nStep do
      mytester:assertTensorEq(model.rnns[i]._lookup.output, dropout2.output:select(2,i), 0.0000001, "lookup table output err "..i)
      mytester:assertTensorEq(model.rnns[i]._lookup.gradInput, dropout2.gradInput:select(2,i), 0.0000001, "lookup table gradInput err "..i)
   end
   
   for i=1,#params_ do
      mytester:assertTensorEq(params_[i], params2_[i], 0.00001, "nn vs nngraph second update params err "..i)
   end
end

function rnntest.LSTM_char_rnn()
   -- benchmark our LSTM against char-rnn's LSTM
   if not benchmark then
      return
   end
   
   local success = pcall(function() 
         require 'nngraph' 
         require 'cunn' 
      end)
   if not success then
      return
   end
   
   local batch_size = 50
   local input_size = 65
   local rnn_size = 128
   local n_layer = 2
   local seq_len = 50
   
   local inputs = {}
   local gradOutputs = {}
   for i=1,seq_len do
      table.insert(inputs, torch.Tensor(batch_size):random(1,input_size):cuda())
      table.insert(gradOutputs, torch.randn(batch_size, input_size):cuda())
   end
   
   local a = torch.Timer()
   local function clone_list(tensor_list, zero_too)
       -- utility function. todo: move away to some utils file?
       -- takes a list of tensors and returns a list of cloned tensors
       local out = {}
       for k,v in pairs(tensor_list) do
           out[k] = v:clone()
           if zero_too then out[k]:zero() end
       end
       return out
   end
   
   local model_utils = {}
   function model_utils.combine_all_parameters(...)
      local con = nn.Container()
      for i, net in ipairs{...} do
         con:add(net)
      end
      return con:getParameters()
   end

   function model_utils.clone_many_times(net, T)
       local clones = {}

       local params, gradParams
       if net.parameters then
           params, gradParams = net:parameters()
           if params == nil then
               params = {}
           end
       end

       local paramsNoGrad
       if net.parametersNoGrad then
           paramsNoGrad = net:parametersNoGrad()
       end

       local mem = torch.MemoryFile("w"):binary()
       mem:writeObject(net)

       for t = 1, T do
           -- We need to use a new reader for each clone.
           -- We don't want to use the pointers to already read objects.
           local reader = torch.MemoryFile(mem:storage(), "r"):binary()
           local clone = reader:readObject()
           reader:close()

           if net.parameters then
               local cloneParams, cloneGradParams = clone:parameters()
               local cloneParamsNoGrad
               for i = 1, #params do
                   cloneParams[i]:set(params[i])
                   cloneGradParams[i]:set(gradParams[i])
               end
               if paramsNoGrad then
                   cloneParamsNoGrad = clone:parametersNoGrad()
                   for i =1,#paramsNoGrad do
                       cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                   end
               end
           end

           clones[t] = clone
           collectgarbage()
       end

       mem:close()
       return clones
   end
   
   local function makeCharLSTM(input_size, rnn_size, n)
      local dropout = 0 

      -- there will be 2*n+1 inputs
      local inputs = {}
      table.insert(inputs, nn.Identity()()) -- x
      for L = 1,n do
         table.insert(inputs, nn.Identity()()) -- prev_c[L]
         table.insert(inputs, nn.Identity()()) -- prev_h[L]
      end

      local x, input_size_L
      local outputs = {}
      for L = 1,n do
         -- c,h from previos timesteps
         local prev_h = inputs[L*2+1]
         local prev_c = inputs[L*2]
         -- the input to this layer
         if L == 1 then 
            x = nn.OneHot(input_size)(inputs[1])
            input_size_L = input_size
         else 
            x = outputs[(L-1)*2] 
            if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
            input_size_L = rnn_size
         end
         -- evaluate the input sums at once for efficiency
         local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
         local h2h = nn.LinearNoBias(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
         local all_input_sums = nn.CAddTable()({i2h, h2h})

         local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
         local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
         -- decode the gates
         local in_gate = nn.Sigmoid()(n1)
         local forget_gate = nn.Sigmoid()(n2)
         local out_gate = nn.Sigmoid()(n3)
         -- decode the write inputs
         local in_transform = nn.Tanh()(n4)
         -- perform the LSTM update
         local next_c           = nn.CAddTable()({
           nn.CMulTable()({forget_gate, prev_c}),
           nn.CMulTable()({in_gate,     in_transform})
         })
         -- gated cells form the output
         local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

         table.insert(outputs, next_c)
         table.insert(outputs, next_h)
      end

      -- set up the decoder
      local top_h = outputs[#outputs]
      if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
      local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
      local logsoft = nn.LogSoftMax()(proj)
      table.insert(outputs, logsoft)

      local lstm = nn.gModule(inputs, outputs):cuda()
      return lstm
   end
   
   -- the initial state of the cell/hidden states
   local init_state = {}
   for L=1,n_layer do
       local h_init = torch.zeros(batch_size, rnn_size):cuda()
       table.insert(init_state, h_init:clone())
       table.insert(init_state, h_init:clone())
   end
   
   local lstm1 = makeCharLSTM(input_size, rnn_size, n_layer)   
   local crit1 = nn.ClassNLLCriterion()
   local protos = {rnn=lstm1,criterion=crit1}
   
   -- make a bunch of clones after flattening, as that reallocates memory
   local clones = {}
   for name,proto in pairs(protos) do
       clones[name] = model_utils.clone_many_times(proto, seq_len, not proto.parameters)
   end

   -- put the above things into one flattened parameters tensor
   local params, grad_params = model_utils.combine_all_parameters(lstm1)
   
   local init_state_global = clone_list(init_state)
   
   -- do fwd/bwd and return loss, grad_params
   local function trainCharrnn(x, y, fwdOnly)
      local rnn_state = {[0] = init_state_global}
      local predictions = {}           -- softmax outputs
      local loss = 0
      for t=1,seq_len do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        --loss = loss + clones.criterion[t]:forward(predictions[t], y[t])
      end
      
      if not fwdOnly then
         --loss = loss / seq_len
         ------------------ backward pass -------------------
         -- initialize gradient at time t to be zeros (there's no influence from future)
         local drnn_state = {[seq_len] = clone_list(init_state, true)} -- true also zeros the clones
         for t=seq_len,1,-1 do
           -- backprop through loss, and softmax/linear
           --local doutput_t = clones.criterion[t]:backward(predictions[t], y[t])
           local doutput_t = y[t]
           table.insert(drnn_state[t], doutput_t)
           local dlst = clones.rnn[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
           drnn_state[t-1] = {}
           for k,v in pairs(dlst) do
               if k > 1 then -- k == 1 is gradient on x, which we dont need
                   -- note we do k-1 because first item is dembeddings, and then follow the 
                   -- derivatives of the state, starting at index 2. I know...
                   drnn_state[t-1][k-1] = v
               end
           end
         end
      end
      ------------------------ misc ----------------------
      -- transfer final state to initial state (BPTT)
      init_state_global = rnn_state[#rnn_state]
   end
   
   local charrnnsetuptime = a:time().real
   
   local a = torch.Timer()
   
   local function makeRnnLSTM(input_size, rnn_size, n)
      local seq = nn.Sequential()
         :add(nn.OneHot(input_size))
      
      local inputSize = input_size
      for L=1,n do
         seq:add(nn.FastLSTM(inputSize, rnn_size))
         inputSize = rnn_size
      end
      
      seq:add(nn.Linear(rnn_size, input_size))
      seq:add(nn.LogSoftMax())
      
      local lstm = nn.Sequencer(seq)
      
      lstm:cuda()
      
      return lstm 
   end
   
   nn.FastLSTM.usenngraph = true
   local lstm2 = makeRnnLSTM(input_size, rnn_size, n_layer, gpu)
   nn.FastLSTM.usenngraph = false
   
   local function trainRnn(x, y, fwdOnly)
      local outputs = lstm2:forward(x)
      if not fwdOnly then
         local gradInputs = lstm2:backward(x, y)
      end
   end
   
   local rnnsetuptime = a:time().real
   
   -- char-rnn (nngraph)
   
   local a = torch.Timer()
   trainCharrnn(inputs, gradOutputs)
   cutorch.synchronize()
   charrnnsetuptime = charrnnsetuptime + a:time().real
   collectgarbage()
   
   local a = torch.Timer()
   for i=1,10 do
      trainCharrnn(inputs, gradOutputs)
   end
   cutorch.synchronize()
   local chartime = a:time().real
   
   -- rnn
   local a = torch.Timer()
   trainRnn(inputs, gradOutputs)
   cutorch.synchronize()
   rnnsetuptime = rnnsetuptime + a:time().real
   collectgarbage()
   print("Benchmark")
   print("setuptime : char, rnn, char/rnn", charrnnsetuptime, rnnsetuptime, charrnnsetuptime/rnnsetuptime)
   local a = torch.Timer()
   for i=1,10 do
      trainRnn(inputs, gradOutputs)
   end
   cutorch.synchronize()
   local rnntime = a:time().real
   print("runtime: char, rnn, char/rnn", chartime, rnntime, chartime/rnntime)
   
   -- on NVIDIA Titan Black :
   -- with FastLSTM.usenngraph = false  :
   -- setuptime : char, rnn, char/rnn 1.5070691108704 1.1547832489014 1.3050666541138 
   -- runtime: char, rnn, char/rnn    1.0558769702911 1.7060630321503 0.61889681119246
   
   -- with FastLSTM.usenngraph = true :
   -- setuptime : char, rnn, char/rnn 1.5920469760895 2.4352579116821 0.65374881586558
   -- runtime: char, rnn, char/rnn    1.0614919662476 1.124755859375  0.94375322199913
end

-- https://github.com/Element-Research/rnn/issues/28
function rnntest.Recurrent_checkgrad()
   if not pcall(function() require 'optim' end) then return end

   local batchSize = 3
   local hiddenSize = 2
   local nIndex = 2
   local rnn = nn.Recurrent(hiddenSize, nn.LookupTable(nIndex, hiddenSize),
                    nn.Linear(hiddenSize, hiddenSize))
   local seq = nn.Sequential()
      :add(rnn)
      :add(nn.Linear(hiddenSize, hiddenSize))
   
   rnn = nn.Sequencer(seq)

   local criterion = nn.SequencerCriterion(nn.MSECriterion())
   local inputs, targets = {}, {}
   for i=1,2 do
      inputs[i] = torch.Tensor(batchSize):random(1,nIndex)
      targets[i] = torch.randn(batchSize, hiddenSize)
   end
   
   local parameters, grads = rnn:getParameters()
   
   function f(x)
      parameters:copy(x)
      -- Do the forward prop
      rnn:zeroGradParameters()
      assert(grads:sum() == 0)
      local outputs = rnn:forward(inputs)
      local err = criterion:forward(outputs, targets)
      local gradOutputs = criterion:backward(outputs, targets)
      rnn:backward(inputs, gradOutputs)
      return err, grads
   end

   local err = optim.checkgrad(f, parameters:clone())
   mytester:assert(err < 0.0001, "Recurrent optim.checkgrad error")
end

function rnntest.LSTM_checkgrad()
   if not pcall(function() require 'optim' end) then return end

   local hiddenSize = 2
   local nIndex = 2
   local r = nn.LSTM(hiddenSize, hiddenSize)

   local rnn = nn.Sequential()
   rnn:add(r)
   rnn:add(nn.Linear(hiddenSize, nIndex))
   rnn:add(nn.LogSoftMax())
   rnn = nn.Recursor(rnn)

   local criterion = nn.ClassNLLCriterion()
   local inputs = torch.randn(4, 2)
   local targets = torch.Tensor{1, 2, 1, 2}:resize(4, 1)
   local parameters, grads = rnn:getParameters()
   
   function f(x)
      parameters:copy(x)
      
      -- Do the forward prop
      rnn:zeroGradParameters()
      local err = 0
      local outputs = {}
      for i = 1, inputs:size(1) do
         outputs[i] = rnn:forward(inputs[i])
         err = err + criterion:forward(outputs[i], targets[i])
      end
      for i = inputs:size(1), 1, -1 do
         local gradOutput = criterion:backward(outputs[i], targets[i])
         rnn:backward(inputs[i], gradOutput)
      end
      rnn:forget()
      return err, grads
   end

   local err = optim.checkgrad(f, parameters:clone())
   mytester:assert(err < 0.0001, "LSTM optim.checkgrad error")
end

function rnntest.Recursor()
   local batchSize = 4
   local inputSize = 3
   local hiddenSize = 12
   local outputSize = 7
   local rho = 5 
   
   -- USE CASE 1. Recursor(Recurrent)
   
   local inputModule = nn.Linear(inputSize, outputSize)
   local transferModule = nn.Sigmoid()
   -- test MLP feedback Module (because of Module:representations())
   local feedbackModule = nn.Sequential()
   feedbackModule:add(nn.Linear(outputSize, hiddenSize))
   feedbackModule:add(nn.Sigmoid())
   feedbackModule:add(nn.Linear(hiddenSize, outputSize))
   local start = nn.Add(outputSize)
   
   local rnn = nn.Recurrent(start, nn.Identity(), feedbackModule, transferModule:clone(), rho)
   local re = nn.Recursor(nn.Sequential():add(inputModule):add(rnn), rho)
   re:zeroGradParameters()
   
   local re2 = nn.Recurrent(start:clone(), inputModule:clone(), feedbackModule:clone(), transferModule:clone(), rho)
   re2:zeroGradParameters()
   
   local inputs = {}
   local gradOutputs = {}
   local outputs, outputs2 = {}, {}
   local gradInputs = {}
   
   for i=1,rho do
      table.insert(inputs, torch.randn(batchSize, inputSize))
      table.insert(gradOutputs, torch.randn(batchSize, outputSize))
      -- forward
      table.insert(outputs, re:forward(inputs[i]))
      table.insert(outputs2, re2:forward(inputs[i]))
   end
   
   local gradInputs_2 = {}
   for i=rho,1,-1 do
      -- backward
      gradInputs_2[i] = re2:backward(inputs[i], gradOutputs[i])
   end
   
   re2:updateParameters(0.1)
   
   -- recursor requires reverse-time-step order during backward
   for i=rho,1,-1 do
      gradInputs[i] = re:backward(inputs[i], gradOutputs[i])
   end
   
   for i=1,rho do
      mytester:assertTensorEq(outputs[i], outputs2[i], 0.0000001, "Recursor(Recurrent) fwd err "..i)
      mytester:assertTensorEq(gradInputs[i], gradInputs_2[i], 0.0000001, "Recursor(Recurrent) bwd err "..i)
   end
   
   re:updateParameters(0.1)
   
   local mlp = nn.Container():add(rnn.feedbackModule):add(rnn.startModule):add(inputModule)
   local mlp2 = nn.Container():add(re2.feedbackModule):add(re2.startModule):add(re2.inputModule)
   
   local params, gradParams = mlp:parameters()
   local params2, gradParams2 = mlp2:parameters()
   
   mytester:assert(#params == #params2, "Recursor(Recurrent) #params err")
   for i=1,#params do
      mytester:assertTensorEq(params[i], params2[i], 0.0000001, "Recursor(Recurrent) updateParameter err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001, "Recursor(Recurrent) accGradParams err "..i)
   end
   
   -- USE CASE 2. Recursor(LSTM)
   
   local rnn = nn.LSTM(inputSize, outputSize)
   local re2 = rnn:clone()
   local re = nn.Recursor(nn.Sequential():add(rnn))
   re:zeroGradParameters()
   re2:zeroGradParameters()
   
   local outputs, outputs2 = {}, {}
   local gradInputs = {}
   
   for i=1,rho do
      -- forward
      table.insert(outputs, re:forward(inputs[i]))
      table.insert(outputs2, re2:forward(inputs[i]))
   end
   
   local gradInputs_2 = {}
   for i=rho,1,-1 do
      -- backward
      gradInputs_2[i] = re2:backward(inputs[i], gradOutputs[i])
   end

   re2:updateParameters(0.1)
   
   -- recursor requires reverse-time-step order during backward
   for i=rho,1,-1 do
      gradInputs[i] = re:backward(inputs[i], gradOutputs[i])
   end
   
   for i=1,rho do
      mytester:assertTensorEq(outputs[i], outputs2[i], 0.0000001, "Recursor(LSTM) fwd err "..i)
      mytester:assertTensorEq(gradInputs[i], gradInputs_2[i], 0.0000001, "Recursor(LSTM) bwd err "..i)
   end
   
   re:updateParameters(0.1)
   
   local params, gradParams = rnn:parameters()
   local params2, gradParams2 = re2:parameters()
   
   mytester:assert(#params == #params2, "Recursor(LSTM) #params err")
   for i=1,#params do
      mytester:assertTensorEq(params[i], params2[i], 0.0000001, "Recursor(LSTM) updateParameter err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001, "Recursor(LSTM) accGradParams err "..i)
   end
   
   -- USE CASE 3. Sequencer(Recursor)
   
   local re2 = nn.LSTM(inputSize, outputSize)
   local lstm2 = re2:clone()
   local rec = nn.Recursor(lstm2)
   local seq = nn.Sequencer(rec)
   mytester:assert(not rec.copyInputs)
   mytester:assert(not rec.copyGradOutputs)
   mytester:assert(not lstm2.copyInputs)
   mytester:assert(not lstm2.copyGradOutputs)
   
   seq:zeroGradParameters()
   re2:zeroGradParameters()
   
   local outputs = seq:forward(inputs)
   local gradInputs = seq:backward(inputs, gradOutputs)
   
   local outputs2 = {}
   for i=1,rho do
      table.insert(outputs2, re2:forward(inputs[i]))
   end
   
   local gradInputs_2 = {}
   for i=rho,1,-1 do
      gradInputs_2[i] = re2:backward(inputs[i], gradOutputs[i])
   end
   
   re2:updateParameters(0.1)
   
   for i=1,rho do
      mytester:assertTensorEq(outputs[i], outputs2[i], 0.0000001, "Sequencer(Recursor(LSTM)) fwd err "..i)
      mytester:assertTensorEq(gradInputs[i], gradInputs_2[i], 0.0000001, "Sequencer(Recursor(LSTM)) bwd err "..i)
   end
   
   seq:updateParameters(0.1)
   
   local params, gradParams = seq:parameters()
   local params2, gradParams2 = re2:parameters()
   
   mytester:assert(#params == #params2, "Sequencer(Recursor(LSTM)) #params err")
   for i=1,#params do
      mytester:assertTensorEq(params[i], params2[i], 0.0000001, "Sequencer(Recursor(LSTM)) updateParameter err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001, "Sequencer(Recursor(LSTM)) accGradParams err "..i)
   end
   
   -- USE CASE 4. Recursor(Recursor(LSTM))
   
   local rnn = nn.LSTM(inputSize, outputSize)
   local re2 = rnn:clone()
   local re = nn.Recursor(nn.Recursor(nn.Sequential():add(rnn)))
   re:zeroGradParameters()
   re2:zeroGradParameters()
   
   local outputs, outputs2 = {}, {}
   local gradInputs = {}
   
   for i=1,rho do
      -- forward
      table.insert(outputs, re:forward(inputs[i]))
      table.insert(outputs2, re2:forward(inputs[i]))
   end
   
   local gradInputs_2 = {}
   for i=rho,1,-1 do
      -- backward
      gradInputs_2[i] = re2:backward(inputs[i], gradOutputs[i])
   end

   re2:updateParameters(0.1)
   
   -- recursor requires reverse-time-step order during backward
   for i=rho,1,-1 do
      gradInputs[i] = re:backward(inputs[i], gradOutputs[i])
   end
   
   for i=1,rho do
      mytester:assertTensorEq(outputs[i], outputs2[i], 0.0000001, "Recursor(Recursor(LSTM)) fwd err "..i)
      mytester:assertTensorEq(gradInputs[i], gradInputs_2[i], 0.0000001, "Recursor(Recursor(LSTM)) bwd err "..i)
   end
   
   re:updateParameters(0.1)
   
   local params, gradParams = rnn:parameters()
   local params2, gradParams2 = re2:parameters()
   
   mytester:assert(#params == #params2, "Recursor(Recursor(LSTM)) #params err")
   for i=1,#params do
      mytester:assertTensorEq(params[i], params2[i], 0.0000001, "Recursor(Recursor(LSTM)) updateParameter err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001, "Recursor(Recursor(LSTM)) accGradParams err "..i)
   end
   
end

function rnntest.Recurrence()
   local batchSize = 4
   local inputSize = 10
   local outputSize = 12
   local rho = 3

   -- 1. compare to LSTM
   local lstm2 = nn.LSTM(inputSize, outputSize)
   local rm = lstm2.recurrentModule:clone()
   local seq2 = nn.Sequencer(lstm2)
   
   rm:insert(nn.FlattenTable(), 1)
   local recurrence = nn.Recurrence(rm, {{outputSize}, {outputSize}}, 1)
   local lstm = nn.Sequential():add(recurrence):add(nn.SelectTable(1))
   local seq = nn.Sequencer(lstm)
   
   local inputs, gradOutputs = {}, {}
   for i=1,rho do
      table.insert(inputs, torch.randn(batchSize, inputSize))
      table.insert(gradOutputs, torch.randn(batchSize, outputSize))
   end
   
   seq:zeroGradParameters()
   seq2:zeroGradParameters()
   
   local outputs = seq:forward(inputs)
   local outputs2 = seq2:forward(inputs)
   
   for i=1,rho do
      mytester:assertTensorEq(outputs[i], outputs2[i], 0.0000001, "Recurrence fwd err "..i)
   end
   
   local gradInputs = seq:backward(inputs, gradOutputs)
   local gradInputs2 = seq2:backward(inputs, gradOutputs)
   
   for i=1,rho do
      mytester:assertTensorEq(gradInputs[i], gradInputs2[i], 0.0000001, "Recurrence bwd err "..i)
   end
   
   seq:updateParameters(0.1)
   seq2:updateParameters(0.1)
   
   local params, gradParams = seq:parameters()
   local params2, gradParams2 = seq2:parameters()
   
   mytester:assert(#params == #params2, "Recurrence #params err")
   for i=1,#params do
      mytester:assertTensorEq(params[i], params2[i], 0.0000001, "Recurrence updateParameter err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001, "Recurrence accGradParams err "..i)
   end
   
   -- 2. compare to simple RNN
   
   local nIndex = 50
   local hiddenSize = 20
   
   local inputLayer = nn.LookupTable(nIndex, hiddenSize)
   local feedbackLayer = nn.Linear(hiddenSize, hiddenSize)
   local outputLayer = nn.Linear(hiddenSize, outputSize)
   
   local rnn = nn.Recurrent(hiddenSize, inputLayer, feedbackLayer, nn.Sigmoid(), 99999 )
   rnn.startModule:share(rnn.feedbackModule, 'bias')
   
   -- just so the params are aligned
   local seq2_ = nn.Sequential()
      :add(nn.ParallelTable()
         :add(inputLayer)
         :add(feedbackLayer))
      :add(outputLayer)
   
   local seq2 = nn.Sequencer(nn.Sequential():add(rnn):add(outputLayer):add(nn.LogSoftMax()))
   
   local rm = nn.Sequential()
   :add(nn.ParallelTable()
      :add(inputLayer:clone())
      :add(feedbackLayer:clone()))
   :add(nn.CAddTable())
   :add(nn.Sigmoid())

   local seq = nn.Sequencer(nn.Sequential()
      :add(nn.Recurrence(rm, hiddenSize, 0))
      :add(outputLayer:clone())
      :add(nn.LogSoftMax()))
   
   local inputs, gradOutputs = {}, {}
   for i=1,rho do
      table.insert(inputs, torch.IntTensor(batchSize):random(1,nIndex))
      table.insert(gradOutputs, torch.randn(batchSize, outputSize))
   end
   
   seq:zeroGradParameters()
   seq2:zeroGradParameters()
   
   local outputs = seq:forward(inputs)
   local outputs2 = seq2:forward(inputs)
   
   for i=1,rho do
      mytester:assertTensorEq(outputs[i], outputs2[i], 0.0000001, "Recurrence RNN fwd err "..i)
   end
   
   seq:backward(inputs, gradOutputs)
   seq2:backward(inputs, gradOutputs)
   
   seq:updateParameters(0.1)
   seq2:updateParameters(0.1)
   
   local params, gradParams = seq:parameters()
   local params2, gradParams2 = seq2_:parameters()
   
   mytester:assert(#params == #params2, "Recurrence RNN #params err")
   for i=1,#params do
      mytester:assertTensorEq(params[i], params2[i], 0.0000001, "Recurrence RNN updateParameter err "..i)
      if i~= 3 then -- the gradBias isn't shared (else udpated twice)
         mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.0000001, "Recurrence RNN accGradParams err "..i)
      end
   end
end

function rnntest.Recurrence_FastLSTM()
   -- issue 107
   -- this will test the use case where an AbstractRecurrent.recurrentModule
   -- contains an AbstractRecurrent instance!
   
   local batchSize = 4
   local hiddenSize = 10
   local rho = 3
   
   local lstm = nn.FastLSTM(hiddenSize,hiddenSize)
   
   local rm = nn.Sequential()
      :add(nn.CSubTable())
      :add(lstm)
      :add(nn.Linear(hiddenSize,hiddenSize))
      :add(nn.Sigmoid())    
      
   local rnn = nn.Recurrence(rm, hiddenSize, 1)

   local seq = nn.Sequencer(rnn)
   
   local inputs, gradOutputs = {}, {}
   for i=1,rho do
      inputs[i] = torch.randn(batchSize, hiddenSize)
      gradOutputs[i] = torch.randn(batchSize, hiddenSize)
   end
   
   for n=1,3 do
      seq:evaluate()
      seq:training()
      seq:zeroGradParameters()
      
      seq:forward(inputs)
      seq:backward(inputs, gradOutputs)
      
      mytester:assert(rnn.step == 4)
      mytester:assert(lstm.step == 4)
   end
end

-- mock Recurrent and LSTM recurrentModules for UT
-- must be stateless
-- forwarding zeros must not return zeros -> use Sigmoid()
local function recurrentModule()
   local recurrent = nn.Sequential()
   local parallel = nn.ParallelTable()
   parallel:add(nn.Sigmoid()); parallel:add(nn.Identity())
   recurrent = nn.Sequential()
   recurrent:add(parallel)
   recurrent:add(nn.SelectTable(1))
   return recurrent
end

local function lstmModule()
   local recurrent = nn.Sequential()
   local parallel = nn.ParallelTable()
   parallel:add(nn.Sigmoid()); parallel:add(nn.Identity()); parallel:add(nn.Identity())
   recurrent = nn.Sequential()
   recurrent:add(parallel)
   recurrent:add(nn.NarrowTable(1, 2))
   return recurrent
end

local function firstElement(a)
   return torch.type(a) == 'table' and a[1] or a
end

function rnntest.MaskZero()
   local recurrents = {['recurrent'] = recurrentModule(), ['lstm'] = lstmModule()}
   -- Note we use lstmModule input signature and firstElement to prevent duplicate code
   for name, recurrent in pairs(recurrents) do
      -- test encapsulated module first
      -- non batch
      local i = torch.rand(10)
      local e = nn.Sigmoid():forward(i)
      local o = firstElement(recurrent:forward({i, torch.zeros(10), torch.zeros(10)}))
      mytester:assertlt(torch.norm(o - e), precision, 'mock ' .. name .. ' failed for non batch')
      -- batch
      local i = torch.rand(5, 10)
      local e = nn.Sigmoid():forward(i)
      local o = firstElement(recurrent:forward({i, torch.zeros(5, 10), torch.zeros(5, 10)}))
      mytester:assertlt(torch.norm(o - e), precision, 'mock ' .. name .. ' module failed for batch')
    
      -- test mask zero module now
      local module = nn.MaskZero(recurrent, 1)
      -- non batch forward
      local i = torch.rand(10)
      local e = firstElement(recurrent:forward({i, torch.rand(10), torch.rand(10)}))
      local o = firstElement(module:forward({i, torch.rand(10), torch.rand(10)}))
      mytester:assertgt(torch.norm(i - o), precision, 'error on non batch forward for ' .. name)
      mytester:assertlt(torch.norm(e - o), precision, 'error on non batch forward for ' .. name)
      local i = torch.zeros(10)
      local o = firstElement(module:forward({i, torch.rand(10), torch.rand(10)}))
      mytester:assertlt(torch.norm(i - o), precision, 'error on non batch forward for ' .. name)
      -- batch forward
      local i = torch.rand(5, 10)
      local e = firstElement(recurrent:forward({i, torch.rand(5, 10), torch.rand(5, 10)}))
      local o = firstElement(module:forward({i, torch.rand(5, 10), torch.rand(5, 10)}))
      mytester:assertgt(torch.norm(i - o), precision, 'error on batch forward for ' .. name)
      mytester:assertlt(torch.norm(e - o), precision, 'error on batch forward for ' .. name)
      local i = torch.zeros(5, 10)
      local o = firstElement(module:forward({i, torch.rand(5, 10), torch.rand(5, 10)}))
      mytester:assertlt(torch.norm(i - o), precision, 'error on batch forward for ' .. name)
      local i = torch.Tensor({{0, 0, 0}, {1, 2, 5}})
      -- clone r because it will be update by module:forward call
      local r = firstElement(recurrent:forward({i, torch.rand(2, 3), torch.rand(2, 3)})):clone()
      local o = firstElement(module:forward({i, torch.rand(2, 3), torch.rand(2, 3)}))
      mytester:assertgt(torch.norm(r - o), precision, 'error on batch forward for ' .. name)
      r[1]:zero()
      mytester:assertlt(torch.norm(r - o), precision, 'error on batch forward for ' .. name)

      -- check gradients
      local jac = nn.Jacobian
      local sjac = nn.SparseJacobian
      -- Note: testJacobian doesn't support table inputs or outputs
      -- Use a SplitTable and SelectTable to adapt module
      local module = nn.Sequential()
      module:add(nn.SplitTable(1))
      module:add(nn.MaskZero(recurrent, 1))
      if name == 'lstm' then module:add(nn.SelectTable(1)) end

      local input = torch.rand(name == 'lstm' and 3 or 2, 10)
      local err = jac.testJacobian(module, input)
      mytester:assertlt(err, precision, 'error on state for ' .. name)
      -- IO
      local ferr,berr = jac.testIO(module,input)
      mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err for ' .. name)
      mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err for ' .. name)
      -- batch
      -- rebuild module to avoid correlated tests
      local module = nn.Sequential()
      module:add(nn.SplitTable(1))
      module:add(nn.MaskZero(recurrent, 1))
      if name == 'lstm' then module:add(nn.SelectTable(1)) end

      local input = torch.rand(name == 'lstm' and 3 or 2, 5, 10)
      local err = jac.testJacobian(module,input)
      mytester:assertlt(err, precision, 'batch error on state for ' .. name)

      -- full test on convolution and linear modules
      local module = nn.Sequential() :add( nn.ParallelTable() :add(nn.SpatialConvolution(1,2,3,3)) :add(nn.Linear(100,2)) )
      --module = module:float()
      local batchNum = 5
      local input = {torch.rand(batchNum,1,10,10), torch.rand(batchNum,100)}
      local zeroRowNum = 2
      for i = 1,#input do
         input[i]:narrow(1,1,zeroRowNum):zero()
      end
      --module = nn.MaskZero(module, 3)
      local output = module:forward(input)
      for i = 1,#input do
         for j = 1,batchNum do
            local rmi = input[i][j]:view(-1) -- collapse dims
            local vectorDim = rmi:dim()
            local rn = rmi.new()
            rn:norm(rmi, 2, vectorDim)
            local err = rn[1]
            if j<=zeroRowNum then
               -- check zero outputs
               mytester:assertlt(err, precision, 'batch ' ..i.. ':' ..j.. ' error on state for ' .. name)
            else
               -- check non-zero outputs
               mytester:assertgt(err, precision, 'batch ' ..i.. ':' ..j.. ' error on state for ' .. name)
            end
         end
      end
   end
end

function rnntest.TrimZero()
   local recurrents = {['recurrent'] = recurrentModule(), ['lstm'] = lstmModule()}
   -- Note we use lstmModule input signature and firstElement to prevent duplicate code
   for name, recurrent in pairs(recurrents) do
      -- test encapsulated module first
      -- non batch
      local i = torch.rand(10)
      local e = nn.Sigmoid():forward(i)
      local o = firstElement(recurrent:forward({i, torch.zeros(10), torch.zeros(10)}))
      mytester:assertlt(torch.norm(o - e), precision, 'mock ' .. name .. ' failed for non batch')
      -- batch
      local i = torch.rand(5, 10)
      local e = nn.Sigmoid():forward(i)
      local o = firstElement(recurrent:forward({i, torch.zeros(5, 10), torch.zeros(5, 10)}))
      mytester:assertlt(torch.norm(o - e), precision, 'mock ' .. name .. ' module failed for batch')
    
      -- test mask zero module now
      local module = nn.TrimZero(recurrent, 1)
      local module2 = nn.MaskZero(recurrent, 1)
      -- non batch forward
      local i = torch.rand(10)
      local e = firstElement(recurrent:forward({i, torch.rand(10), torch.rand(10)}))
      local o = firstElement(module:forward({i, torch.rand(10), torch.rand(10)}))
      local o2 = firstElement(module2:forward({i, torch.rand(10), torch.rand(10)}))
      mytester:assertgt(torch.norm(i - o), precision, 'error on non batch forward for ' .. name)
      mytester:assertlt(torch.norm(e - o), precision, 'error on non batch forward for ' .. name)
      mytester:assertlt(torch.norm(o2 - o), precision, 'error on non batch forward for ' .. name)
      local i = torch.zeros(10)
      local o = firstElement(module:forward({i, torch.rand(10), torch.rand(10)}))
      local o2 = firstElement(module2:forward({i, torch.rand(10), torch.rand(10)}))
      mytester:assertlt(torch.norm(i - o), precision, 'error on non batch forward for ' .. name)
      mytester:assertlt(torch.norm(o2 - o), precision, 'error on non batch forward for ' .. name)
      -- batch forward
      local i = torch.rand(5, 10)
      local e = firstElement(recurrent:forward({i, torch.rand(5, 10), torch.rand(5, 10)}))
      local o = firstElement(module:forward({i, torch.rand(5, 10), torch.rand(5, 10)}))
      local o2 = firstElement(module2:forward({i, torch.rand(5, 10), torch.rand(5, 10)}))
      mytester:assertgt(torch.norm(i - o), precision, 'error on batch forward for ' .. name)
      mytester:assertlt(torch.norm(e - o), precision, 'error on batch forward for ' .. name)
      mytester:assertlt(torch.norm(o2 - o), precision, 'error on batch forward for ' .. name)
      local i = torch.zeros(5, 10)
      local o = firstElement(module:forward({i, torch.rand(5, 10), torch.rand(5, 10)}))
      local o2 = firstElement(module2:forward({i, torch.rand(5, 10), torch.rand(5, 10)}))
      mytester:assertlt(torch.norm(i - o), precision, 'error on batch forward for ' .. name)
      mytester:assertlt(torch.norm(o2 - o), precision, 'error on batch forward for ' .. name)
      local i = torch.Tensor({{0, 0, 0}, {1, 2, 5}})
      -- clone r because it will be update by module:forward call
      local r = firstElement(recurrent:forward({i, torch.rand(2, 3), torch.rand(2, 3)})):clone()
      local o = firstElement(module:forward({i, torch.rand(2, 3), torch.rand(2, 3)}))
      local o2 = firstElement(module2:forward({i, torch.rand(2, 3), torch.rand(2, 3)}))
      mytester:assertgt(torch.norm(r - o), precision, 'error on batch forward for ' .. name)
      r[1]:zero()
      mytester:assertlt(torch.norm(r - o), precision, 'error on batch forward for ' .. name)
      mytester:assertlt(torch.norm(o2 - o), precision, 'error on batch forward for ' .. name)

      -- check gradients
      local jac = nn.Jacobian
      local sjac = nn.SparseJacobian
      -- Note: testJacobian doesn't support table inputs or outputs
      -- Use a SplitTable and SelectTable to adapt module
      local module = nn.Sequential()
      module:add(nn.SplitTable(1))
      module:add(nn.TrimZero(recurrent, 1))
      if name == 'lstm' then module:add(nn.SelectTable(1)) end

      local input = torch.rand(name == 'lstm' and 3 or 2, 10)
      local err = jac.testJacobian(module, input)
      mytester:assertlt(err, precision, 'error on state for ' .. name)
      -- IO
      local ferr,berr = jac.testIO(module,input)
      mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err for ' .. name)
      mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err for ' .. name)
      -- batch
      -- rebuild module to avoid correlated tests
      local module = nn.Sequential()
      module:add(nn.SplitTable(1))
      module:add(nn.TrimZero(recurrent, 1))
      if name == 'lstm' then module:add(nn.SelectTable(1)) end

      local input = torch.rand(name == 'lstm' and 3 or 2, 5, 10)
      local err = jac.testJacobian(module,input)
      mytester:assertlt(err, precision, 'batch error on state for ' .. name)

      -- full test on convolution and linear modules
      local module = nn.Sequential() :add( nn.ParallelTable() :add(nn.SpatialConvolution(1,2,3,3)) :add(nn.Linear(100,2)) )
      local batchNum = 5
      local input = {torch.rand(batchNum,1,10,10), torch.rand(batchNum,100)}
      local zeroRowNum = 2
      for i = 1,#input do
         input[i]:narrow(1,1,zeroRowNum):zero()
      end
      local output = module:forward(input)
      for i = 1,#input do
         for j = 1,batchNum do
            local rmi = input[i][j]:view(-1) -- collapse dims
            local vectorDim = rmi:dim()
            local rn = rmi.new()
            rn:norm(rmi, 2, vectorDim)
            local err = rn[1]
            if j<=zeroRowNum then
               -- check zero outputs
               mytester:assertlt(err, precision, 'batch ' ..i.. ':' ..j.. ' error on state for ' .. name)
            else
               -- check non-zero outputs
               mytester:assertgt(err, precision, 'batch ' ..i.. ':' ..j.. ' error on state for ' .. name)
            end
         end
      end
   end

   -- check to have the same loss
   local rnn_size = 8
   local vocabSize = 7
   local word_embedding_size = 10
   local x = torch.Tensor{{{1,2,3},{0,4,5},{0,0,7}},
                          {{1,2,3},{2,4,5},{0,0,7}},
                          {{1,2,3},{2,4,5},{3,0,7}}}
   local t = torch.ceil(torch.rand(x:size(2)))
   local rnns = {'FastLSTM','GRU'}
   local methods = {'maskZero', 'trimZero'}
   local loss = torch.Tensor(#rnns, #methods, 3)

   for ir,arch in pairs(rnns) do
      local rnn = nn[arch](word_embedding_size, rnn_size)
      local model = nn.Sequential()
                  :add(nn.LookupTableMaskZero(vocabSize, word_embedding_size))
                  :add(nn.SplitTable(2))
                  :add(nn.Sequencer(rnn))
                  :add(nn.SelectTable(-1))
                  :add(nn.Linear(rnn_size, 10))
      model:getParameters():uniform(-0.1, 0.1)
      local criterion = nn.CrossEntropyCriterion()
      local models = {}
      for j=1,#methods do
         table.insert(models, model:clone())
      end
      for im,method in pairs(methods) do
         -- print('-- '..arch..' with '..method)
         model = models[im]
         local rnn = model:get(3).module
         rnn[method](rnn, 1)
         -- sys.tic()
         for i=1,loss:size(3) do
            model:zeroGradParameters()
            local y = model:forward(x[i])
            loss[ir][im][i] = criterion:forward(y,t)
            -- print('loss:', loss[ir][im][i])
            local dy = criterion:backward(y,t)
            model:backward(x[i], dy)
            local w,dw = model:parameters()
            model:updateParameters(.5)
         end
         -- elapse = sys.toc()
         -- print('elapse time:', elapse)   
      end
   end
   mytester:assertTensorEq(loss:select(2,1), loss:select(2,2), 0.0000001, "loss check")
end

function rnntest.AbstractRecurrent_maskZero()
   local inputs = {}

   local input = torch.zeros(4,4,10)
   local sequence = torch.randn(4,10)
   input:select(2,1):select(1,4):copy(sequence[1])
   input:select(2,2):narrow(1,3,2):copy(sequence:narrow(1,1,2))
   input:select(2,3):narrow(1,2,3):copy(sequence:narrow(1,1,3))
   input:select(2,4):copy(sequence)


   for i=1,4 do
      table.insert(inputs, input[i])
   end


   local function testmask(rnn)
      local seq = nn.Sequencer(rnn:maskZero(1))

      local outputs = seq:forward(inputs)

      mytester:assert(math.abs(outputs[1]:narrow(1,1,3):sum()) < 0.0000001, torch.type(rnn).." mask zero 1 err")
      mytester:assert(math.abs(outputs[2]:narrow(1,1,2):sum()) < 0.0000001, torch.type(rnn).." mask zero 2 err")
      mytester:assert(math.abs(outputs[3]:narrow(1,1,1):sum()) < 0.0000001, torch.type(rnn).." mask zero 3 err")
      
      mytester:assertTensorEq(outputs[1][4], outputs[2][3], 0.0000001, torch.type(rnn).." mask zero err")
      mytester:assertTensorEq(outputs[1][4], outputs[3][2], 0.0000001, torch.type(rnn).." mask zero err")
      mytester:assertTensorEq(outputs[1][4], outputs[4][1], 0.0000001, torch.type(rnn).." mask zero err")
      
      mytester:assertTensorEq(outputs[2][4], outputs[3][3], 0.0000001, torch.type(rnn).." mask zero err")
      mytester:assertTensorEq(outputs[2][4], outputs[4][2], 0.0000001, torch.type(rnn).." mask zero err")
      
      mytester:assertTensorEq(outputs[3][4], outputs[4][3], 0.0000001, torch.type(rnn).." mask zero err")
   end
   
   local rm = nn.Sequential()
      :add(nn.ParallelTable()
         :add(nn.Linear(10,10))
         :add(nn.Linear(10,10)))
      :add(nn.CAddTable())
      :add(nn.Sigmoid())
   
   testmask(nn.Recurrence(rm, 10, 1))
   testmask(nn.LSTM(10,10))
   testmask(nn.GRU(10,10))
   
   local success, err = pcall(function() nn.Recurrent(10, nn.Linear(10,10), nn.Linear(10,10)):maskZero() end)
   mytester:assert(not success, "nn.Recurrent supposed to give error on maskZero()")
end

function rnntest.AbstractRecurrent_trimZero()
   local inputs = {}

   local input = torch.zeros(4,4,10)
   local sequence = torch.randn(4,10)
   input:select(2,1):select(1,4):copy(sequence[1])
   input:select(2,2):narrow(1,3,2):copy(sequence:narrow(1,1,2))
   input:select(2,3):narrow(1,2,3):copy(sequence:narrow(1,1,3))
   input:select(2,4):copy(sequence)


   for i=1,4 do
      table.insert(inputs, input[i])
   end


   local function testmask(rnn)
      local seq = nn.Sequencer(rnn:trimZero(1))

      local outputs = seq:forward(inputs)

      mytester:assert(math.abs(outputs[1]:narrow(1,1,3):sum()) < 0.0000001, torch.type(rnn).." mask zero 1 err")
      mytester:assert(math.abs(outputs[2]:narrow(1,1,2):sum()) < 0.0000001, torch.type(rnn).." mask zero 2 err")
      mytester:assert(math.abs(outputs[3]:narrow(1,1,1):sum()) < 0.0000001, torch.type(rnn).." mask zero 3 err")
      
      mytester:assertTensorEq(outputs[1][4], outputs[2][3], 0.0000001, torch.type(rnn).." mask zero err")
      mytester:assertTensorEq(outputs[1][4], outputs[3][2], 0.0000001, torch.type(rnn).." mask zero err")
      mytester:assertTensorEq(outputs[1][4], outputs[4][1], 0.0000001, torch.type(rnn).." mask zero err")
      
      mytester:assertTensorEq(outputs[2][4], outputs[3][3], 0.0000001, torch.type(rnn).." mask zero err")
      mytester:assertTensorEq(outputs[2][4], outputs[4][2], 0.0000001, torch.type(rnn).." mask zero err")
      
      mytester:assertTensorEq(outputs[3][4], outputs[4][3], 0.0000001, torch.type(rnn).." mask zero err")
   end
   
   local rm = nn.Sequential()
      :add(nn.ParallelTable()
         :add(nn.Linear(10,10))
         :add(nn.Linear(10,10)))
      :add(nn.CAddTable())
      :add(nn.Sigmoid())
   
   testmask(nn.Recurrence(rm, 10, 1))
   testmask(nn.LSTM(10,10))
   testmask(nn.GRU(10,10))
   
   local success, err = pcall(function() nn.Recurrent(10, nn.Linear(10,10), nn.Linear(10,10)):trimZero() end)
   mytester:assert(not success, "nn.Recurrent supposed to give error on trimZero()")
end

local function forwardbackward(module, criterion, input, expected)
  local output = module:forward(input)
  criterion:forward(output, expected)
  module:zeroGradParameters()
  module:backward(input, criterion:backward(output, expected))
  module:updateParameters(1)
  return output
end

function rnntest.LookupTableMaskZero()
   local batchSize = math.random(5, 10)
   local outputSize = math.random(5, 10)
   local indexSize = batchSize

   local m1 = nn.LookupTable(indexSize, outputSize)
   local m2 = nn.LookupTableMaskZero(indexSize, outputSize)
   m2.weight:narrow(1, 2, indexSize):copy(m1.weight)
   local criterion = nn.MSECriterion()
   -- Zero padding will change averaging
   -- TODO create Criterion supporting padding
   criterion.sizeAverage = false

   -- verify that LookupTables have the same results (modulo zero padding)
   -- through multiple backpropagations
   for i=1, 10 do
      local input1 = torch.randperm(batchSize)
      local input2 = torch.zeros(batchSize + 2)
      input2:narrow(1, 1, batchSize):copy(input1)
      local expected1 = torch.rand(batchSize, outputSize)
      local expected2 = torch.rand(batchSize + 2, outputSize)
      expected2:narrow(1, 1, batchSize):copy(expected1)
      local o1 = forwardbackward(m1, criterion, input1, expected1)
      local o2 = forwardbackward(m2, criterion, input2, expected2)
      -- output modulo zero index should be the same
      mytester:assertlt(torch.norm(o1 - o2:narrow(1, 1, batchSize), 2), precision)
      -- zero index should yield zero vector
      mytester:assertlt(o2[batchSize + 1]:norm(2), precision)
      mytester:assertlt(o2[batchSize + 2]:norm(2), precision)
      -- weights should be equivalent
      mytester:assertlt(torch.norm(m1.weight - m2.weight:narrow(1, 2, indexSize), 2), precision)
  end
end

function rnntest.MaskZeroCriterion()
   local batchSize = 3
   local nClass = 10
   local input = torch.randn(batchSize, nClass)
   local target = torch.LongTensor(batchSize):random(1,nClass)
   
   local nll = nn.ClassNLLCriterion()
   local mznll = nn.MaskZeroCriterion(nll, 1)
   
   -- test that it works when nothing to mask
   local err = mznll:forward(input, target)
   local gradInput = mznll:backward(input, target):clone()
   
   local err2 = nll:forward(input, target)
   local gradInput2 = nll:backward(input, target)
   
   mytester:assert(math.abs(err - err2) < 0.0000001, "MaskZeroCriterion No-mask fwd err")
   mytester:assertTensorEq(gradInput, gradInput2, 0.0000001, "MaskZeroCriterion No-mask bwd err")
   
   -- test that it works when last row to mask
   input[batchSize]:zero()
   target[batchSize] = 0
   
   local err = mznll:forward(input, target)
   local gradInput = mznll:backward(input, target):clone()
   
   local input2 = input:narrow(1,1,batchSize-1)
   local target2 = target:narrow(1,1,batchSize-1)
   local err2 = nll:forward(input2, target2)
   local gradInput2 = nll:backward(input2, target2)
   
   mytester:assert(gradInput[batchSize]:sum() == 0, "MaskZeroCriterion last-mask bwd zero err")
   mytester:assert(math.abs(err - err2) < 0.0000001, "MaskZeroCriterion last-mask fwd err")
   mytester:assertTensorEq(gradInput:narrow(1,1,batchSize-1), gradInput2, 0.0000001, "MaskZeroCriterion last-mask bwd err") 
   
   -- test type-casting
   mznll:float()
   local input3 = input:float()
   local err3 = mznll:forward(input3, target)
   local gradInput3 = mznll:backward(input3, target):clone()
   
   mytester:assert(math.abs(err3 - err) < 0.0000001, "MaskZeroCriterion cast fwd err")
   mytester:assertTensorEq(gradInput3, gradInput:float(), 0.0000001, "MaskZeroCriterion cast bwd err")
   
   if pcall(function() require 'cunn' end) then
      -- test cuda
      mznll:cuda()
      local input4 = input:cuda()
      local target4 = target:cuda()
      local err4 = mznll:forward(input4, target4)
      local gradInput4 = mznll:backward(input4, target4):clone()
      
      mytester:assert(math.abs(err4 - err) < 0.0000001, "MaskZeroCriterion cuda fwd err")
      mytester:assertTensorEq(gradInput4:float(), gradInput3, 0.0000001, "MaskZeroCriterion cuda bwd err")
   end
   
   -- issue 128
   local input, target=torch.zeros(3,2), torch.Tensor({1,2,1}) -- batch size 3, 2 classes
   local crit=nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1)
   -- output from a masked module gives me all zeros
   local loss = crit:forward(input, target)
   mytester:assert(loss == 0, "MaskZeroCriterion all zeros fwd err")
   
   local gradInput = crit:backward(input, target)
   mytester:assert(gradInput:sum() == 0, "MaskZeroCriterion all zeros bwd err")
end

function rnntest.MaskZero_where()
   local hiddensize = 5
   local batchsize = 4
   local seqlen = 7
   
   local rnn = nn.FastLSTM(hiddensize, hiddensize)
   rnn:maskZero(1)
   rnn = nn.Sequencer(rnn)
   
   -- is there any difference between start and end padding?
   
   local inputs, gradOutputs = {}, {}
   
   for i=1,seqlen do
      if i==1 then
         inputs[i] = torch.zeros(batchsize, hiddensize)
      else
         inputs[i] = torch.randn(batchsize, hiddensize)
      end
      gradOutputs[i] = torch.randn(batchsize, hiddensize)
   end
   
   local outputs = rnn:forward(inputs)
   rnn:zeroGradParameters()
   local gradInputs = rnn:backward(inputs, gradOutputs)
   
   local params, gradParams = rnn:parameters()
   local params2, gradParams2 = {}, {}
   for i=1,#params do
      params2[i] = params[i]:clone()
      gradParams2[i] = gradParams[i]:clone()
   end
   
   local outputs2, gradInputs2 = {}, {}
   for i=1,seqlen do
      outputs2[i] = outputs[i]:clone()
      gradInputs2[i] = gradInputs[i]:clone()
   end
   inputs[seqlen] = table.remove(inputs, 1)
   gradOutputs[seqlen] = table.remove(gradOutputs, 1)

   rnn:forget()
   local outputs = rnn:forward(inputs)
   rnn:zeroGradParameters()
   local gradInputs = rnn:backward(inputs, gradOutputs)
   
   for i=1,seqlen-1 do
      mytester:assertTensorEq(outputs[i], outputs2[i+1], 0.000001)
      mytester:assertTensorEq(gradInputs[i], gradInputs2[i+1], 0.000001)
   end
   
   for i=1,#params do
      mytester:assertTensorEq(gradParams2[i], gradParams[i], 0.000001)
   end
   
   -- how about in the middle? is it the same as a forget() in between
   
   local inputs, gradOutputs = {}, {}
   
   for i=1,seqlen do
      if i==4 then
         inputs[i] = torch.zeros(batchsize, hiddensize)
      else
         inputs[i] = torch.randn(batchsize, hiddensize)
      end
      gradOutputs[i] = torch.randn(batchsize, hiddensize)
   end
   
   rnn:forget()
   local rnn2 = rnn:clone()
   
   local outputs = rnn:forward(inputs)
   rnn:zeroGradParameters()
   local gradInputs = rnn:backward(inputs, gradOutputs)
   
   local _ = require 'moses'
   local inputs1 = _.first(inputs, 3)
   local gradOutputs1 = _.first(gradOutputs, 3)
   
   local outputs1 = rnn2:forward(inputs1)
   rnn2:zeroGradParameters()
   local gradInputs1 = rnn2:backward(inputs1, gradOutputs1)
   
   for i=1,3 do
      mytester:assertTensorEq(outputs[i], outputs1[i], 0.000001)
      mytester:assertTensorEq(gradInputs[i], gradInputs1[i], 0.000001)
   end
   
   rnn2:forget() -- forget at mask zero
   
   local inputs2 = _.last(inputs, 3)
   local gradOutputs2 = _.last(gradOutputs, 3)
   
   local outputs2 = rnn2:forward(inputs2)
   local gradInputs2 = rnn2:backward(inputs2, gradOutputs2)
   
   local params, gradParams = rnn:parameters()
   local params2, gradParams2 = rnn2:parameters()
   
   for i=1,#params do
      mytester:assertTensorEq(gradParams2[i], gradParams[i], 0.000001)
   end
   
   for i=1,3 do
      mytester:assertTensorEq(outputs[i+4], outputs2[i], 0.000001)
      mytester:assertTensorEq(gradInputs[i+4], gradInputs2[i], 0.000001)
   end
end

function rnntest.issue129()
   -- test without rnn
   local model1 = nn.Sequential()
   model1:add(nn.SpatialBatchNormalization(2))

   local input = torch.randn(4, 2, 64, 64)  -- batch_size X channels X height X width

   model1:training()
   local output
   for i=1, 1000 do  -- to run training enough times
      output = model1:forward(input):clone()
   end

   model1:evaluate()
   local output2 = model1:forward(input):clone()

   mytester:assertTensorEq(output, output2,  0.0002, "issue 129 err")
   
   -- test with rnn
   local normalize = nn.Sequential()
   normalize:add(nn.SpatialBatchNormalization(2))

   local model = nn.Sequential()
   model:add(nn.SplitTable(1))  -- since sequencer expects table as input
   model:add(nn.Sequencer(normalize))  -- wrapping batch-normalization in a sequencer
   model:add(nn.JoinTable(1))  -- since sequencer outputs table

   input:resize(1, 4, 2, 64, 64)  -- time_step X batch_size X channels X height X width

   model:training()

   local output
   for i=1, 1000 do  -- to run training enough times
      output = model:forward(input):clone()
   end
   
   mytester:assertTensorEq(model1:get(1).running_mean, model:get(2).module.sharedClones[1].modules[1].running_mean, 0.000001)
   mytester:assertTensorEq(model:get(2).module.sharedClones[1].modules[1].running_mean, model:get(2).module.recurrentModule.modules[1].running_mean, 0.0000001)

   model:evaluate()
   local output2 = model:forward(input):clone()

   mytester:assertTensorEq(output, output2,  0.0002, "issue 129 err")
end

function rnntest.issue170()
   torch.manualSeed(123)

   local rnn_size = 8
   local vocabSize = 7
   local word_embedding_size = 10
   local rnn_dropout = .00000001  -- dropout ignores manualSeed()
   local mono = true
   local x = torch.Tensor{{1,2,3},{0,4,5},{0,0,7}}
   local t = torch.ceil(torch.rand(x:size(2)))
   local rnns = {'GRU'}
   local methods = {'maskZero', 'trimZero'}
   local loss = torch.Tensor(#rnns, #methods,1)

   for ir,arch in pairs(rnns) do
      local rnn = nn[arch](word_embedding_size, rnn_size, nil, rnn_dropout)
      local model = nn.Sequential()
                  :add(nn.LookupTableMaskZero(vocabSize, word_embedding_size))
                  :add(nn.SplitTable(2))
                  :add(nn.Sequencer(rnn))
                  :add(nn.SelectTable(-1))
                  :add(nn.Linear(rnn_size, 10))
      model:getParameters():uniform(-0.1, 0.1)
      local criterion = nn.CrossEntropyCriterion()
      local models = {}
      for j=1,#methods do
         table.insert(models, model:clone())
      end
      for im,method in pairs(methods) do
         model = models[im]
         local rnn = model:get(3).module
         rnn[method](rnn, 1)
         for i=1,loss:size(3) do
            model:zeroGradParameters()
            local y = model:forward(x)
            loss[ir][im][i] = criterion:forward(y,t)
            local dy = criterion:backward(y,t)
            model:backward(x, dy)
            local w,dw = model:parameters()
            model:updateParameters(.5)
         end
      end
   end
   mytester:assertTensorEq(loss:select(2,1), loss:select(2,2), 0.0000001, "loss check")
end

function rnntest.encoderdecoder()
   torch.manualSeed(123)
   
   local opt = {}
   opt.learningRate = 0.1
   opt.hiddenSz = 2
   opt.vocabSz = 5
   opt.inputSeqLen = 3 -- length of the encoded sequence

   --[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
   local function forwardConnect(encLSTM, decLSTM)
     decLSTM.userPrevOutput = nn.rnn.recursiveCopy(decLSTM.userPrevOutput, encLSTM.outputs[opt.inputSeqLen])
     decLSTM.userPrevCell = nn.rnn.recursiveCopy(decLSTM.userPrevCell, encLSTM.cells[opt.inputSeqLen])
   end

   --[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
   local function backwardConnect(encLSTM, decLSTM)
     encLSTM.userNextGradCell = nn.rnn.recursiveCopy(encLSTM.userNextGradCell, decLSTM.userGradPrevCell)
     encLSTM.gradPrevOutput = nn.rnn.recursiveCopy(encLSTM.gradPrevOutput, decLSTM.userGradPrevOutput)
   end

   -- Encoder
   local enc = nn.Sequential()
   enc:add(nn.LookupTable(opt.vocabSz, opt.hiddenSz))
   enc:add(nn.SplitTable(1, 2)) -- works for both online and mini-batch mode
   local encLSTM = nn.LSTM(opt.hiddenSz, opt.hiddenSz)
   enc:add(nn.Sequencer(encLSTM))
   enc:add(nn.SelectTable(-1))

   -- Decoder
   local dec = nn.Sequential()
   dec:add(nn.LookupTable(opt.vocabSz, opt.hiddenSz))
   dec:add(nn.SplitTable(1, 2)) -- works for both online and mini-batch mode
   local decLSTM = nn.LSTM(opt.hiddenSz, opt.hiddenSz)
   dec:add(nn.Sequencer(decLSTM))
   dec:add(nn.Sequencer(nn.Linear(opt.hiddenSz, opt.vocabSz)))
   dec:add(nn.Sequencer(nn.LogSoftMax()))

   local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

   local encParams, encGradParams = enc:getParameters()
   local decParams, decGradParams = dec:getParameters()

   enc:zeroGradParameters()
   dec:zeroGradParameters()

   -- Some example data (batchsize = 2)
   local encInSeq = torch.Tensor({{1,2,3},{3,2,1}}) 
   local decInSeq = torch.Tensor({{1,2,3,4},{4,3,2,1}})
   local decOutSeq = torch.Tensor({{2,3,4,1},{1,2,4,3}})
   decOutSeq = nn.SplitTable(1, 1):forward(decOutSeq)

   -- Forward pass
   local encOut = enc:forward(encInSeq)
   forwardConnect(encLSTM, decLSTM)
   local decOut = dec:forward(decInSeq)
   local Edec = criterion:forward(decOut, decOutSeq)

   -- Backward pass
   local gEdec = criterion:backward(decOut, decOutSeq)
   dec:backward(decInSeq, gEdec)
   backwardConnect(encLSTM, decLSTM)
   local zeroTensor = torch.Tensor(2):zero()
   enc:backward(encInSeq, zeroTensor)

   local function numgradtest()
      -- Here, we do a numerical gradient check to make sure the coupling is correct:
      local eps = 1e-5

      local decGP_est, encGP_est = torch.DoubleTensor(decGradParams:size()), torch.DoubleTensor(encGradParams:size())

      -- Easy function to do forward pass over coupled network and get error
      local function forwardPass()
         local encOut = enc:forward(encInSeq)
         forwardConnect(encLSTM, decLSTM)
         local decOut = dec:forward(decInSeq)
         local E = criterion:forward(decOut, decOutSeq)
         return E
      end

      -- Check encoder
      for i = 1, encGradParams:size(1) do
         -- Forward with \theta+eps
         encParams[i] = encParams[i] + eps
         local C1 = forwardPass()
         -- Forward with \theta-eps
         encParams[i] = encParams[i] - 2 * eps
         local C2 = forwardPass()

         encParams[i] = encParams[i] + eps
         encGP_est[i] = (C1 - C2) / (2 * eps)
      end
      mytester:assertTensorEq(encGradParams, encGP_est, eps, "Numerical gradient check for encoder failed")

      -- Check decoder
      for i = 1, decGradParams:size(1) do
         -- Forward with \theta+eps
         decParams[i] = decParams[i] + eps
         local C1 = forwardPass()
         -- Forward with \theta-eps
         decParams[i] = decParams[i] - 2 * eps
         local C2 = forwardPass()

         decParams[i] = decParams[i] + eps
         decGP_est[i] = (C1 - C2) / (2 * eps)
      end
      mytester:assertTensorEq(decGradParams, decGP_est, eps, "Numerical gradient check for decoder failed")
   end
   
   numgradtest()
   
   encGradParams:zero()
   decGradParams:zero()

   -- issue 142

   -- batchsize = 3
   
   encInSeq = torch.Tensor({{1,2,3},{3,2,1},{1,3,5}}) 
   decInSeq = torch.Tensor({{1,2,3,4},{4,3,2,1},{1,3,5,1}})
   decOutSeq = torch.Tensor({{2,3,4,1},{1,2,4,3},{1,2,5,3}})
   decOutSeq = nn.SplitTable(1, 1):forward(decOutSeq)

   -- Forward pass
   local encOut = enc:forward(encInSeq)
   forwardConnect(encLSTM, decLSTM)
   local decOut = dec:forward(decInSeq)
   local Edec = criterion:forward(decOut, decOutSeq)

   -- Backward pass
   local gEdec = criterion:backward(decOut, decOutSeq)
   dec:backward(decInSeq, gEdec)
   backwardConnect(encLSTM, decLSTM)
   local zeroTensor = torch.Tensor(2):zero()
   enc:backward(encInSeq, zeroTensor)
   
   numgradtest()
end

function rnntest.reinforce()
   -- test that AbstractRecurrent:reinforce(rewards) words
   local seqLen = 4
   local batchSize = 3
   local rewards = {}
   for i=1,seqLen do
      rewards[i] = torch.randn(batchSize)
   end
   local rf = nn.ReinforceNormal(0.1)
   local rnn = nn.Recursor(rf)
   rnn:reinforce(rewards)
   for i=1,seqLen do
      local rm = rnn:getStepModule(i)
      mytester:assertTensorEq(rm.reward, rewards[i], 0.000001, "Reinforce error")
   end
end

function rnntest.rnnlm()
   if not pcall(function() require 'nngraph' end) then
      return
   end
   
   local vocabsize = 100
   local opt = {
      seqlen = 5,
      batchsize = 3,
      hiddensize = {20,20},
      lstm = true
   }
   
   local lm = nn.Sequential()

   -- input layer (i.e. word embedding space)
   local lookup = nn.LookupTable(vocabsize, opt.hiddensize[1])
   lookup.maxnormout = -1 -- prevent weird maxnormout behaviour
   lm:add(lookup) -- input is seqlen x batchsize
   lm:add(nn.SplitTable(1)) -- tensor to table of tensors

   -- rnn layers
   local stepmodule = nn.Sequential() -- applied at each time-step
   local inputsize = opt.hiddensize[1]
   local rnns = {}
   for i,hiddensize in ipairs(opt.hiddensize) do 
      nn.FastLSTM.usenngraph = true -- faster
      local rnn = nn.FastLSTM(inputsize, hiddensize)
      table.insert(rnns, rnn)
      stepmodule:add(rnn)
      inputsize = hiddensize
   end
   nn.FastLSTM.usenngraph = false
   -- output layer
   local linear = nn.Linear(inputsize, vocabsize)
   stepmodule:add(linear)
   stepmodule:add(nn.LogSoftMax())
   lm:add(nn.Sequencer(stepmodule))
   lm:remember('both')
   
   
   --[[ multiple sequencer ]]--
   
   
   local lm2 = nn.Sequential()

   local inputSize = opt.hiddensize[1]
   for i,hiddenSize in ipairs(opt.hiddensize) do 
      local rnn = nn.Sequencer(rnns[i]:clone())
      lm2:add(rnn)
      inputSize = hiddenSize
   end

   -- input layer (i.e. word embedding space)
   lm2:insert(nn.SplitTable(1,2), 1) -- tensor to table of tensors
   local lookup2 = lookup:clone()
   lookup.maxOutNorm = -1 -- disable maxParamNorm on the lookup table
   lm2:insert(lookup2, 1)

   -- output layer
   local softmax = nn.Sequential()
   softmax:add(linear:clone())
   softmax:add(nn.LogSoftMax())
   lm2:add(nn.Sequencer(softmax))
   lm2:remember('both')
   
   -- compare
   
   for j=1,2 do
      local inputs = torch.LongTensor(opt.seqlen, opt.batchsize):random(1,vocabsize)
      local gradOutputs = torch.randn(opt.seqlen, opt.batchsize, vocabsize)
      local gradOutputs = nn.SplitTable(1):forward(gradOutputs)
      
      local params, gradParams = lm:parameters()
      local params2, gradParams2 = lm2:parameters()
      
      lm:training()
      lm2:training()
      for i=1,4 do
         local outputs = lm:forward(inputs)
         lm:zeroGradParameters()
         local gradInputs = lm:backward(inputs, gradOutputs)
         lm:updateParameters(0.1)
         
         local inputs2 = inputs:transpose(1,2)
         local outputs2 = lm2:forward(inputs2)
         lm2:zeroGradParameters()
         local gradInputs2 = lm2:backward(inputs2, gradOutputs)
         lm2:updateParameters(0.1)
         
         mytester:assertTensorEq(gradInputs, gradInputs2, 0.0000001, "gradInputs err")
         for k=1,#outputs2 do
            mytester:assertTensorEq(outputs2[k], outputs[k], 0.0000001, "outputs err "..k)
         end
         
         for k=1,#params do
            mytester:assertTensorEq(gradParams[k], gradParams2[k], 0.0000001, "gradParam err "..k)
            mytester:assertTensorEq(params[k], params2[k], 0.0000001, "param err"..k)
         end
      end
      
      lm:evaluate()
      lm2:evaluate()
      for i=1,3 do
         local outputs = lm:forward(inputs)
         
         local inputs2 = inputs:transpose(1,2)
         local outputs2 = lm2:forward(inputs2)
         
         for k=1,#outputs2 do
            mytester:assertTensorEq(outputs2[k], outputs[k], 0.0000001, "outputs err "..k)
         end
      end
   end
end

function rnn.test(tests, benchmark_)
   mytester = torch.Tester()
   benchmark = benchmark_
   mytester:add(rnntest)
   math.randomseed(os.time())
   mytester:run(tests)
end
