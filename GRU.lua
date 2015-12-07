------------------------------------------------------------------------
--[[ GRU ]]--
-- Gated Recurrent Units architecture.
-- http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-gruGRU-rnn-with-python-and-theano/
-- Expects 1D or 2D input.
-- The first input in sequence uses zero value for cell and hidden state
------------------------------------------------------------------------
assert(not nn.GRU, "update nnx package : luarocks install nnx")
local GRU, parent = torch.class('nn.GRU', 'nn.AbstractRecurrent')

function GRU:__init(inputSize, outputSize, rho)
   parent.__init(self, rho or 9999)
   self.inputSize = inputSize
   self.outputSize = outputSize   
   -- build the model
   self.recurrentModule = self:buildModel()
   -- make it work with nn.Container
   self.modules[1] = self.recurrentModule
   self.sharedClones[1] = self.recurrentModule 
   
   -- for output(0), cell(0) and gradCell(T)
   self.zeroTensor = torch.Tensor() 
   
   self.cells = {}
   self.gradCells = {}
end

-------------------------- factory methods -----------------------------
function GRU:buildModel()
   -- input : {input, prevOutput}
   -- output : {output}
   
   -- Calculate all four gates in one go : input, hidden, forget, output
   self.i2g = nn.Linear(self.inputSize, 2*self.outputSize)
   self.o2g = nn.LinearNoBias(self.outputSize, 2*self.outputSize)

   local para = nn.ParallelTable():add(self.i2g):add(self.o2g)
   local gates = nn.Sequential()
   gates:add(para)
   gates:add(nn.CAddTable())

   -- Reshape to (batch_size, n_gates, hid_size)
   -- Then slize the n_gates dimension, i.e dimension 2
   gates:add(nn.Reshape(2,self.outputSize))
   gates:add(nn.SplitTable(1,2))
   local transfer = nn.ParallelTable()
   transfer:add(nn.Sigmoid()):add(nn.Sigmoid())
   gates:add(transfer)

   local concat = nn.ConcatTable()
   concat:add(nn.Identity()):add(gates)
   local seq = nn.Sequential()
   seq:add(concat)
   seq:add(nn.FlattenTable()) -- x(t), s(t-1), r, z

   -- Rearrange to x(t), s(t-1), r, z, s(t-1)
   local concat = nn.ConcatTable()  -- 
   concat:add(nn.NarrowTable(1,4)):add(nn.SelectTable(2))
   seq:add(concat):add(nn.FlattenTable())

   -- h
   local hidden = nn.Sequential()
   local concat = nn.ConcatTable()
   local t1 = nn.Sequential()
   t1:add(nn.SelectTable(1)):add(nn.Linear(self.inputSize, self.outputSize))
   local t2 = nn.Sequential()
   t2:add(nn.NarrowTable(2,2)):add(nn.CMulTable()):add(nn.LinearNoBias(self.outputSize, self.outputSize))
   concat:add(t1):add(t2)
   hidden:add(concat):add(nn.CAddTable()):add(nn.Tanh())
   
   local z1 = nn.Sequential()
   z1:add(nn.SelectTable(4))
   z1:add(nn.SAdd(-1, true))  -- Scalar add & negation

   local z2 = nn.Sequential()
   z2:add(nn.NarrowTable(4,2))
   z2:add(nn.CMulTable())

   local o1 = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(hidden):add(z1)
   o1:add(concat):add(nn.CMulTable())

   local o2 = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(o1):add(z2)
   o2:add(concat):add(nn.CAddTable())

   seq:add(o2)
   
   return seq
end

------------------------- forward backward -----------------------------
function GRU:updateOutput(input)
   local prevOutput
   if self.step == 1 then
      prevOutput = self.userPrevOutput or self.zeroTensor
      if input:dim() == 2 then
         self.zeroTensor:resize(input:size(1), self.outputSize):zero()
      else
         self.zeroTensor:resize(self.outputSize):zero()
      end
   else
      -- previous output and cell of this module
      prevOutput = self.output
   end

   -- output(t) = gru{input(t), output(t-1)}
   local output
   if self.train ~= false then
      self:recycle()
      local recurrentModule = self:getStepModule(self.step)
      -- the actual forward propagation
      output = recurrentModule:updateOutput{input, prevOutput}
   else
      output = self.recurrentModule:updateOutput{input, prevOutput}
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
   -- note that we don't return the cell, just the output
   return self.output
end

function GRU:backwardThroughTime(timeStep, rho)
   assert(self.step > 1, "expecting at least one updateOutput")
   self.gradInputs = {} -- used by Sequencer, Repeater
   timeStep = timeStep or self.step
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho
   
   if self.fastBackward then
      for step=timeStep-1,math.max(stop,1),-1 do
         -- set the output/gradOutput states of current Module
         local recurrentModule = self:getStepModule(step)
         
         -- backward propagate through this step
         local gradOutput = self.gradOutputs[step]
         if self.gradPrevOutput then
            self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
            nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
            gradOutput = self._gradOutputs[step]
         end
         
         local scale = self.scales[step]
         local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
         local inputTable = {self.inputs[step], output, cell}
         local gradInputTable = recurrentModule:backward(inputTable, gradOutput, scale)
         gradInput, self.gradPrevOutput = unpack(gradInputTable)
         table.insert(self.gradInputs, 1, gradInput)
         if self.userPrevOutput then self.userGradPrevOutput = self.gradPrevOutput end
      end
      self.gradParametersAccumulated = true
      return gradInput
   else
      local gradInput = self:updateGradInputThroughTime()
      self:accGradParametersThroughTime()
      return gradInput
   end
end

function GRU:updateGradInputThroughTime(timeStep, rho)
   assert(self.step > 1, "expecting at least one updateOutput")
   self.gradInputs = {}
   local gradInput
   timeStep = timeStep or self.step
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho

   for step=timeStep-1,math.max(stop,1),-1 do
      -- set the output/gradOutput states of current Module
      local recurrentModule = self:getStepModule(step)
      
      -- backward propagate through this step
      local gradOutput = self.gradOutputs[step]
      if self.gradPrevOutput then
         self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
         nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
         gradOutput = self._gradOutputs[step]
      end
      
      local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
      local inputTable = {self.inputs[step], output}
      local gradInputTable = recurrentModule:updateGradInput(inputTable, gradOutput)
      gradInput, self.gradPrevOutput = unpack(gradInputTable)
      table.insert(self.gradInputs, 1, gradInput)
      if self.userPrevOutput then self.userGradPrevOutput = self.gradPrevOutput end
   end
   
   return gradInput
end

function GRU:accGradParametersThroughTime(timeStep, rho)
   timeStep = timeStep or self.step
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho
   
   for step=timeStep-1,math.max(stop,1),-1 do
      -- set the output/gradOutput states of current Module
      local recurrentModule = self:getStepModule(step)
      
      -- backward propagate through this step
      local scale = self.scales[step]
      local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
      local inputTable = {self.inputs[step], output}
      local gradOutput = (step == self.step-1) and self.gradOutputs[step] or self._gradOutputs[step]
      recurrentModule:accGradParameters(inputTable, gradOutput, scale)
   end
   
   self.gradParametersAccumulated = true
   return gradInput
end

function GRU:accUpdateGradParametersThroughTime(lr, timeStep, rho)
   timeStep = timeStep or self.step
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho
   
   for step=timeStep-1,math.max(stop,1),-1 do
      -- set the output/gradOutput states of current Module
      local recurrentModule = self:getStepModule(step)
      
      -- backward propagate through this step
      local scale = self.scales[step] 
      local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
      local inputTable = {self.inputs[step], output}
      local gradOutput = (step == self.step-1) and self.gradOutputs[step] or self._gradOutputs[step]
      recurrentModule:accUpdateGradParameters(inputTable, self.gradOutputs[step], lr*scale)
   end
   
   return gradInput
end