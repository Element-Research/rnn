------------------------------------------------------------------------
--[[ LSTM ]]--
-- Long Short Term Memory architecture.
-- Ref. A.: http://arxiv.org/pdf/1303.5778v1 (blueprint for this module)
-- B. http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf
-- C. http://arxiv.org/pdf/1503.04069v1.pdf
-- D. https://github.com/wojzaremba/lstm
-- Expects 1D or 2D input.
-- The first input in sequence uses zero value for cell and hidden state
------------------------------------------------------------------------
assert(not nn.LSTM, "update nnx package : luarocks install nnx")
local LSTM, parent = torch.class('nn.LSTM', 'nn.AbstractRecurrent')

function LSTM:__init(inputSize, outputSize, rho, cell2gate)
   parent.__init(self, rho or 9999)
   self.inputSize = inputSize
   self.outputSize = outputSize or inputSize
   -- build the model
   self.cell2gate = (cell2gate == nil) and true or cell2gate
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
function LSTM:buildGate()
   -- Note : gate expects an input table : {input, output(t-1), cell(t-1)}
   local gate = nn.Sequential()
   if not self.cell2gate then
      gate:add(nn.NarrowTable(1,2))
   end
   local input2gate = nn.Linear(self.inputSize, self.outputSize)
   local output2gate = nn.LinearNoBias(self.outputSize, self.outputSize)
   local para = nn.ParallelTable()
   para:add(input2gate):add(output2gate) 
   if self.cell2gate then
      para:add(nn.CMul(self.outputSize)) -- diagonal cell to gate weight matrix
   end
   gate:add(para)
   gate:add(nn.CAddTable())
   gate:add(nn.Sigmoid())
   return gate
end

function LSTM:buildInputGate()
   self.inputGate = self:buildGate()
   return self.inputGate
end

function LSTM:buildForgetGate()
   self.forgetGate = self:buildGate()
   return self.forgetGate
end

function LSTM:buildHidden()
   local hidden = nn.Sequential()
   -- input is {input, output(t-1), cell(t-1)}, but we only need {input, output(t-1)}
   hidden:add(nn.NarrowTable(1,2))
   local input2hidden = nn.Linear(self.inputSize, self.outputSize)
   local output2hidden = nn.LinearNoBias(self.outputSize, self.outputSize)
   local para = nn.ParallelTable()
   para:add(input2hidden):add(output2hidden)
   hidden:add(para)
   hidden:add(nn.CAddTable())
   hidden:add(nn.Tanh())
   self.hiddenLayer = hidden
   return hidden
end

function LSTM:buildCell()
   -- build
   self.inputGate = self:buildInputGate() 
   self.forgetGate = self:buildForgetGate()
   self.hiddenLayer = self:buildHidden()
   -- forget = forgetGate{input, output(t-1), cell(t-1)} * cell(t-1)
   local forget = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(self.forgetGate):add(nn.SelectTable(3))
   forget:add(concat)
   forget:add(nn.CMulTable())
   -- input = inputGate{input, output(t-1), cell(t-1)} * hiddenLayer{input, output(t-1), cell(t-1)}
   local input = nn.Sequential()
   local concat2 = nn.ConcatTable()
   concat2:add(self.inputGate):add(self.hiddenLayer)
   input:add(concat2)
   input:add(nn.CMulTable())
   -- cell(t) = forget + input
   local cell = nn.Sequential()
   local concat3 = nn.ConcatTable()
   concat3:add(forget):add(input)
   cell:add(concat3)
   cell:add(nn.CAddTable())
   self.cellLayer = cell
   return cell
end   
   
function LSTM:buildOutputGate()
   self.outputGate = self:buildGate()
   return self.outputGate
end

-- cell(t) = cellLayer{input, output(t-1), cell(t-1)}
-- output(t) = outputGate{input, output(t-1), cell(t)}*tanh(cell(t))
-- output of Model is table : {output(t), cell(t)} 
function LSTM:buildModel()
   -- build components
   self.cellLayer = self:buildCell()
   self.outputGate = self:buildOutputGate()
   -- assemble
   local concat = nn.ConcatTable()
   concat:add(nn.NarrowTable(1,2)):add(self.cellLayer)
   local model = nn.Sequential()
   model:add(concat)
   -- output of concat is {{input, output}, cell(t)}, 
   -- so flatten to {input, output, cell(t)}
   model:add(nn.FlattenTable())
   local cellAct = nn.Sequential()
   cellAct:add(nn.SelectTable(3))
   cellAct:add(nn.Tanh())
   local concat3 = nn.ConcatTable()
   concat3:add(self.outputGate):add(cellAct)
   local output = nn.Sequential()
   output:add(concat3)
   output:add(nn.CMulTable())
   -- we want the model to output : {output(t), cell(t)}
   local concat4 = nn.ConcatTable()
   concat4:add(output):add(nn.SelectTable(3))
   model:add(concat4)
   return model
end

------------------------- forward backward -----------------------------
function LSTM:updateOutput(input)
   local prevOutput, prevCell
   if self.step == 1 then
      prevOutput = self.userPrevOutput or self.zeroTensor
      prevCell = self.userPrevCell or self.zeroTensor
      if input:dim() == 2 then
         self.zeroTensor:resize(input:size(1), self.outputSize):zero()
      else
         self.zeroTensor:resize(self.outputSize):zero()
      end
   else
      -- previous output and cell of this module
      prevOutput = self.outputs[self.step-1]
      prevCell = self.cells[self.step-1]
   end
   
   -- output(t), cell(t) = lstm{input(t), output(t-1), cell(t-1)}
   local output, cell
   if self.train ~= false then
      self:recycle()
      local recurrentModule = self:getStepModule(self.step)
      -- the actual forward propagation
      output, cell = unpack(recurrentModule:updateOutput{input, prevOutput, prevCell})
   else
      output, cell = unpack(self.recurrentModule:updateOutput{input, prevOutput, prevCell})
   end
   
   self.outputs[self.step] = output
   self.cells[self.step] = cell
   
   self.output = output
   self.cell = cell
   
   self.step = self.step + 1
   self.gradPrevOutput = nil
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil
   -- note that we don't return the cell, just the output
   return self.output
end

function LSTM:_updateGradInput(input, gradOutput)
   assert(self.step > 1, "expecting at least one updateOutput")
   local step = self.updateGradInputStep - 1
   assert(step >= 1)
   
   -- set the output/gradOutput states of current Module
   local recurrentModule = self:getStepModule(step)
   
   -- backward propagate through this step
   if self.gradPrevOutput then
      self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
      nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
      gradOutput = self._gradOutputs[step]
   end
   
   local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
   local cell = (step == 1) and (self.userPrevCell or self.zeroTensor) or self.cells[step-1]
   local inputTable = {input, output, cell}
   local gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
   
   local gradInputTable = recurrentModule:updateGradInput(inputTable, {gradOutput, gradCell})
   
   local gradInput
   gradInput, self.gradPrevOutput, gradCell = unpack(gradInputTable)
   self.gradCells[step-1] = gradCell
   if self.userPrevOutput then self.userGradPrevOutput = self.gradPrevOutput end
   if self.userPrevCell then self.userGradPrevCell = gradCell end
   
   return gradInput
end

function LSTM:_accGradParameters(input, gradOutput, scale)
   local step = self.accGradParametersStep - 1
   assert(step >= 1)
   
   -- set the output/gradOutput states of current Module
   local recurrentModule = self:getStepModule(step)
   
   -- backward propagate through this step
   local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
   local cell = (step == 1) and (self.userPrevCell or self.zeroTensor) or self.cells[step-1]
   local inputTable = {input, output, cell}
   local gradOutput = (step == self.step-1) and gradOutput or self._gradOutputs[step]
   local gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
   local gradOutputTable = {gradOutput, gradCell}
   recurrentModule:accGradParameters(inputTable, gradOutputTable, scale)
end

