------------------------------------------------------------------------
--[[ LSTM ]]--
-- Long Short Term Memory architecture.
-- Ref. A.: http://arxiv.org/pdf/1303.5778v1 (blueprint for this module)
-- B. http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf
-- C. http://arxiv.org/pdf/1503.04069v1.pdf
-- D. https://github.com/wojzaremba/lstm
-- Expects 1D or 2D input.
-- The first input in sequence uses zero value for cell and hidden state

-- For p > 0, it becomes Bayesian GRUs [Gal, 2015].
-- In this case, please do not dropout on input as BGRUs handle the input with 
-- its own dropouts. First, try 0.25 for p as Gal (2016) suggested, 
-- presumably, because of summations of two parts in GRUs connections. 
------------------------------------------------------------------------
assert(not nn.LSTM, "update nnx package : luarocks install nnx")
local LSTM, parent = torch.class('nn.LSTM', 'nn.AbstractRecurrent')

function LSTM:__init(inputSize, outputSize, rho, cell2gate, p, mono)
   parent.__init(self, rho or 9999)
   self.p = p or 0
   if p and p ~= 0 then
      assert(nn.Dropout(p,false,false,true).lazy, 'only work with Lazy Dropout!')
   end
   self.mono = mono or false
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
   local input2gate = nn.Sequential()
         :add(nn.Dropout(self.p,false,false,true,self.mono))
         :add(nn.Linear(self.inputSize, self.outputSize))
   local output2gate = nn.Sequential()
         :add(nn.Dropout(self.p,false,false,true,self.mono))
         :add(nn.LinearNoBias(self.outputSize, self.outputSize))
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
   local input2hidden = nn.Sequential()
         :add(nn.Dropout(self.p,false,false,true,self.mono))
         :add(nn.Linear(self.inputSize, self.outputSize))
   local output2hidden = nn.Sequential()
         :add(nn.Dropout(self.p,false,false,true,self.mono))
         :add(nn.LinearNoBias(self.outputSize, self.outputSize))
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

function LSTM:getHiddenState(step, input)
   step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
   local prevOutput, prevCell
   if step == 0 then
      prevOutput = self.userPrevOutput or self.outputs[step] or self.zeroTensor
      prevCell = self.userPrevCell or self.cells[step] or self.zeroTensor
      if input then
         if input:dim() == 2 then
            self.zeroTensor:resize(input:size(1), self.outputSize):zero()
         else
            self.zeroTensor:resize(self.outputSize):zero()
         end
      end
   else
      -- previous output and cell of this module
      prevOutput = self.outputs[step]
      prevCell = self.cells[step]
   end
   return {prevOutput, prevCell}
end

function LSTM:setHiddenState(step, hiddenState)
   step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
   assert(torch.type(hiddenState) == 'table')
   assert(#hiddenState == 2)

   -- previous output of this module
   self.outputs[step] = hiddenState[1]
   self.cells[step] = hiddenState[2]
end

------------------------- forward backward -----------------------------
function LSTM:updateOutput(input)
   local prevOutput, prevCell = unpack(self:getHiddenState(self.step-1, input))

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

function LSTM:getGradHiddenState(step)
   self.gradOutputs = self.gradOutputs or {}
   self.gradCells = self.gradCells or {}
   local _step = self.updateGradInputStep or self.step
   step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step
   local gradOutput, gradCell
   if step == self.step-1 then
      gradOutput = self.userNextGradOutput or self.gradOutputs[step] or self.zeroTensor
      gradCell = self.userNextGradCell or self.gradCells[step] or self.zeroTensor
   else
      gradOutput = self.gradOutputs[step]
      gradCell = self.gradCells[step]
   end
   return {gradOutput, gradCell}
end

function LSTM:setGradHiddenState(step, gradHiddenState)
   local _step = self.updateGradInputStep or self.step
   step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step
   assert(torch.type(gradHiddenState) == 'table')
   assert(#gradHiddenState == 2)

   self.gradOutputs[step] = gradHiddenState[1]
   self.gradCells[step] = gradHiddenState[2]
end

function LSTM:_updateGradInput(input, gradOutput)
   assert(self.step > 1, "expecting at least one updateOutput")
   local step = self.updateGradInputStep - 1
   assert(step >= 1)

   -- set the output/gradOutput states of current Module
   local recurrentModule = self:getStepModule(step)

   -- backward propagate through this step
   local gradHiddenState = self:getGradHiddenState(step)
   local _gradOutput, gradCell = gradHiddenState[1], gradHiddenState[2]
   assert(_gradOutput and gradCell)

   self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], _gradOutput)
   nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
   gradOutput = self._gradOutputs[step]

   local inputTable = self:getHiddenState(step-1)
   table.insert(inputTable, 1, input)

   local gradInputTable = recurrentModule:updateGradInput(inputTable, {gradOutput, gradCell})

   local _ = require 'moses'
   self:setGradHiddenState(step-1, _.slice(gradInputTable, 2, 3))

   return gradInputTable[1]
end

function LSTM:_accGradParameters(input, gradOutput, scale)
   local step = self.accGradParametersStep - 1
   assert(step >= 1)

   -- set the output/gradOutput states of current Module
   local recurrentModule = self:getStepModule(step)

   -- backward propagate through this step
   local inputTable = self:getHiddenState(step-1)
   table.insert(inputTable, 1, input)
   local gradOutputTable = self:getGradHiddenState(step)
   gradOutputTable[1] = self._gradOutputs[step] or gradOutputTable[1]
   recurrentModule:accGradParameters(inputTable, gradOutputTable, scale)
end

function LSTM:clearState()
   self.zeroTensor:set()
   if self.userPrevOutput then self.userPrevOutput:set() end
   if self.userPrevCell then self.userPrevCell:set() end
   if self.userGradPrevOutput then self.userGradPrevOutput:set() end
   if self.userGradPrevCell then self.userGradPrevCell:set() end
   return parent.clearState(self)
end

function LSTM:type(type, ...)
   if type then
      self:forget()
      self:clearState()
      self.zeroTensor = self.zeroTensor:type(type)
   end
   return parent.type(self, type, ...)
end
