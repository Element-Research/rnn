local FastLSTM, parent = torch.class("nn.FastLSTM", "nn.LSTM")

function FastLSTM:__init(inputSize, outputSize, rho)
   parent.__init(self, inputSize, outputSize, rho, false)
end

function FastLSTM:buildModel()
   -- input : {input, prevOutput, prevCell}
   -- output : {output, cell}
   
   -- Calculate all four gates in one go : input, hidden, forget, output
   self.i2g = nn.Linear(self.inputSize, 4*self.outputSize)
   self.o2g = nn.LinearNoBias(self.outputSize, 4*self.outputSize)

   local para = nn.ParallelTable():add(self.i2g):add(self.o2g)
   local gates = nn.Sequential()
   gates:add(nn.NarrowTable(1,2))
   gates:add(para)
   gates:add(nn.CAddTable())

   -- Reshape to (batch_size, n_gates, hid_size)
   -- Then slize the n_gates dimension, i.e dimension 2
   gates:add(nn.Reshape(4,self.outputSize))
   gates:add(nn.SplitTable(1,2))
   local transfer = nn.ParallelTable()
   transfer:add(nn.Sigmoid()):add(nn.Tanh()):add(nn.Sigmoid()):add(nn.Sigmoid())
   gates:add(transfer)

   local concat = nn.ConcatTable()
   concat:add(gates):add(nn.SelectTable(3))
   local seq = nn.Sequential()
   seq:add(concat)
   seq:add(nn.FlattenTable()) -- input, hidden, forget, output, cell
   
   -- input gate * hidden state
   local hidden = nn.Sequential()
   hidden:add(nn.NarrowTable(1,2))
   hidden:add(nn.CMulTable())
   
   -- forget gate * cell
   local cell = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(nn.SelectTable(3)):add(nn.SelectTable(5))
   cell:add(concat)
   cell:add(nn.CMulTable())
   
   local nextCell = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(hidden):add(cell)
   nextCell:add(concat)
   nextCell:add(nn.CAddTable())
   
   local concat = nn.ConcatTable()
   concat:add(nextCell):add(nn.SelectTable(4))
   seq:add(concat)
   seq:add(nn.FlattenTable()) -- nextCell, outputGate
   
   local cellAct = nn.Sequential()
   cellAct:add(nn.SelectTable(1))
   cellAct:add(nn.Tanh())
   local concat = nn.ConcatTable()
   concat:add(cellAct):add(nn.SelectTable(2))
   local output = nn.Sequential()
   output:add(concat)
   output:add(nn.CMulTable())
   
   local concat = nn.ConcatTable()
   concat:add(output):add(nn.SelectTable(1))
   seq:add(concat)
   
   return seq
end

function FastLSTM:buildGate()
   error"Not Implemented"
end

function FastLSTM:buildInputGate()
   error"Not Implemented"
end

function FastLSTM:buildForgetGate()
   error"Not Implemented"
end

function FastLSTM:buildHidden()
   error"Not Implemented"
end

function FastLSTM:buildCell()
   error"Not Implemented"
end   
   
function FastLSTM:buildOutputGate()
   error"Not Implemented"
end
