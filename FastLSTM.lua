local FastLSTM, parent = torch.class("nn.FastLSTM", "nn.LSTM")

-- set this to true to have it use nngraph instead of nn
-- setting this to true can make your next FastLSTM significantly faster
FastLSTM.usenngraph = false

function FastLSTM:__init(inputSize, outputSize, rho)
   parent.__init(self, inputSize, outputSize, rho, false)
end

function FastLSTM:buildModel()
   -- input : {input, prevOutput, prevCell}
   -- output : {output, cell}
   
   -- Calculate all four gates in one go : input, hidden, forget, output
   self.i2g = nn.Linear(self.inputSize, 4*self.outputSize)
   self.o2g = nn.LinearNoBias(self.outputSize, 4*self.outputSize)
   
   if self.usenngraph then
      require 'nngraph'
      return self:nngraphModel()
   end

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

function FastLSTM:nngraphModel()
   assert(nngraph, "Missing nngraph package")
   
   local inputs = {}
   table.insert(inputs, nn.Identity()()) -- x
   table.insert(inputs, nn.Identity()()) -- prev_h[L]
   table.insert(inputs, nn.Identity()()) -- prev_c[L]
   
   local x, prev_h, prev_c = unpack(inputs)
   
   -- evaluate the input sums at once for efficiency
   local i2h = self.i2g(x):annotate{name='i2h'}
   local h2h = self.o2g(prev_h):annotate{name='h2h'}
   local all_input_sums = nn.CAddTable()({i2h, h2h})

   local reshaped = nn.Reshape(4, self.outputSize)(all_input_sums)
   -- input, hidden, forget, output
   local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
   local in_gate = nn.Sigmoid()(n1)
   local in_transform = nn.Tanh()(n2)
   local forget_gate = nn.Sigmoid()(n3)
   local out_gate = nn.Sigmoid()(n4)
   
   -- perform the LSTM update
   local next_c           = nn.CAddTable()({
     nn.CMulTable()({forget_gate, prev_c}),
     nn.CMulTable()({in_gate,     in_transform})
   })
   -- gated cells form the output
   local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

   local outputs = {next_h, next_c}
   
   return nn.gModule(inputs, outputs)
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
