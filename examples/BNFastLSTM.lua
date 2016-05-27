local BNFastLSTM, parent = torch.class("nn.BNFastLSTM", "nn.FastLSTM")

-- set this to true to have it use nngraph instead of nn
-- setting this to true can make your next BNFastLSTM significantly faster
BNFastLSTM.usenngraph = true
BNFastLSTM.bn = true

function BNFastLSTM:__init(inputSize, outputSize, rho, cell2gate, eps, momentum, affine) 
   --  initialize batch norm variance with 0.1
   self.eps = self.eps or 0.1
   self.momentum = self.momentum or 0.1 --gamma
   self.affine = self.affine or true
   self.bn = self.bn or true

   parent.__init(self, inputSize, outputSize, rho, false, self.eps, self.momentum, self.affine)  
end

function BNFastLSTM:buildModel()
   -- input : {input, prevOutput, prevCell}
   -- output : {output, cell}
   
   --apply recurrent batch normalization 
   -- http://arxiv.org/pdf/1502.03167v3.pdf
   --normalize recurrent terms W_h*h_{t-1} and W_x*x_t separately 

   -- Calculate all four gates in one go : input, hidden, forget, output
   self.i2g = nn.Linear(self.inputSize, 4*self.outputSize)
   self.o2g = nn.LinearNoBias(self.outputSize, 4*self.outputSize)

   local bn_wx, bn_wh, bn_c  
   if self.bn then  
      bn_wx = nn.BatchNormalization(4*self.outputSize, self.eps, self.momentum, self.affine)
      bn_wh = nn.BatchNormalization(4*self.outputSize, self.eps, self.momentum, self.affine)
      bn_c  = nn.BatchNormalization(self.outputSize, self.eps, self.momentum, self.affine)
   else
      bn_wx = nn.Identity()
      bn_wh = nn.Identity()
      bn_c  = nn.Identity()
   end

   if self.usenngraph then
      require 'nngraph'
      return self:nngraphModel(bn_wx, bn_wh, bn_c)
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
   bn_c:forward(nextCell)  --batch norm of next_cell
   
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

function BNFastLSTM:nngraphModel(bn_wx, bn_wh, bn_c)
   assert(nngraph, "Missing nngraph package")
   
   local inputs = {}
   table.insert(inputs, nn.Identity()()) -- x
   table.insert(inputs, nn.Identity()()) -- prev_h[L]
   table.insert(inputs, nn.Identity()()) -- prev_c[L]
   
   local x, prev_h, prev_c = unpack(inputs)
  
   -- evaluate the input sums at once for efficiency

   -- if type(x) == 'table' then
   --   for i = 1, #x do
   --     x[i] = BN:forward(x[i])
   --     x[i] = transfer_data(x[i])
   --   end
   -- end  

   local i2h = bn_wx(self.i2g(x):annotate{name='i2h'}):annotate {name='bn_wx'}
   local h2h = bn_wh(self.o2g(prev_h):annotate{name='h2h'}):annotate {name = 'bn_wh'}
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
   local next_h = nn.CMulTable()({out_gate, nn.Tanh()(bn_c(next_c):annotate {name = 'bn_c'}) })

   local outputs = {next_h, next_c}

   nngraph.annotateNodes()
   
   return nn.gModule(inputs, outputs)
end

function BNFastLSTM:buildGate()
   error"Not Implemented"
end

function BNFastLSTM:buildInputGate()
   error"Not Implemented"
end

function BNFastLSTM:buildForgetGate()
   error"Not Implemented"
end

function BNFastLSTM:buildHidden()
   error"Not Implemented"
end

function BNFastLSTM:buildCell()
   error"Not Implemented"
end   
   
function BNFastLSTM:buildOutputGate()
   error"Not Implemented"
end
