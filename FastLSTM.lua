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
local FastLSTM, parent = torch.class("nn.FastLSTM", "nn.LSTM")

-- set this to true to have it use nngraph instead of nn
-- setting this to true can make your next FastLSTM significantly faster
FastLSTM.usenngraph = false
FastLSTM.bn = false

function FastLSTM:__init(inputSize, outputSize, rho, eps, momentum, affine, p, mono)
   -- when FastLSTM.bn=true, the default values of eps and momentum are set by nn.BatchNormalization
   self.eps = eps
   self.momentum = momentum
   self.affine = affine == nil and true or affine
   self.p = p or 0
   if p and p ~= 0 then
      assert(nn.Dropout(p,false,false,true).lazy, 'only work with Lazy Dropout!')
   end
   self.mono = mono or false

   parent.__init(self, inputSize, outputSize, rho, nil, p, mono)
end

function FastLSTM:buildModel()
   -- input : {input, prevOutput, prevCell}
   -- output : {output, cell}

   -- Calculate all four gates in one go : input, hidden, forget, output
   if self.p ~= 0 then
      self.i2g = nn.Sequential()
                     :add(nn.ConcatTable()
                        :add(nn.Dropout(self.p,false,false,true,self.mono))
                        :add(nn.Dropout(self.p,false,false,true,self.mono))
                        :add(nn.Dropout(self.p,false,false,true,self.mono))
                        :add(nn.Dropout(self.p,false,false,true,self.mono)))
                     :add(nn.ParallelTable()
                        :add(nn.Linear(self.inputSize, self.outputSize))
                        :add(nn.Linear(self.inputSize, self.outputSize))
                        :add(nn.Linear(self.inputSize, self.outputSize))
                        :add(nn.Linear(self.inputSize, self.outputSize)))
                     :add(nn.JoinTable(2))
      self.o2g = nn.Sequential()
                     :add(nn.ConcatTable()
                        :add(nn.Dropout(self.p,false,false,true,self.mono))
                        :add(nn.Dropout(self.p,false,false,true,self.mono))
                        :add(nn.Dropout(self.p,false,false,true,self.mono))
                        :add(nn.Dropout(self.p,false,false,true,self.mono)))
                     :add(nn.ParallelTable()
                        :add(nn.LinearNoBias(self.outputSize, self.outputSize))
                        :add(nn.LinearNoBias(self.outputSize, self.outputSize))
                        :add(nn.LinearNoBias(self.outputSize, self.outputSize))
                        :add(nn.LinearNoBias(self.outputSize, self.outputSize)))
                     :add(nn.JoinTable(2))
   else
      self.i2g = nn.Linear(self.inputSize, 4*self.outputSize)
      self.o2g = nn.LinearNoBias(self.outputSize, 4*self.outputSize)
   end

   if self.usenngraph or self.bn then
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

   local bn_wx, bn_wh, bn_c
   local i2h, h2h
   if self.bn then
      -- apply recurrent batch normalization
      -- http://arxiv.org/pdf/1502.03167v3.pdf
      -- normalize recurrent terms W_h*h_{t-1} and W_x*x_t separately
      -- Olalekan Ogunmolu <patlekano@gmail.com>

      bn_wx = nn.BatchNormalization(4*self.outputSize, self.eps, self.momentum, self.affine)
      bn_wh = nn.BatchNormalization(4*self.outputSize, self.eps, self.momentum, self.affine)
      bn_c  = nn.BatchNormalization(self.outputSize, self.eps, self.momentum, self.affine)

      -- initialize gamma (the weight) to the recommended value
      -- (https://github.com/torch/nn/blob/master/lib/THNN/generic/BatchNormalization.c#L61)
      bn_wx.weight:fill(0.1)
      bn_wh.weight:fill(0.1)
      bn_c.weight:fill(0.1)

      -- evaluate the input sums at once for efficiency
      i2h = bn_wx(self.i2g(x):annotate{name='i2h'}):annotate {name='bn_wx'}
      h2h = bn_wh(self.o2g(prev_h):annotate{name='h2h'}):annotate {name = 'bn_wh'}

      -- add bias after BN as per paper
      h2h = nn.Add(4*self.outputSize)(h2h)
   else
      -- evaluate the input sums at once for efficiency
      i2h = self.i2g(x):annotate{name='i2h'}
      h2h = self.o2g(prev_h):annotate{name='h2h'}
   end
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
   local next_h
   if self.bn then
      -- gated cells form the output
      next_h = nn.CMulTable()({out_gate, nn.Tanh()(bn_c(next_c):annotate {name = 'bn_c'}) })
   else
      -- gated cells form the output
      next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
   end

   local outputs = {next_h, next_c}

   nngraph.annotateNodes()

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
