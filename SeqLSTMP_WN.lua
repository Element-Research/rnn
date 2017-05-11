-- Modified by Richard Assar
-- Weight Normalized LSTM with weighted peephole connections
local SeqLSTMP_WN, parent = torch.class('nn.SeqLSTMP_WN', 'nn.SeqLSTM_WN')

function SeqLSTMP_WN:__init(inputsize, hiddensize, outputsize)
   outputsize = outputsize or hiddensize
   assert(inputsize and hiddensize and outputsize, "Expecting input, hidden and output size")
   local D, H, R = inputsize, hiddensize, outputsize
   
   self.weightO = torch.Tensor(H, R)
   self.gradWeightO = torch.Tensor(H, R)

   self.gO = torch.Tensor(1, R)
   self.gradGO = torch.Tensor(1, R):zero()
   self.vO = torch.Tensor(H, R)
   self.gradVO = torch.Tensor(H, R):zero()

   self.normO = torch.Tensor(1, R)
   self.scaleO = torch.Tensor(1, R)

   self.bufferO1 = torch.Tensor()
   self.bufferO2 = torch.Tensor()
   
   parent.__init(self, inputsize, hiddensize, outputsize)
end

function SeqLSTMP_WN:initFromWeight(weight, weightO)
   parent.initFromWeight(self, weight)

   weightO = weightO or self.weightO

   self.gO = weightO:norm(2,1):clamp(self.eps,math.huge)
   self.vO:copy(weightO)

   return self
end

function SeqLSTMP_WN:reset(std)
   self.bias:zero()
   self.bias[{{self.outputsize + 1, 2 * self.outputsize}}]:fill(1)

   if not std then
      self.weight:normal(0, 1.0 / math.sqrt(self.hiddensize + self.inputsize))
      self.weightO:normal(0, 1.0 / math.sqrt(self.outputsize + self.hiddensize))
   else
      self.weight:normal(0, std)
      self.weightO:normal(0, std)
   end

   self:initFromWeight()

   return self
end

function SeqLSTMP_WN:updateWeightMatrix()
   parent.updateWeightMatrix(self)

   local H, R, D = self.hiddensize, self.outputsize, self.inputsize

   self.normO:norm(self.vO,2,1):clamp(self.eps,math.huge)
   self.scaleO:cdiv(self.gO,self.normO)
   self.weightO:cmul(self.vO,self.scaleO:expandAs(self.vO))
end

function SeqLSTMP_WN:adapter(t)
   local T, N = self._output:size(1), self._output:size(2)
   self._hidden = self._hidden or self.next_h.new()
   self._hidden:resize(T, N, self.hiddensize)
   
   self._hidden[t]:copy(self.next_h)
   self.next_h:resize(N,self.outputsize)
   self.next_h:mm(self._hidden[t], self.weightO)
end

function SeqLSTMP_WN:gradAdapter(scale, t)
   scale = scale or 1.0
   assert(scale == 1.0, 'must have scale=1')

   self.buffer3:resizeAs(self.grad_next_h):copy(self.grad_next_h)
   
   local dWo = self.bufferO1:resize(self._hidden[t]:t():size(1), self.grad_next_h:size(2)):mm(self._hidden[t]:t(), self.grad_next_h)

   local normO = self.normO:expandAs(self.vO)
   local scaleO = self.scaleO:expandAs(self.vO)

   self.gradWeightO:cmul(dWo,self.vO):cdiv(normO)

   local dGradGO = self.bufferO2:resize(1,self.gradWeightO:size(2)):sum(self.gradWeightO,1)
   self.gradGO:add(dGradGO)

   dWo:cmul(scaleO)

   self.gradWeightO:cmul(self.vO,scaleO):cdiv(normO)
   self.gradWeightO:cmul(dGradGO:expandAs(self.gradWeightO))    

   dWo:add(-1,self.gradWeightO)

   self.gradVO:add(dWo)

   self.grad_next_h:resize(self._output:size(2), self.hiddensize)
   self.grad_next_h:mm(self.buffer3, self.weightO:t())
end

function SeqLSTMP_WN:parameters()
   local param,dparam = parent.parameters(self)

   table.insert(param, self.gO)
   table.insert(param, self.vO)

   table.insert(dparam, self.gradGO)
   table.insert(dparam, self.gradVO)

   return param,dparam
end

function SeqLSTMP_WN:clearState()
   parent.clearState(self)
   self.bufferO1:set()
   self.bufferO2:set()
end

function SeqLSTMP_WN:accUpdateGradParameters(input, gradOutput, lr)
   error "accUpdateGradParameters not implemented for SeqLSTMP_WN"
end

function SeqLSTMP_WN:toFastLSTM()
   error "toFastLSTM not supported for SeqLSTMP_WN"
end