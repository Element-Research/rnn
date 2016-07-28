local SeqLSTMP, parent = torch.class('nn.SeqLSTMP', 'nn.SeqLSTM')

SeqLSTMP.dpnn_parameters = {'weight', 'bias', 'weightO'}
SeqLSTMP.dpnn_gradParameters = {'gradWeight', 'gradBias', 'gradWeightO'}

function SeqLSTMP:__init(inputsize, hiddensize, outputsize)
   assert(inputsize and hiddensize and outputsize, "Expecting input, hidden and output size")
   local D, H, R = inputsize, hiddensize, outputsize
   
   self.weightO = torch.Tensor(H, R)
   self.gradWeightO = torch.Tensor(H, R)
   
   parent.__init(self, inputsize, hiddensize, outputsize)
end

function SeqLSTMP:reset(std)
   self.bias:zero()
   self.bias[{{self.outputsize + 1, 2 * self.outputsize}}]:fill(1)
   if not std then
      self.weight:normal(0, 1.0 / math.sqrt(self.hiddensize + self.inputsize))
      self.weightO:normal(0, 1.0 / math.sqrt(self.outputsize + self.hiddensize))
   else
      self.weight:normal(0, std)
      self.weightO:normal(0, std)
   end
   return self
end

function SeqLSTMP:adapter(t)
   local T, N = self._output:size(1), self._output:size(2)
   self._hidden = self._hidden or self.next_h.new()
   self._hidden:resize(T, N, self.hiddensize)
   
   self._hidden[t]:copy(self.next_h)
   self.next_h:resize(N,self.outputsize)
   self.next_h:mm(self._hidden[t], self.weightO)
end

function SeqLSTMP:gradAdapter(scale, t)
   self.buffer3:resizeAs(self.grad_next_h):copy(self.grad_next_h)
   
   self.gradWeightO:addmm(scale, self._hidden[t]:t(), self.grad_next_h)
   self.grad_next_h:resize(self._output:size(2), self.hiddensize)
   self.grad_next_h:mm(self.buffer3, self.weightO:t())
end

function SeqLSTMP:parameters()
   return {self.weight, self.bias, self.weightO}, {self.gradWeight, self.gradBias, self.gradWeightO}
end

function SeqLSTMP:accUpdateGradParameters(input, gradOutput, lr)
   error"accUpdateGradParameters not implemented for SeqLSTMP"
end

function SeqLSTMP:toFastLSTM()
   error"toFastLSTM not supported for SeqLSTMP"
end
