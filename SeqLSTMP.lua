local SeqLSTMP, parent = torch.class('nn.SeqLSTMP', 'nn.SeqLSTM')

function SeqLSTMP:__init(inputsize, hiddensize, outputsize)
   assert(inputsize and hiddensize and outputsize, "Expecting input, hidden and output size")
   parent.__init(self, inputsize, hiddensize, outputsize)
   
   local D, H, R = inputsize, hiddensize, outputsize
   
   self.weightO = torch.Tensor(H, R):zero()
   self.gradWeightO = torch.Tensor(H, R):zero()
end

function SeqLSTMP:adapter(t)
   self._hidden = self._hidden or self.next_h.new(T, N, H)
   
   self._hidden[t]:copy(self.next_h)
   self.next_h:mm(self._hidden[t], self.weightO)
end

function SeqLSTMP:gradAdapter(scale, t)
   self.buffer3:resizeAs(self.grad_next_h):copy(self.grad_next_h)
   
   self.gradWeightO:addmm(scale, self._hidden[t]:t(), self.grad_next_h)
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
