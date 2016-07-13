local SeqLSTMP, parent = torch.class('nn.SeqLSTMP', 'nn.SeqLSTM')

function SeqLSTMP:__init(inputsize, hiddensize, outputsize)
   -- parent.__init(self, inputsize, hiddensize, outputsize)

   local D, H, R = inputsize, hiddensize, outputsize
   self.inputsize, self.hiddensize, self.outputsize = D, H, R
   
   -- assert(H == R)
   self.weightX = torch.Tensor(D, 4 * H)
   self.weightR = torch.Tensor(R, 4 * H)
   self.weightO = torch.Tensor(H, R)
   self.weight = parent.flatten({self.weightX, self.weightR, self.weightO})
   self.gradWeightX = torch.Tensor(D, 4 * H):zero()
   self.gradWeightR = torch.Tensor(R, 4 * H):zero()
   self.gradWeightO = torch.Tensor(H, R):zero()
   self.gradWeight = parent.flatten({self.gradWeightX, self.gradWeightR, self.gradWeightO})
   self.bias = torch.Tensor(4 * H)
   self.gradBias = torch.Tensor(4 * H):zero()
   self:reset()

   self.cell = torch.Tensor()      -- This will be  (T, N, H)
   self.gates = torch.Tensor()    -- This will be (T, N, 4H)
   self.buffer1 = torch.Tensor() -- This will be (N, H)
   self.buffer2 = torch.Tensor() -- This will be (N, H)
   self.buffer3 = torch.Tensor() -- This will be (1, 4H)
   self.grad_a_buffer = torch.Tensor() -- This will be (N, 4H)
   self.output = torch.Tensor()

   self.h0 = torch.Tensor()
   self.c0 = torch.Tensor()

   self._remember = 'neither'

   self.grad_c0 = torch.Tensor()
   self.grad_h0 = torch.Tensor()
   self.grad_x = torch.Tensor()
   self._hidden = {}
   self.gradInput = {self.grad_c0, self.grad_h0, self.grad_x}
   
   -- set this to true to forward inputs as batchsize x seqlen x ...
   -- instead of seqlen x batchsize
   self.batchfirst = false
   -- set this to true for variable length sequences that seperate
   -- independent sequences with a step of zeros (a tensor of size D)
   self.maskzero = false
end

function SeqLSTMP:adapter(t)
   self._hidden[t] = self.next_h
   self.next_h = torch.mm(self.next_h, self.weightO)
   self._output[t] = self.next_h
end

function SeqLSTMP:gradAdapter(scale, t)
   self.gradWeightO:addmm(scale, self._hidden[t]:t(), self.grad_next_h)
   self.grad_next_h = torch.mm(self.grad_next_h, self.weightO:t())
end

function SeqLSTMP:toFastLSTM()
   assert("toFastLSTM not supported for SeqLSTMP")
end
