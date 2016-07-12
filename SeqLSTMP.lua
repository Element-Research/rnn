--[[
The MIT License (MIT)

Copyright (c) 2016 Justin Johnson

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
--]]

--[[
Thank you Justin for this awesome super fast code: 
 * https://github.com/jcjohnson/torch-rnn

If we add up the sizes of all the tensors for output, gradInput, weights,
gradWeights, and temporary buffers, we get that a SeqLSTMP stores this many
scalar values:

NTD + 6NTH + 8NH + 8H^2 + 8DH + 9H

N : batchsize; T : seqlen; D : inputsize; H : outputsize

For N = 100, D = 512, T = 100, H = 1024 and with 4 bytes per number, this comes
out to 305MB. Note that this class doesn't own input or gradOutput, so you'll
see a bit higher memory usage in practice.
--]]
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
end

function SeqLSTMP:gradAdapter(scale, t)
   self.gradWeightO:addmm(scale, self._hidden[t]:t(), self.grad_next_h)
   self.grad_next_h = torch.mm(self.grad_next_h, self.weightO:t())
end

function SeqLSTMP:toFastLSTM()
   assert("toFastLSTM not supported for SeqLSTMP")
end
