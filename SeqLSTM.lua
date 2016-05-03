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
gradWeights, and temporary buffers, we get that a SeqLSTM stores this many
scalar values:

NTD + 6NTH + 8NH + 8H^2 + 8DH + 9H

N : batchsize; T : seqlen; D : inputsize; H : outputsize

For N = 100, D = 512, T = 100, H = 1024 and with 4 bytes per number, this comes
out to 305MB. Note that this class doesn't own input or gradOutput, so you'll
see a bit higher memory usage in practice.
--]]
local SeqLSTM, parent = torch.class('nn.SeqLSTM', 'nn.Module')

function SeqLSTM:__init(inputsize, outputsize)
   parent.__init(self)

   local D, H = inputsize, outputsize
   self.inputsize, self.outputsize = D, H

   self.weight = torch.Tensor(D + H, 4 * H)
   self.gradWeight = torch.Tensor(D + H, 4 * H):zero()
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
   self.gradInput = {self.grad_c0, self.grad_h0, self.grad_x}
   
   -- set this to true to forward inputs as batchsize x seqlen x ...
   -- instead of seqlen x batchsize
   self.batchfirst = false
end


function SeqLSTM:reset(std)
   if not std then
      std = 1.0 / math.sqrt(self.outputsize + self.inputsize)
   end
   self.bias:zero()
   self.bias[{{self.outputsize + 1, 2 * self.outputsize}}]:fill(1)
   self.weight:normal(0, std)
   return self
end

function SeqLSTM:resetStates()
   self.h0 = self.h0.new()
   self.c0 = self.c0.new()
end

local function check_dims(x, dims)
   assert(x:dim() == #dims)
   for i, d in ipairs(dims) do
      assert(x:size(i) == d)
   end
end

-- makes sure x, h0, c0 and gradOutput have correct sizes.
-- batchfirst = true will transpose the N x T to conform to T x N
function SeqLSTM:_prepare_size(input, gradOutput)
   local c0, h0, x
   if torch.type(input) == 'table' and #input == 3 then
      c0, h0, x = unpack(input)
   elseif torch.type(input) == 'table' and #input == 2 then
      h0, x = unpack(input)
   elseif torch.isTensor(input) then
      x = input
   else
      assert(false, 'invalid input')
   end
   assert(x:dim() == 3, "Only supports batch mode")
   
   if self.batchfirst then
      x = x:transpose(1,2)
      gradOutput = gradOutput and gradOutput:transpose(1,2) or nil
   end
   
   local T, N = x:size(1), x:size(2)
   local H, D = self.outputsize, self.inputsize
   
   check_dims(x, {T, N, D})
   if h0 then
      check_dims(h0, {N, H})
   end
   if c0 then
      check_dims(c0, {N, H})
   end
   if gradOutput then
      check_dims(gradOutput, {T, N, H})
   end
   return c0, h0, x, gradOutput
end

--[[
Input:
- c0: Initial cell state, (N, H)
- h0: Initial hidden state, (N, H)
- x: Input sequence, (T, N, D)  

Output:
- h: Sequence of hidden states, (T, N, H)
--]]

function SeqLSTM:updateOutput(input)
   self.recompute_backward = true
   local c0, h0, x = self:_prepare_size(input)
   local N, T = x:size(2), x:size(1)
   local D, H = self.inputsize, self.outputsize
   
   self._output = self._output or self.weight.new()
   
   -- remember previous state?
   local remember
   if self.train ~= false then -- training
      if self._remember == 'both' or self._remember == 'train' then
         remember = true
      elseif self._remember == 'neither' or self._remember == 'eval' then
         remember = false
      end
   else -- evaluate
      if self._remember == 'both' or self._remember == 'eval' then
         remember = true
      elseif self._remember == 'neither' or self._remember == 'train' then
         remember = false
      end
   end

   self._return_grad_c0 = (c0 ~= nil)
   self._return_grad_h0 = (h0 ~= nil)
   if not c0 then
      c0 = self.c0
      if self.userPrevCell then
         local prev_N = self.userPrevCell:size(1)
         assert(prev_N == N, 'batch sizes must be consistent with userPrevCell')
         c0:resizeAs(self.userPrevCell):copy(self.userPrevCell)
      elseif c0:nElement() == 0 or not remember then
         c0:resize(N, H):zero()
      elseif remember then
         local prev_T, prev_N = self.cell:size(1), self.cell:size(2)
         assert(prev_N == N, 'batch sizes must be constant to remember states')
         c0:copy(self.cell[prev_T])
      end
   end
   if not h0 then
      h0 = self.h0
      if self.userPrevOutput then
         local prev_N = self.userPrevOutput:size(1)
         assert(prev_N == N, 'batch sizes must be consistent with userPrevOutput')
         h0:resizeAs(self.userPrevOutput):copy(self.userPrevOutput)
      elseif h0:nElement() == 0 or not remember then
         h0:resize(N, H):zero()
      elseif remember then
         local prev_T, prev_N = self._output:size(1), self._output:size(2)
         assert(prev_N == N, 'batch sizes must be the same to remember states')
         h0:copy(self._output[prev_T])
      end
   end

   local bias_expand = self.bias:view(1, 4 * H):expand(N, 4 * H)
   local Wx = self.weight[{{1, D}}]
   local Wh = self.weight[{{D + 1, D + H}}]

   local h, c = self._output, self.cell
   h:resize(T, N, H):zero()
   c:resize(T, N, H):zero()
   local prev_h, prev_c = h0, c0
   self.gates:resize(T, N, 4 * H):zero()
   for t = 1, T do
      local cur_x = x[t]
      local next_h = h[t]
      local next_c = c[t]
      local cur_gates = self.gates[t]
      cur_gates:addmm(bias_expand, cur_x, Wx)
      cur_gates:addmm(prev_h, Wh)
      cur_gates[{{}, {1, 3 * H}}]:sigmoid()
      cur_gates[{{}, {3 * H + 1, 4 * H}}]:tanh()
      local i = cur_gates[{{}, {1, H}}] -- input gate
      local f = cur_gates[{{}, {H + 1, 2 * H}}] -- forget gate
      local o = cur_gates[{{}, {2 * H + 1, 3 * H}}] -- output gate
      local g = cur_gates[{{}, {3 * H + 1, 4 * H}}] -- input transform
      next_h:cmul(i, g)
      next_c:cmul(f, prev_c):add(next_h)
      next_h:tanh(next_c):cmul(o)
      prev_h, prev_c = next_h, next_c
   end
   self.userPrevOutput = nil
   self.userPrevCell = nil
   
   if self.batchfirst then
      self.output = self._output:transpose(1,2) -- T x N -> N X T
   else
      self.output = self._output
   end

   return self.output
end

function SeqLSTM:backward(input, gradOutput, scale)
   self.recompute_backward = false
   scale = scale or 1.0
   assert(scale == 1.0, 'must have scale=1')
   
   local c0, h0, x, grad_h = self:_prepare_size(input, gradOutput)
   assert(grad_h, "Expecting gradOutput")
   local N, T = x:size(2), x:size(1)
   local D, H = self.inputsize, self.outputsize
   
   self._grad_x = self._grad_x or self.weight.new()
   
   if not c0 then c0 = self.c0 end
   if not h0 then h0 = self.h0 end

   local grad_c0, grad_h0, grad_x = self.grad_c0, self.grad_h0, self._grad_x
   local h, c = self._output, self.cell
   
   local Wx = self.weight[{{1, D}}]
   local Wh = self.weight[{{D + 1, D + H}}]
   local grad_Wx = self.gradWeight[{{1, D}}]
   local grad_Wh = self.gradWeight[{{D + 1, D + H}}]
   local grad_b = self.gradBias

   grad_h0:resizeAs(h0):zero()
   grad_c0:resizeAs(c0):zero()
   grad_x:resizeAs(x):zero()
   self.buffer1:resizeAs(h0)
   self.buffer2:resizeAs(c0)
   local grad_next_h = self.gradPrevOutput and self.buffer1:copy(self.gradPrevOutput) or self.buffer1:zero()
   local grad_next_c = self.userNextGradCell and self.buffer2:copy(self.userNextGradCell) or self.buffer2:zero()
   
   for t = T, 1, -1 do
      local next_h, next_c = h[t], c[t]
      local prev_h, prev_c = nil, nil
      if t == 1 then
         prev_h, prev_c = h0, c0
      else
         prev_h, prev_c = h[t - 1], c[t - 1]
      end
      grad_next_h:add(grad_h[t])

      local i = self.gates[{t, {}, {1, H}}]
      local f = self.gates[{t, {}, {H + 1, 2 * H}}]
      local o = self.gates[{t, {}, {2 * H + 1, 3 * H}}]
      local g = self.gates[{t, {}, {3 * H + 1, 4 * H}}]
      
      local grad_a = self.grad_a_buffer:resize(N, 4 * H):zero()
      local grad_ai = grad_a[{{}, {1, H}}]
      local grad_af = grad_a[{{}, {H + 1, 2 * H}}]
      local grad_ao = grad_a[{{}, {2 * H + 1, 3 * H}}]
      local grad_ag = grad_a[{{}, {3 * H + 1, 4 * H}}]
      
      -- We will use grad_ai, grad_af, and grad_ao as temporary buffers
      -- to to compute grad_next_c. We will need tanh_next_c (stored in grad_ai)
      -- to compute grad_ao; the other values can be overwritten after we compute
      -- grad_next_c
      local tanh_next_c = grad_ai:tanh(next_c)
      local tanh_next_c2 = grad_af:cmul(tanh_next_c, tanh_next_c)
      local my_grad_next_c = grad_ao
      my_grad_next_c:fill(1):add(-1, tanh_next_c2):cmul(o):cmul(grad_next_h)
      grad_next_c:add(my_grad_next_c)
      
      -- We need tanh_next_c (currently in grad_ai) to compute grad_ao; after
      -- that we can overwrite it.
      grad_ao:fill(1):add(-1, o):cmul(o):cmul(tanh_next_c):cmul(grad_next_h)

      -- Use grad_ai as a temporary buffer for computing grad_ag
      local g2 = grad_ai:cmul(g, g)
      grad_ag:fill(1):add(-1, g2):cmul(i):cmul(grad_next_c)

      -- We don't need any temporary storage for these so do them last
      grad_ai:fill(1):add(-1, i):cmul(i):cmul(g):cmul(grad_next_c)
      grad_af:fill(1):add(-1, f):cmul(f):cmul(prev_c):cmul(grad_next_c)
      
      grad_x[t]:mm(grad_a, Wx:t())
      grad_Wx:addmm(scale, x[t]:t(), grad_a)
      grad_Wh:addmm(scale, prev_h:t(), grad_a)
      local grad_a_sum = self.buffer3:resize(1, 4 * H):sum(grad_a, 1)
      grad_b:add(scale, grad_a_sum)

      grad_next_h:mm(grad_a, Wh:t())
      grad_next_c:cmul(f)
   end
   grad_h0:copy(grad_next_h)
   grad_c0:copy(grad_next_c)
   
   if self.batchfirst then
      self.grad_x = grad_x:transpose(1,2) -- T x N -> N x T
   else
      self.grad_x = grad_x
   end
   self.gradPrevOutput = nil
   self.userNextGradCell = nil
   self.userGradPrevCell = self.grad_c0
   self.userGradPrevOutput = self.grad_h0
   
   if self._return_grad_c0 and self._return_grad_h0 then
      self.gradInput = {self.grad_c0, self.grad_h0, self.grad_x}
   elseif self._return_grad_h0 then
      self.gradInput = {self.grad_h0, self.grad_x}
   else
      self.gradInput = self.grad_x
   end

   return self.gradInput
end

function SeqLSTM:clearState()
   self.cell:set()
   self.gates:set()
   self.buffer1:set()
   self.buffer2:set()
   self.buffer3:set()
   self.grad_a_buffer:set()

   self.grad_c0:set()
   self.grad_h0:set()
   self.grad_x:set()
   self._grad_x = nil
   self.output:set()
   self._output = nil
   self.gradInput = nil
end

function SeqLSTM:updateGradInput(input, gradOutput)
   if self.recompute_backward then
      self:backward(input, gradOutput, 1.0)
   end
   return self.gradInput
end

function SeqLSTM:accGradParameters(input, gradOutput, scale)
   if self.recompute_backward then
      self:backward(input, gradOutput, scale)
   end
end

function SeqLSTM:forget()
   self.c0:resize(0)
   self.h0:resize(0)
end

-- Toggle to feed long sequences using multiple forwards.
-- 'eval' only affects evaluation (recommended for RNNs)
-- 'train' only affects training
-- 'neither' affects neither training nor evaluation
-- 'both' affects both training and evaluation (recommended for LSTMs)
SeqLSTM.remember = nn.Sequencer.remember

function SeqLSTM:training()
   if self.train == false then
      -- forget at the start of each training
      self:forget()
   end
   parent.training(self)
end

function SeqLSTM:evaluate()
   if self.train ~= false then
      -- forget at the start of each evaluation
      self:forget()
   end
   parent.evaluate(self)
   assert(self.train == false)
end

function SeqLSTM:toFastLSTM()   
   local D, H = self.inputsize, self.outputsize
   -- input : x to ...
   local Wxi = self.weight[{{1, D},{1, H}}]
   local Wxf = self.weight[{{1, D},{H + 1, 2 * H}}]
   local Wxo = self.weight[{{1, D},{2 * H + 1, 3 * H}}]
   local Wxg = self.weight[{{1, D},{3 * H + 1, 4 * H}}]
   
   local gWxi = self.gradWeight[{{1, D},{1, H}}]
   local gWxf = self.gradWeight[{{1, D},{H + 1, 2 * H}}]
   local gWxo = self.gradWeight[{{1, D},{2 * H + 1, 3 * H}}]
   local gWxg = self.gradWeight[{{1, D},{3 * H + 1, 4 * H}}]
   
   -- hidden : h to ...
   local Whi = self.weight[{{D + 1, D + H},{1, H}}]
   local Whf = self.weight[{{D + 1, D + H},{H + 1, 2 * H}}]
   local Who = self.weight[{{D + 1, D + H},{2 * H + 1, 3 * H}}]
   local Whg = self.weight[{{D + 1, D + H},{3 * H + 1, 4 * H}}]
   
   local gWhi = self.gradWeight[{{D + 1, D + H},{1, H}}]
   local gWhf = self.gradWeight[{{D + 1, D + H},{H + 1, 2 * H}}]
   local gWho = self.gradWeight[{{D + 1, D + H},{2 * H + 1, 3 * H}}]
   local gWhg = self.gradWeight[{{D + 1, D + H},{3 * H + 1, 4 * H}}]
   
   -- bias
   local bi = self.bias[{{1, H}}]
   local bf = self.bias[{{H + 1, 2 * H}}]
   local bo = self.bias[{{2 * H + 1, 3 * H}}]
   local bg = self.bias[{{3 * H + 1, 4 * H}}]
   
   local gbi = self.gradBias[{{1, H}}]
   local gbf = self.gradBias[{{H + 1, 2 * H}}]
   local gbo = self.gradBias[{{2 * H + 1, 3 * H}}]
   local gbg = self.gradBias[{{3 * H + 1, 4 * H}}]
   
   local lstm = nn.FastLSTM(self.inputsize, self.outputsize)
   local params, gradParams = lstm:parameters()
   local Wx, b, Wh = params[1], params[2], params[3]
   local gWx, gb, gWh = gradParams[1], gradParams[2], gradParams[3]
   
   Wx[{{1, H}}]:t():copy(Wxi)
   Wx[{{H + 1, 2 * H}}]:t():copy(Wxg)
   Wx[{{2 * H + 1, 3 * H}}]:t():copy(Wxf)
   Wx[{{3 * H + 1, 4 * H}}]:t():copy(Wxo)
   
   gWx[{{1, H}}]:t():copy(gWxi)
   gWx[{{H + 1, 2 * H}}]:t():copy(gWxg)
   gWx[{{2 * H + 1, 3 * H}}]:t():copy(gWxf)
   gWx[{{3 * H + 1, 4 * H}}]:t():copy(gWxo)
   
   Wh[{{1, H}}]:t():copy(Whi)
   Wh[{{H + 1, 2 * H}}]:t():copy(Whg)
   Wh[{{2 * H + 1, 3 * H}}]:t():copy(Whf)
   Wh[{{3 * H + 1, 4 * H}}]:t():copy(Who)
   
   gWh[{{1, H}}]:t():copy(gWhi)
   gWh[{{H + 1, 2 * H}}]:t():copy(gWhg)
   gWh[{{2 * H + 1, 3 * H}}]:t():copy(gWhf)
   gWh[{{3 * H + 1, 4 * H}}]:t():copy(gWho)
   
   b[{{1, H}}]:copy(bi)
   b[{{H + 1, 2 * H}}]:copy(bg)
   b[{{2 * H + 1, 3 * H}}]:copy(bf)
   b[{{3 * H + 1, 4 * H}}]:copy(bo)
   
   gb[{{1, H}}]:copy(gbi)
   gb[{{H + 1, 2 * H}}]:copy(gbg)
   gb[{{2 * H + 1, 3 * H}}]:copy(gbf)
   gb[{{3 * H + 1, 4 * H}}]:copy(gbo)
   
   return lstm
end
