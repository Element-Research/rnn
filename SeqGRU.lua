--[[
The MIT License (MIT)

Copyright (c) 2016 Stéphane Guillitte, Joost van Doorn

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

require 'torch'
require 'nn'

local SeqGRU, parent = torch.class('nn.SeqGRU', 'nn.Module')

--[[
If we add up the sizes of all the tensors for output, gradInput, weights,
gradWeights, and temporary buffers, we get that a SequenceGRU stores this many
scalar values:

NTD + 4NTH + 5NH + 6H^2 + 6DH + 7H

Note that this class doesn't own input or gradOutput, so you'll
see a bit higher memory usage in practice.
--]]

function SeqGRU:__init(inputSize, outputSize)
  parent.__init(self)

  self.inputSize = inputSize
  self.outputSize = outputSize
  self.seqLength = 1
  self.miniBatch = 1

  local D, H = inputSize, outputSize

  self.weight = torch.Tensor(D + H, 3 * H)
  self.gradWeight = torch.Tensor(D + H, 3 * H):zero()
  self.bias = torch.Tensor(3 * H)
  self.gradBias = torch.Tensor(3 * H):zero()
  self:reset()

  self.gates = torch.Tensor() -- This will be (T, N, 3H)
  self.buffer1 = torch.Tensor() -- This will be (N, H)
  self.buffer2 = torch.Tensor() -- This will be (N, H)
  self.buffer3 = torch.Tensor() -- This will be (H,)
  self.grad_a_buffer = torch.Tensor() -- This will be (N, 3H)

  self.h0 = torch.Tensor()
  
  self._remember = 'neither'

  self.grad_h0 = torch.Tensor()
  self.grad_x = torch.Tensor()
  self.gradInput = {self.grad_h0, self.grad_x}

  -- set this to true to forward inputs as batchsize x seqlen x ...
  -- instead of seqlen x batchsize
  self.batchfirst = false
  -- set this to true for variable length sequences that seperate
  -- independent sequences with a step of zeros (a tensor of size D)
  self.maskzero = false
end

function SeqGRU:reset(std)
  if not std then
    std = 1.0 / math.sqrt(self.outputSize + self.inputSize)
  end
  self.bias:zero()
  self.bias[{{self.outputSize + 1, 2 * self.outputSize}}]:fill(1)
  self.weight:normal(0, std)
  return self
end

function SeqGRU:resetStates()
  self.h0 = self.h0.new()
end

-- unlike MaskZero, the mask is applied in-place
function SeqGRU:recursiveMask(output, mask)
  if torch.type(output) == 'table' then
    for k,v in ipairs(output) do
      self:recursiveMask(output[k], mask)
    end
  else
    assert(torch.isTensor(output))

    -- make sure mask has the same dimension as the output tensor
    local outputSize = output:size():fill(1)
    outputSize[1] = output:size(1)
    mask:resize(outputSize)
    -- build mask
    local zeroMask = mask:expandAs(output)
    output:maskedFill(zeroMask, 0)
  end
end

local function check_dims(x, dims)
  assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    assert(x:size(i) == d)
  end
end

-- makes sure x, h0 and gradOutput have correct sizes.
-- batchfirst = true will transpose the N x T to conform to T x N
function SeqGRU:_prepare_size(input, gradOutput)
  local h0, x
  if torch.type(input) == 'table' and #input == 2 then
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
  local H, D = self.outputSize, self.inputSize

  check_dims(x, {T, N, D})
  if h0 then
    check_dims(h0, {N, H})
  end
  if gradOutput then
    check_dims(gradOutput, {T, N, H})
  end
  return h0, x, gradOutput
end

--[[
Input:
- h0: Initial hidden state, (N, H)
- x: Input sequence, (T, N, D)

Output:
- h: Sequence of hidden states, (T, N, H)
--]]

function SeqGRU:updateOutput(input)
  self.recompute_backward = true
  local h0, x = self:_prepare_size(input)
  local T, N = x:size(1), x:size(2)
  local D, H = self.inputSize, self.outputSize
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

  self._return_grad_h0 = (h0 ~= nil)

  if not h0 then
    h0 = self.h0
    if self.userPrevOutput then
      local prev_N = self.userPrevOutput:size(1)
      assert(prev_N == N, 'batch sizes must be consistent with userPrevOutput')
      h0:resizeAs(self.userPrevOutput):copy(self.userPrevOutput)
    elseif h0:nElement() == 0 or not remember then
      h0:resize(N, H):zero()
    elseif remember then
      local prev_T, prev_N  = self._output:size(1), self._output:size(2)
      assert(prev_N == N, 'batch sizes must be the same to remember states')
      h0:copy(self._output[prev_T])
    end
  end

  local bias_expand = self.bias:view(1, 3 * H):expand(N, 3 * H)
  local Wx = self.weight[{{1, D}}]
  local Wh = self.weight[{{D + 1, D + H}}]

  local h = self._output
  h:resize(T, N, H):zero()
  local prev_h = h0
  self.gates:resize(T, N, 3 * H):zero()
  for t = 1, T do
    local cur_x = x[t]
    local next_h = h[t]
    local cur_gates = self.gates[t]

    cur_gates:addmm(bias_expand, cur_x, Wx)
    cur_gates[{{}, {1, 2 * H}}]:addmm(prev_h, Wh[{{}, {1, 2 * H}}])
    cur_gates[{{}, {1, 2 * H}}]:sigmoid()
    local r = cur_gates[{{}, {1, H}}] --reset gate : r = sig(Wx * x + Wh * prev_h + b)
    local u = cur_gates[{{}, {H + 1, 2 * H}}] --update gate : u = sig(Wx * x + Wh * prev_h + b)
    next_h:cmul(r, prev_h) --temporary buffer : r . prev_h
    cur_gates[{{}, {2 * H + 1, 3 * H}}]:addmm(next_h, Wh[{{}, {2 * H + 1, 3 * H}}]) -- hc += Wh * r . prev_h
    local hc = cur_gates[{{}, {2 * H + 1, 3 * H}}]:tanh() --hidden candidate : hc = tanh(Wx * x + Wh * r . prev_h + b)
    next_h:addcmul(hc, -1, u, hc)
    next_h:addcmul(u, prev_h) --next_h = (1-u) . hc + u . prev_h

    if self.maskzero then
      -- build mask from input
      local vectorDim = cur_x:dim()
      self._zeroMask = self._zeroMask or cur_x.new()
      self._zeroMask:norm(cur_x, 2, vectorDim)
      self.zeroMask = self.zeroMask or ((torch.type(cur_x) == 'torch.CudaTensor') and torch.CudaTensor() or torch.ByteTensor())
      self._zeroMask.eq(self.zeroMask, self._zeroMask, 0)
      -- zero masked output
      self:recursiveMask({next_h, cur_gates}, self.zeroMask)
    end

    prev_h = next_h
  end
  self.userPrevOutput = nil

  if self.batchfirst then
    self.output = self._output:transpose(1,2) -- T x N -> N X T
  else
    self.output = self._output
  end
  
  return self.output
end

function SeqGRU:backward(input, gradOutput, scale)
  self.recompute_backward = false
  scale = scale or 1.0
  assert(scale == 1.0, 'must have scale=1')

  local h0, x, grad_h = self:_prepare_size(input, gradOutput)
  assert(grad_h, "Expecting gradOutput")
  local N, T = x:size(2), x:size(1)
  local D, H = self.inputSize, self.outputSize

  self._grad_x = self._grad_x or self.weight.new()

  if not h0 then h0 = self.h0 end

  local grad_h0, grad_x = self.grad_h0, self._grad_x
  local h = self._output

  local Wx = self.weight[{{1, D}}]
  local Wh = self.weight[{{D + 1, D + H}}]
  local grad_Wx = self.gradWeight[{{1, D}}]
  local grad_Wh = self.gradWeight[{{D + 1, D + H}}]
  local grad_b = self.gradBias

  grad_h0:resizeAs(h0):zero()

  grad_x:resizeAs(x):zero()
  self.buffer1:resizeAs(h0)
  local grad_next_h = self.gradPrevOutput and self.buffer1:copy(self.gradPrevOutput) or self.buffer1:zero()
  local temp_buffer = self.buffer2:resizeAs(h0):zero()
  for t = T, 1, -1 do
    local next_h = h[t]
    local prev_h = nil
    if t == 1 then
      prev_h = h0
    else
      prev_h = h[t - 1]
    end
    grad_next_h:add(grad_h[t])
    
    if self.maskzero then    
      -- build mask from input
      local cur_x = x[t]
      local vectorDim = cur_x:dim()
      self._zeroMask = self._zeroMask or cur_x.new()
      self._zeroMask:norm(cur_x, 2, vectorDim)
      self.zeroMask = self.zeroMask or ((torch.type(cur_x) == 'torch.CudaTensor') and torch.CudaTensor() or torch.ByteTensor())
      self._zeroMask.eq(self.zeroMask, self._zeroMask, 0)
      -- zero masked gradOutput
      self:recursiveMask(grad_next_h, self.zeroMask)
    end

    local r = self.gates[{t, {}, {1, H}}]
    local u = self.gates[{t, {}, {H + 1, 2 * H}}]
    local hc = self.gates[{t, {}, {2 * H + 1, 3 * H}}]

    local grad_a = self.grad_a_buffer:resize(N, 3 * H):zero()
    local grad_ar = grad_a[{{}, {1, H}}]
    local grad_au = grad_a[{{}, {H + 1, 2 * H}}]
    local grad_ahc = grad_a[{{}, {2 * H + 1, 3 * H}}]
    
    

    -- We will use grad_au as temporary buffer
    -- to compute grad_ahc.

    local grad_hc = grad_au:fill(0):addcmul(grad_next_h, -1, u, grad_next_h)
    grad_ahc:fill(1):addcmul(-1, hc,hc):cmul(grad_hc)
    local grad_r = grad_au:fill(0):addmm(grad_ahc, Wh[{{}, {2 * H + 1, 3 * H}}]:t() ):cmul(prev_h)
    grad_ar:fill(1):add(-1, r):cmul(r):cmul(grad_r)

    temp_buffer:fill(0):add(-1, hc):add(prev_h)
    grad_au:fill(1):add(-1, u):cmul(u):cmul(temp_buffer):cmul(grad_next_h)
    grad_x[t]:mm(grad_a, Wx:t())
    grad_Wx:addmm(scale, x[t]:t(), grad_a)
    grad_Wh[{{}, {1, 2 * H}}]:addmm(scale, prev_h:t(), grad_a[{{}, {1, 2 * H}}])

    local grad_a_sum = self.buffer3:resize(H):sum(grad_a, 1)
    grad_b:add(scale, grad_a_sum)
    temp_buffer:fill(0):add(prev_h):cmul(r)
    grad_Wh[{{}, {2 * H + 1, 3 * H}}]:addmm(scale, temp_buffer:t(), grad_ahc)
    grad_next_h:cmul(u)
    grad_next_h:addmm(grad_a[{{}, {1, 2 * H}}], Wh[{{}, {1, 2 * H}}]:t())
    temp_buffer:fill(0):addmm(grad_a[{{}, {2 * H + 1, 3 * H}}], Wh[{{}, {2 * H + 1, 3 * H}}]:t()):cmul(r)
    grad_next_h:add(temp_buffer)
  end
  grad_h0:copy(grad_next_h)

  if self.batchfirst then
    self.grad_x = grad_x:transpose(1,2) -- T x N -> N x T
  else
    self.grad_x = grad_x
  end
  self.gradPrevOutput = nil
  self.userGradPrevOutput = self.grad_h0

  if self._return_grad_h0 then
    self.gradInput = {self.grad_h0, self.grad_x}
  else
    self.gradInput = self.grad_x
  end

  return self.gradInput
end

function SeqGRU:clearState()
  self.gates:set()
  self.buffer1:set()
  self.buffer2:set()
  self.buffer3:set()
  self.grad_a_buffer:set()

  self.grad_h0:set()
  self.grad_x:set()
  self._grad_x = nil
  self.output:set()
  self._output = nil
  self.gradInput = nil

  self.zeroMask = nil
  self._zeroMask = nil
  self._maskbyte = nil
  self._maskindices = nil

  self.userGradPrevOutput = nil
  self.gradPrevOutput = nil
end

function SeqGRU:updateGradInput(input, gradOutput)
  if self.recompute_backward then
    self:backward(input, gradOutput, 1.0)
  end
  return self.gradInput
end

function SeqGRU:forget()
  self.h0:resize(0)
end

function SeqGRU:accGradParameters(input, gradOutput, scale)
  if self.recompute_backward then
    self:backward(input, gradOutput, scale)
  end
end

function SeqGRU:type(type, ...)
  self.zeroMask = nil
  self._zeroMask = nil
  self._maskbyte = nil
  self._maskindices = nil
  return parent.type(self, type, ...)
end

-- Toggle to feed long sequences using multiple forwards.
-- 'eval' only affects evaluation (recommended for RNNs)
-- 'train' only affects training
-- 'neither' affects neither training nor evaluation
-- 'both' affects both training and evaluation (recommended for LSTMs)
SeqGRU.remember = nn.Sequencer.remember

function SeqGRU:training()
  if self.train == false then
    -- forget at the start of each training
    self:forget()
  end
  parent.training(self)
end

function SeqGRU:evaluate()
  if self.train ~= false then
    -- forget at the start of each evaluation
    self:forget()
  end
  parent.evaluate(self)
  assert(self.train == false)
end

function SeqGRU:toGRU()
  local D, H = self.inputSize, self.outputSize

  local Wx = self.weight[{{1, D}}]
  local Wh = self.weight[{{D + 1, D + H}}]
  local gWx = self.gradWeight[{{1, D}}]
  local gWh = self.gradWeight[{{D + 1, D + H}}]

  -- bias
  local bxi = self.bias[{{1, 2 * H}}]
  local bxo = self.bias[{{2 * H + 1, 3 * H}}]

  local gbxi = self.gradBias[{{1, 2 * H}}]
  local gbxo = self.gradBias[{{2 * H + 1, 3 * H}}]

  local gru = nn.GRU(self.inputSize, self.outputSize)
  local params, gradParams = gru:parameters()
  local nWxi, nbxi, nWhi, nWxo, nbxo, nWho = unpack(params)
  local ngWxi, ngbxi, ngWhi, ngWxo, ngbxo, ngWho = unpack(gradParams)
  
  
  nWxi:t():copy(Wx[{{}, {1, 2*H}}]) -- update and reset gate
  nWxo:t():copy(Wx[{{}, {2 * H + 1, 3 * H}}])
  nWhi:t():copy(Wh[{{}, {1, 2*H}}])
  nWho:t():copy(Wh[{{}, {2 * H + 1, 3 * H}}])
  nbxi:copy(bxi[{{1, 2 * H}}])
  nbxo:copy(bxo)
  ngWxi:t():copy(gWx[{{}, {1, 2*H}}]) -- update and reset gate
  ngWxo:t():copy(gWx[{{}, {2 * H + 1, 3 * H}}]) --
  ngWhi:t():copy(gWh[{{}, {1, 2*H}}])
  ngWho:t():copy(gWh[{{}, {2 * H + 1, 3 * H}}])
  ngbxi:copy(gbxi[{{1, 2 * H}}])
  ngbxo:copy(gbxo)
  
  return gru
end


function SeqGRU:maskZero()
  self.maskzero = true
end
