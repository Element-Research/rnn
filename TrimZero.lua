------------------------------------------------------------------------
--[[ TrimZero ]]--
-- Author: Jin-Hwa Kim
-- License: LICENSE.2nd.txt

-- Decorator that zeroes the output rows of the encapsulated module
-- for commensurate input rows which are tensors of zeros

-- The only difference from `MaskZero` is that it reduces computational costs 
-- by varying a batch size, if any, for the case that varying lengths 
-- are provided in the input. Notice that when the lengths are consistent, 
-- `MaskZero` will be faster, because `TrimZero` has an operational cost. 

-- In short, the result is the same with `MaskZero`'s, however, `TrimZero` is
-- faster than `MaskZero` only when sentence lengths is costly vary.
-- In practice, e.g. language model, `TrimZero` is expected to be faster than
--  `MaskZero` 30%. (You can test with it using `test/test_trimzero.lua`.)
------------------------------------------------------------------------
local TrimZero, parent = torch.class("nn.TrimZero", "nn.MaskZero")

require 'torchx'

function TrimZero:__init(module, nInputDim, silent)
   parent.__init(self, module, nInputDim, silent)
   if torch.typename(module)=='nn.GRU' and module.p ~= 0 then
      assert(module.mono, 'TrimZero + BGRUs needs `mono` option!')
   end
   self.temp = torch.Tensor()
   self.gradTemp = torch.Tensor()
end

function TrimZero:recursiveMask(output, input, mask)
   if torch.type(input) == 'table' then
      output = torch.type(output) == 'table' and output or {}
      for k,v in ipairs(input) do
         output[k], mask = self:recursiveMask(output[k], v, mask)
      end
   else
      assert(torch.isTensor(input))
      output = torch.isTensor(output) and output or input.new()
      
      -- make sure mask has the same dimension as the input tensor
      if torch.type(mask) ~= 'torch.LongTensor' then
         local inputSize = input:size():fill(1)
         assert(self.nInputDim)
         if self.batchmode then
            inputSize[1] = input:size(1)
         end
         mask:resize(inputSize)
      end
      
      -- build mask
      if self.batchmode then
         assert(torch.find, 'install torchx package : luarocks install torchx')
         -- use torch.find to convert mask from onehot to indices
         if torch.type(mask) ~= 'torch.LongTensor' then
            if torch.type(mask) == 'torch.CudaTensor' then
               self._maskbyte = self._maskbyte or torch.ByteTensor()
               self._maskbyte:resize(mask:size()):copy(mask)
               mask = self._maskbyte
            end
            mask = torch.LongTensor(torch.find(mask, 0))
         end
         self._maskindices = mask
         if mask:dim() > 0 then
            output:index(input, 1, mask)
         else
            output:index(input, 1, torch.LongTensor{1}):zero()
         end
      else 
         if mask:dim() == 0 or mask:view(-1)[1] == 1 then 
            output:resize(input:size()):zero() 
         else 
            output:resize(input:size()):copy(input) 
         end
      end
   end
   return output, mask
end

function TrimZero:recursiveUnMask(output, input, mask)
   if torch.type(input) == 'table' then
      output = torch.type(output) == 'table' and output or {}
      for k,v in ipairs(input) do
         output[k] = self:recursiveUnMask(output[k], v, mask)
      end
   else
      assert(torch.isTensor(input))
      output = torch.isTensor(output) and output or input.new()
      
      -- make sure output has the same dimension as the mask
      local inputSize = input:size()
      if self.batchmode then
         inputSize[1] = mask:size(1)
      end
      output:resize(inputSize):zero()
      
      -- build mask
      if self.batchmode then
         assert(self._maskindices)
         mask = self._maskindices
         if mask:dim() > 0 then
            output:indexCopy(1, mask, input)
         end
      else
         if mask:view(-1)[1] == 0 then 
            output:copy(input)
         end
      end
   end
   return output
end

function TrimZero:updateOutput(input)
   -- recurrent module input is always the first one
   local rmi = self:recursiveGetFirst(input):contiguous()
   if rmi:dim() == self.nInputDim then
      self.batchmode = false
      rmi = rmi:view(-1) -- collapse dims
   elseif rmi:dim() - 1 == self.nInputDim then
      self.batchmode = true
      rmi = rmi:view(rmi:size(1), -1) -- collapse non-batch dims
   else
      error("nInputDim error: "..rmi:dim()..", "..self.nInputDim)
   end
   
   -- build mask
   local vectorDim = rmi:dim() 
   self._zeroMask = self._zeroMask or rmi.new()
   self._zeroMask:norm(rmi, 2, vectorDim)
   self.zeroMask = self.zeroMask or ((torch.type(rmi) == 'torch.CudaTensor') and torch.CudaTensor() or torch.ByteTensor())
   self._zeroMask.eq(self.zeroMask, self._zeroMask, 0)
   
   -- forward through decorated module
   self.temp = self:recursiveMask(self.temp, input, self.zeroMask)
   output = self.module:updateOutput(self.temp)
   self.output = self:recursiveUnMask(self.output, output, self.zeroMask, true)

   return self.output
end

function TrimZero:updateGradInput(input, gradOutput)
   self.temp = self:recursiveMask(self.temp, input, self.zeroMask)
   self.gradTemp = self:recursiveMask(self.gradTemp, gradOutput, self.zeroMask)

   local gradInput = self.module:updateGradInput(self.temp, self.gradTemp)

   self.gradInput = self:recursiveUnMask(self.gradInput, gradInput, self.zeroMask)

   return self.gradInput
end

function TrimZero:accGradParameters(input, gradOutput, scale)
   self.temp = self:recursiveMask(self.temp, input, self.zeroMask)
   self.module:accGradParameters(self.temp, gradOutput, scale)
end
