------------------------------------------------------------------------
--[[ TrimZero ]]--
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

-- Zero vectors (i.e. padding) must be at the beginning of the sequence
-- because this decorator will otherwise reset the recurrentModule
-- in the middle or after the sequence
-- TODO add assertion in case padding in uncountered after non padding ?
------------------------------------------------------------------------
local TrimZero, parent = torch.class("nn.TrimZero", "nn.MaskZero")

require 'torchx'

function TrimZero:__init(module, nInputDim, silent)
   parent.__init(self, module, nInputDim, silent)
end

function TrimZero:recursiveMask(output, input, mask)
   if torch.type(input) == 'table' then
      output = torch.type(output) == 'table' and output or {}
      for k,v in ipairs(input) do
         output[k] = self:recursiveMask(output[k], v, mask)
      end
   else
      assert(torch.isTensor(input))
      output = torch.isTensor(output) and output or input.new()
      
      -- make sure mask has the same dimenion as the input tensor
      local inputSize = input:size():fill(1)
      if input:dim() - 1 == self.nInputDim then
         inputSize[1] = input:size(1)
      end
      mask:resize(inputSize)
      -- build mask
      if input:dim() - 1 == self.nInputDim then
         output:resizeAs(input):copy(input)
         assert(torch.find, 'install torchx package : luarocks install torchx')
         local indexes = torch.find(mask, 0)
         if 0 < #indexes then
            output = output:index(1, torch.LongTensor(indexes))
         else
            output = output:index(1, torch.LongTensor{1}):zero()
         end
      else
         if mask[1] == 1 then output = input:clone():zero() 
                         else output = input end
      end
   end
   return output
end

function TrimZero:recursiveUnMask(output, input, mask)
   if torch.type(input) == 'table' then
      output = torch.type(output) == 'table' and output or {}
      for k,v in ipairs(input) do
         output[k] = self:recursiveUnMask(output[k], v, mask)
      end
   else
      assert(torch.isTensor(input))
      local _input = input:clone()
      output = torch.isTensor(output) and output or input.new()
      
      -- make sure output has the same dimension as the mask
      local inputSize = input:size()
      if input:dim() - 1 == self.nInputDim then
         inputSize[1] = mask:size(1)
      end
      output:resize(inputSize):zero()
      -- build mask
      if input:dim() - 1 == self.nInputDim then
         assert(torch.find, 'install torchx package : luarocks install torchx')
         local indexes = torch.find(mask, 0)
         if 0 < #indexes then
            for i = 1,#indexes do
               output[indexes[i]]:copy(_input[i])
            end
         end
      else
         if mask[1] == 0 then output:copy(_input) end
      end
   end
   return output
end

function TrimZero:updateOutput(input)
   -- recurrent module input is always the first one
   local rmi = self:recursiveGetFirst(input):contiguous()
   if rmi:dim() == self.nInputDim then
      rmi = rmi:view(-1) -- collapse dims
   elseif rmi:dim() - 1 == self.nInputDim then
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
   input = self:recursiveMask(input, input, self.zeroMask)
   output = self.module:updateOutput(input)
   self.output = self:recursiveUnMask(self.output, output, self.zeroMask, true)

   return self.output
end

function TrimZero:updateGradInput(input, gradOutput)
   input = self:recursiveMask(input, input, self.zeroMask)
   gradOutput = self:recursiveMask(gradOutput, gradOutput, self.zeroMask)

   self.gradInput = self.module:updateGradInput(input, gradOutput)

   self.gradInput = self:recursiveUnMask(self.gradInput, self.gradInput, self.zeroMask)

   return self.gradInput
end

function TrimZero:accGradParameters(input, gradOutput, scale)
   input = self:recursiveMask(input, input, self.zeroMask)
   self.module:accGradParameters(input, gradOutput, scale)
end