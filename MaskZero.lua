------------------------------------------------------------------------
--[[ MaskZero ]]--
-- Decorator that zeroes the output rows of the encapsulated module
-- for commensurate input rows which are tensors of zeros
------------------------------------------------------------------------
local MaskZero, parent = torch.class("nn.MaskZero", "nn.Decorator")

function MaskZero:__init(module, nInputDim, silent)
   parent.__init(self, module)
   assert(torch.isTypeOf(module, 'nn.Module'))
   if torch.isTypeOf(module, 'nn.AbstractRecurrent') and not silent then
      print("Warning : you are most likely using MaskZero the wrong way. "
      .."You should probably use AbstractRecurrent:maskZero() so that "
      .."it wraps the internal AbstractRecurrent.recurrentModule instead of "
      .."wrapping the AbstractRecurrent module itself.") 
   end
   assert(torch.type(nInputDim) == 'number', 'Expecting nInputDim number at arg 1')
   self.nInputDim = nInputDim
end

function MaskZero:recursiveGetFirst(input)
   if torch.type(input) == 'table' then
      return self:recursiveGetFirst(input[1])
   else
      assert(torch.isTensor(input))
      return input
   end
end

function MaskZero:recursiveMask(output, input, mask)
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
      local zeroMask = mask:expandAs(input)
      output:resizeAs(input):copy(input)
      output:maskedFill(zeroMask, 0)
   end
   return output
end

function MaskZero:updateOutput(input)   
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
   local output = self.module:updateOutput(input)

   self.output = self:recursiveMask(self.output, output, self.zeroMask)
   return self.output
end

function MaskZero:updateGradInput(input, gradOutput)
   -- zero gradOutputs before backpropagating through decorated module
   self.gradOutput = self:recursiveMask(self.gradOutput, gradOutput, self.zeroMask)
   
   self.gradInput = self.module:updateGradInput(input, self.gradOutput)
   return self.gradInput
end

function MaskZero:type(type, ...)
   self.zeroMask = nil
   self._zeroMask = nil
   
   return parent.type(self, type, ...)
end