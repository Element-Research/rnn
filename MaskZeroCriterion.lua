------------------------------------------------------------------------
--[[ MaskZeroCriterion ]]--
-- Decorator that zeros err and gradInputs of the encapsulated criterion
-- for commensurate input rows which are tensors of zeros
------------------------------------------------------------------------
local MaskZeroCriterion, parent = torch.class("nn.MaskZeroCriterion", "nn.Criterion")

function MaskZeroCriterion:__init(criterion, nInputDim)
   parent.__init(self)
   self.criterion = criterion
   assert(torch.isTypeOf(criterion, 'nn.Criterion'))
   assert(torch.type(nInputDim) == 'number', 'Expecting nInputDim number at arg 1')
   self.nInputDim = nInputDim
end

function MaskZeroCriterion:recursiveGetFirst(input)
   if torch.type(input) == 'table' then
      return self:recursiveGetFirst(input[1])
   else
      assert(torch.isTensor(input))
      return input
   end
end

function MaskZeroCriterion:recursiveMask(dst, src, mask)
   if torch.type(src) == 'table' then
      dst = torch.type(dst) == 'table' and dst or {}
      for k,v in ipairs(src) do
         dst[k] = self:recursiveMask(dst[k], v, mask)
      end
   else
      assert(torch.isTensor(src))
      dst = torch.isTensor(dst) and dst or src.new()
   	
      dst:index(src, 1, mask)
   end
   return dst
end

function MaskZeroCriterion:updateOutput(input, target)   
   -- recurrent module input is always the first one
   local rmi = self:recursiveGetFirst(input):contiguous()
   if rmi:dim() == self.nInputDim then
      error("does not support online (i.e. non-batch) mode")
   elseif rmi:dim() - 1 == self.nInputDim then
      rmi = rmi:view(rmi:size(1), -1) -- collapse non-batch dims
   else
      error("nInputDim error: "..rmi:dim()..", "..self.nInputDim)
   end
   
   -- build mask
   local vectorDim = rmi:dim() 
   self._zeroMask = self._zeroMask or rmi.new()
   self._zeroMask:norm(rmi, 2, vectorDim)
   local zeroMask = self._zeroMask
   if torch.isTypeOf(zeroMask, 'torch.CudaTensor') then
      self.__zeroMask = self.__zeroMask or torch.FloatTensor()
      self.__zeroMask:resize(self._zeroMask:size()):copy(self._zeroMask)
      zeroMask = self._zeroMask
   end
  
   self.zeroMask = self.zeroMask or torch.LongTensor()
   self.zeroMask:resize(self._zeroMask:size(1)):zero()
   
   local i, j = 0, 0
   zeroMask:apply(function(norm)
      i = i + 1
      if norm ~= 0 then
         j = j + 1
         self.zeroMask[j] = i
      end
   end)
   self.zeroMask:resize(j)
   
   if j > 0 then
      self.input = self:recursiveMask(self.input, input, self.zeroMask)
      self.target = self:recursiveMask(self.target, target, self.zeroMask)
      
      -- forward through decorated criterion
      self.output = self.criterion:updateOutput(self.input, self.target)
   else
      -- when all samples are masked, then loss is zero (issue 128)
      self.output = 0
   end
   
   return self.output
end

function MaskZeroCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input):zero()
   
   if self.zeroMask:nElement() > 0 then
      self._gradInput = self.criterion:updateGradInput(self.input, self.target)
      self.gradInput:indexCopy(1, self.zeroMask, self._gradInput)
   end
   
   return self.gradInput
end

function MaskZeroCriterion:type(type, ...)
   self.zeroMask = nil
   self._zeroMask = nil
   self.__zeroMask = nil
   self.input = nil
   self.target = nil
   self._gradInput = nil
   
   return parent.type(self, type, ...)
end
