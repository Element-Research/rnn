local LookupTableMaskZero, parent = torch.class('nn.LookupTableMaskZero', 'nn.LookupTable')

function LookupTableMaskZero:__init(nIndex, nOutput)
  parent.__init(self, nIndex + 1, nOutput)
end

function LookupTableMaskZero:updateOutput(input)
	self.weight[1]:zero()
   if self.__input and (torch.type(self.__input) ~= torch.type(input)) then
      self.__input = nil -- fixes old casting bug
   end
   self.__input = self.__input or input.new()
   self.__input:resizeAs(input):add(input, 1)
	return parent.updateOutput(self, self.__input)
end

function LookupTableMaskZero:accGradParameters(input, gradOutput, scale)
	parent.accGradParameters(self, self.__input, gradOutput, scale)
end

function LookupTableMaskZero:type(type, cache)
   self.__input = nil
   return parent.type(self, type, cache)
end
