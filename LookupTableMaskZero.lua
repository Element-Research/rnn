local LookupTableMaskZero, parent = torch.class('nn.LookupTableMaskZero', 'nn.LookupTable')

function LookupTableMaskZero:__init(nIndex, nOutput)
  parent.__init(self, nIndex + 1, nOutput)
end

function LookupTableMaskZero:updateOutput(input)
	self.weight[1]:zero()
   self._minput = self._minput or input.new()
   self._minput:add(input, 1)
	return parent.updateOutput(self, self._minput)
end

function LookupTableMaskZero:accGradParameters(input, gradOutput, scale)
	parent.accGradParameters(self, self._minput, gradOutput, scale)
end
