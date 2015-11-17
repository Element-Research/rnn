local LookupTableMaskZero, parent = torch.class('nn.LookupTableMaskZero', 'nn.LookupTable')

function LookupTableMaskZero:__init(nIndex, nOutput)
  parent.__init(self, nIndex + 1, nOutput)
end

function LookupTableMaskZero:updateOutput(input)
	self.weight[1]:zero()
	return parent.updateOutput(self, torch.add(input, 1))
end

-- No need to override accGradParameters because input is cached
-- by nn.LookupTable implementation and gradOuput is already as expected
--[[function LookupTable:accGradParameters(input, gradOutput, scale)
	parent.accGradParameters(self, torch.add(input, 1), gradOutput, scale)
end--]]
