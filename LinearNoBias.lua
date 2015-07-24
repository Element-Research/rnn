------------------------------------------------------------------------
--[[ LinearNoBias ]]--
-- Subclass of nn.Linear with no bias term
-- The implementation is maximally independent of the implementation of nn.Linear
-- (only assumption is that the bias and its gradient is stored in self.bias and self.gradBias)
-- but is slightly inefficient (the bias term is still stored in memory and its
-- gradient computed at every step).
------------------------------------------------------------------------
local LinearNoBias, Linear = torch.class('nn.LinearNoBias', 'nn.Linear')

function LinearNoBias:__init(inputSize, outputSize)
    Linear.__init(self, inputSize, outputSize)
    self.bias:zero()
end

function LinearNoBias:accGradParameters(input, gradOutput, scale)
    Linear.accGradParameters(self, input, gradOutput, scale)
    self.gradBias:zero()
end

