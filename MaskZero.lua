------------------------------------------------------------------------
--[[ MaskZero ]]--
-- Decorator that zeroes the output state of the encapsulated module
-- for inputs which are zero vectors

-- Emcapsulated module must have the signature of one of the 
-- AbstractRecurrent's recurrentModule implementation :
-- LSTM: output, cell = unpack(recurrentModule:updateOutput{input, prevOutput, prevCell})
-- Recurrent: output = recurrentModule:updateOutput{input, self.output}

-- Zero vectors (i.e. padding) must be at the beginning of the sequence
-- because this decorator will otherwise reset the recurrentModule
-- in the middle or after the sequence
-- TODO add assertion in case padding in uncountered after non padding ?
------------------------------------------------------------------------
local MaskZero, parent = torch.class("nn.MaskZero", "nn.Decorator")

function MaskZero:updateOutput(input)
   self.output = self.module:updateOutput(input)
   
   -- recurrent module input is always the first one
   local rmi = input[1]
   -- build mask once
   local vectorDim = rmi:dim() -- works for batch and non batch
   self._zeroMask = torch.norm(rmi, 2, vectorDim):eq(0)
   -- building mask with code bellow is slower
   -- self._zeroMask = torch.eq(({torch.min(rmi, vectorDim)})[1], ({torch.max(rmi, vectorDim)})[1])

   -- build mask and use for output (and cell)
   if torch.type(self.output) == 'table' then
   	-- LSTM
   	self._zeroMask = self._zeroMask:expandAs(self.output[1])
   	-- output
   	self.output[1]:maskedFill(self._zeroMask, 0)
   	-- cell: i think zeroing the cell is also mandatory
   	self.output[2]:maskedFill(self._zeroMask, 0)
   else
   	-- Recurrent
   	self._zeroMask = self._zeroMask:expandAs(self.output)
   	-- output only
   	self.output:maskedFill(self._zeroMask, 0)
   end
   return self.output
end

function MaskZero:updateGradInput(input, gradOutput)
   self.gradInput = self.module:updateGradInput(input, gradOutput)
   for i=1, #self.gradInput do 
   	self.gradInput[i]:maskedFill(self._zeroMask, 0)
   end
   return self.gradInput
end
