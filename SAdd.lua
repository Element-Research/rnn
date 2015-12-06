local SAdd, parent = torch.class('nn.SAdd', 'nn.Module')

function SAdd:__init(addend, negate)
   parent.__init(self)
  
   self.addend = addend
   self.negate = (negate == nil) and false or negate
end

function SAdd:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.output = self.output + self.addend
   if self.negate then
      self.output = -self.output
   end
   return self.output
end

function SAdd:updateGradInput(input, gradOutput)
   if self.gradInput then
      self.gradInput:resizeAs(gradOutput):copy(gradOutput) 
   else
      self.gradInput = torch.Tensor():resizeAs(gradOutput):copy(gradOutput) 
   end
   if self.negate then
      self.gradInput = -self.gradInput
   end
   return self.gradInput
end