local CopyGrad, _ = torch.class('nn.CopyGrad', 'nn.Identity')

function CopyGrad:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   return self.gradInput
end
