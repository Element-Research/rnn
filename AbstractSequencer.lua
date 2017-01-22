local AbstractSequencer, parent = torch.class("nn.AbstractSequencer", "nn.Container")

function AbstractSequencer:getStepModule(step)
   error"DEPRECATED 27 Oct 2015. Wrap your internal modules into a Recursor instead"
end

function AbstractSequencer:sharedClone(shareParams, shareGradParams, clones, pointers, stepClone)
   -- stepClone is ignored (always false, i.e. uses sharedClone)
   return parent.sharedClone(self, shareParams, shareGradParams, clones, pointers)
end

-- AbstractSequence handles its own rho internally (dynamically)
function AbstractSequencer:maxBPTTstep(rho)
end

