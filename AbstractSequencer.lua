local AbstractSequencer, parent = torch.class("nn.AbstractSequencer", "nn.Container")

function AbstractSequencer:getStepModule(step)
   -- DEPRECATED 27 Oct 2015. Wrap your internal modules into a Recursor instead.
   assert(self.sharedClones, "no sharedClones for type "..torch.type(self))
   assert(step, "expecting step at arg 1")
   local module = self.sharedClones[step]
   if not module then
      module = self.sharedClones[1]:sharedClone()
      self.sharedClones[step] = module
   end
   return module
end

-- AbstractSequencers are expected to handle backwardThroughTime during backward
function AbstractSequencer:backwardThroughTime()
end

function AbstractSequencer:sharedClone(shareParams, shareGradParams, clones, pointers, stepClone)
   -- stepClone is ignored (always false, i.e. uses sharedClone)
   return parent.sharedClone(self, shareParams, shareGradParams, clones, pointers)
end

function AbstractSequencer:backwardOnline(online)
   return
end

-- AbstractSequence handles its own rho internally (dynamically)
function AbstractSequencer:maxBPTTstep(rho)
end

AbstractSequencer.includingSharedClones = nn.AbstractRecurrent.includingSharedClones
AbstractSequencer.type = nn.AbstractRecurrent.type
AbstractSequencer.training = nn.AbstractRecurrent.training
AbstractSequencer.evaluate = nn.AbstractRecurrent.evaluate
AbstractSequencer.reinforce = nn.AbstractRecurrent.reinforce

