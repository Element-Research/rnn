local AbstractSequencer, parent = torch.class("nn.AbstractSequencer", "nn.Container")

function AbstractSequencer:getStepModule(step)
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

AbstractSequencer.includingSharedClones = nn.AbstractRecurrent.includingSharedClones
AbstractSequencer.type = nn.AbstractRecurrent.type
AbstractSequencer.training = nn.AbstractRecurrent.training
AbstractSequencer.evaluate = nn.AbstractRecurrent.evaluate
AbstractSequencer.reinforce = nn.AbstractRecurrent.reinforce

