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

-- Toggle to feed long sequences using multiple forwards.
-- 'eval' only affects evaluation (recommended for RNNs)
-- 'train' only affects training
-- 'neither' affects neither training nor evaluation
-- 'both' affects both training and evaluation (recommended for LSTMs)
-- Essentially, forget() isn't called on rnn module when remember is on
function AbstractSequencer:remember(remember)
   self._remember = (remember == nil) and 'both' or remember
   local _ = require 'moses'
   assert(_.contains({'both','eval','train','neither'}, self._remember),
      "AbstractSequencer : unrecognized value for remember : "..self._remember)
   return self
end

function AbstractSequencer:hasMemory()
   local _ = require 'moses'
   if (self.train ~= false) and _.contains({'both','train'}, self._remember) then -- train (defaults to nil...)
      return true
   elseif (self.train == false) and _.contains({'both','eval'}, self._remember) then -- evaluate
      return true
   else
      return false
   end
end

