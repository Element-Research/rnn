local Module = nn.Module 

-- You can use this to manually forget past memories in AbstractRecurrent instances
function Module:forget()
   if self.modules then
      for i,module in ipairs(self.modules) do
         module:forget()
      end
   end
   return self
end

-- Used by nn.Sequencers
function Module:remember(remember)
   if self.modules then
      for i, module in ipairs(self.modules) do
         module:remember(remember)
      end
   end
   return self
end

-- Calls backwardThroughTime for all encapsulated modules
function Module:backwardThroughTime()
   if self.modules then
      for i, module in ipairs(self.modules) do
         module:backwardThroughTime()
      end
   end
end

function Module:stepClone(shareParams, shareGradParams, clones, pointers)
   return self:sharedClone(shareParams, shareGradParams, clones, pointers, true)
end

-- notifies all AbstractRecurrent instances not wrapped by an AbstractSequencer
-- that the backward calls will be handled online (in reverse order of forward time).
function Module:backwardOnline(online)
   if self.modules then
      for i, module in ipairs(self.modules) do
         module:backwardOnline(online)
      end
   end
end

-- set the maximum number of backpropagation through time (BPTT) time-steps
function Module:maxBPTTstep(rho)
   if self.modules then
      for i, module in ipairs(self.modules) do
         module:maxBPTTstep(rho)
      end
   end
end
