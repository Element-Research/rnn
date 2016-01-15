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

function Module:stepClone(shareParams, shareGradParams)
   return self:sharedClone(shareParams, shareGradParams, true)
end

function Module:backwardOnline()
   print("Deprecated Jan 6, 2016. By default rnn now uses backwardOnline, so no need to call this method")
end

-- calls setOutputStep on all component AbstractRecurrent modules
-- used by Recursor() after calling stepClone.
-- this solves a very annoying bug...
function Module:setOutputStep(step)
   if self.modules then
      for i,module in ipairs(self.modules) do
         module:setOutputStep(step)
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
