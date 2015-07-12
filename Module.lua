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
