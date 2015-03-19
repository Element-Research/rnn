local Module = nn.Module

-- returns a list of modules
function Module:listModules()
   local function tinsert(to, from)
      if torch.type(from) == 'table' then
         for i=1,#from do
            tinsert(to,from[i])
         end
      else
         table.insert(to,from)
      end
   end
   -- include self first
   local modules = {self}
   if self.modules then
      for i=1,#self.modules do
         local modulas = self.modules[i]:listModules()
         if modulas then
            tinsert(modules,modulas)
         end
      end
   end
   return modules
end
