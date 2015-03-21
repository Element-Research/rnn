local Module = nn.Module

Module.__parameters__ = {'weight', 'bias'}
Module.__gradParameters__ = {'gradWeight', 'gradBias'}

-- TODO make this recursive (for table params)
function Module:sharedClone(shareParams, shareGradParams)
   shareParams = (shareParams == nil) and true or shareParams
   shareGradParams = (shareGradParams == nil) and true or shareGradParams
   
   local moduleClones, modules
   if self.modules then
      moduleClones = {}
      for i,module in ipairs(self.modules) do
         moduleClones[i] = module:sharedClone(shareParams, shareGradParams)
      end
      modules = self.modules
      self.modules = nil -- to prevent recloning
   end
   
   local params, pointers = {}, {}
   if shareParams then
      for i,paramName in ipairs(self.__parameters__) do
         local param = self[paramName]
         if param then
            params[paramName] = param
            self[paramName] = nil
            if param:storage() then
               pointers[torch.pointer(param:storage():data())] = true
            end
         end
      end
   end
   
   if shareGradParams then
      for i,paramName in ipairs(self.__gradParameters__) do
         local gradParam = self[paramName]
         if gradParam then
            params[paramName] = gradParam
            self[paramName] = nil
            if gradParam:storage() then
               pointers[torch.pointer(gradParam:storage():data())] = true
            end
         end
      end
   end
   
   -- find all the tensors that share storage with the shared params
   for paramName, param in pairs(self) do
      if torch.isTensor(param) and param:storage() then
         if pointers[torch.pointer(param:storage():data())] then
            params[paramName] = param
            self[paramName] = nil
         end
      end
   end
   
   -- clone everything but parameters and/or gradients
   local clone = self:clone()
   
   for paramName, param in pairs(params) do
      assert(self[paramName] == nil)
      self[paramName] = param
      clone[paramName] = param.new():set(param)
   end
   
   if moduleClones then
      assert(self.modules == nil)
      self.modules = modules
      clone.modules = moduleClones
   end
   return clone
end      

-- for preserving shared params created with sharedClones
function Module:sharedType(type, castmap)
   assert(type, 'Module:sharedType must provide a type to convert to')
   -- key: pointer to old storage 
   -- value : new storage
   castmap = castmap or {} --contains torch.Storage instances
   
   local function recursiveType(param, type_str)
      if torch.type(param) == 'table' then
         for i = 1, #param do
            param[i] = recursiveType(param[i], type_str)
         end
      else
         if torch.isTensor(param) then
            if param:storage() then
               local pointer = torch.pointer(param:storage():data())
               local storage = castmap[pointer]
               if not storage then
                  local _param = param
                  -- cast entire storage
                  param = param.new(param:storage()):type(type_str)
                  param:set(param:storage(), _param:storageOffset(), _param:size(), _param:stride())
                  castmap[pointer] = param:storage()
               else
                  -- set to point to existing storage
                  local _param = param
                  param = torch.getmetatable(type_str).new()
                  param:set(storage, _param:storageOffset(), _param:size(), _param:stride())
               end
            else
               param = param:type(type_str)
            end
         end
      end
      return param
   end
   
   -- find all tensors and convert them
   for key,param in pairs(self) do
      -- Many modules (like CDivTable) have output or gradInput fields which
      -- are table's of tensors.  To be general we need to recursively
      -- cast fields that may be nested tables.
      if key ~= 'modules' then
        self[key] = recursiveType(self[key], type)
      end
   end
   -- find submodules in classic containers 'modules'
   if self.modules then
      for _,module in ipairs(self.modules) do
         module:sharedType(type, castmap)
      end
   end
   return self
end


function Module:float(shared)
   local type = 'torch.FloatTensor'
   return shared and self:sharedType(type) or self:type(type)
end

function Module:double(shared)
   local type = 'torch.DoubleTensor'
   return shared and self:sharedType(type) or self:type(type)
end

function Module:cuda(shared)
   local type = 'torch.CudaTensor'
   return shared and self:sharedType(type) or self:type(type)
end
