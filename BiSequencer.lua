------------------------------------------------------------------------
--[[ BiSequencer ]]--
-- Encapsulates forward, backward and merge modules. 
-- Input is a sequence (a table) of tensors.
-- Output is a sequence (a table) of tensors of the same length.
-- Applies a forward rnn to each element in the sequence in
-- forward order and applies a backward rnn in reverse order.
-- For each step, the outputs of both rnn are merged together using
-- the merge module (defaults to nn.JoinTable(1,1)).
-- The sequences in a batch must have the same size.
-- But the sequence length of each batch can vary.
-- It is implemented by decorating a structure of modules that makes 
-- use of 3 Sequencers for the forward, backward and merge modules.
------------------------------------------------------------------------
local BiSequencer, parent = torch.class('nn.BiSequencer', 'nn.AbstractSequencer')

function BiSequencer:__init(forward, backward, merge)
   
   if not torch.isTypeOf(forward, 'nn.Module') then
      error"BiSequencer: expecting nn.Module instance at arg 1"
   end
   self.forwardModule = forward
   
   self.backwardModule = backward
   if not self.backwardModule then
      self.backwardModule = forward:clone()
      self.backwardModule:reset()
   end
   if not torch.isTypeOf(self.backwardModule, 'nn.Module') then
      error"BiSequencer: expecting nn.Module instance at arg 2"
   end
   
   if torch.type(merge) == 'number' then
      self.mergeModule = nn.JoinTable(1, merge)
   elseif merge == nil then
      self.mergeModule = nn.JoinTable(1, 1)
   elseif torch.isTypeOf(merge, 'nn.Module') then
      self.mergeModule = merge
   else
      error"BiSequencer: expecting nn.Module or number instance at arg 3"
   end
   
   self.fwdSeq = nn.Sequencer(self.forwardModule)
   self.bwdSeq = nn.Sequencer(self.backwardModule)
   self.mergeSeq = nn.Sequencer(self.mergeModule)
   
   local backward = nn.Sequential()
   backward:add(nn.ReverseTable()) -- reverse
   backward:add(self.bwdSeq)
   backward:add(nn.ReverseTable()) -- unreverse
   
   local concat = nn.ConcatTable()
   concat:add(self.fwdSeq):add(backward)
   
   local brnn = nn.Sequential()
   brnn:add(concat)
   brnn:add(nn.ZipTable())
   brnn:add(self.mergeSeq)
   
   parent.__init(self)
   
   self.output = {}
   self.gradInput = {}
   
   self.module = brnn
   -- so that it can be handled like a Container
   self.modules[1] = brnn
end

-- multiple-inheritance
nn.Decorator.decorate(BiSequencer)
