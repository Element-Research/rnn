------------------------------------------------------------------------
--[[ MuFuRu - Multi-Function Recurrent Unit ]]--
-- Author: Jonathan Uesato
-- License: LICENSE.2nd.txt

-- Ref. A.: http://arxiv.org/pdf/1606.03002v1.pdf
------------------------------------------------------------------------

local MuFuRu, parent = torch.class('nn.MuFuRu', 'nn.GRU')

local SqrtDiffLayer = nn.Sequential()
                        :add(nn.CSubTable())
                        :add(nn.Abs())
                        :add(nn.Sqrt())
                        :add(nn.MulConstant(0.25))

local MaxLayer = nn.Sequential()
  :add(nn.MapTable(nn.Unsqueeze(1)))
  :add(nn.JoinTable(1))
  :add(nn.Max(1))

local MinLayer = nn.Sequential()
  :add(nn.MapTable(nn.Unsqueeze(1)))
  :add(nn.JoinTable(1))
  :add(nn.Min(1))

-- all operations take a table {oldState, newState} and return newState
_operations = {
   max = MaxLayer,
   keep = nn.SelectTable(1),
   replace = nn.SelectTable(2),
   mul = nn.CMulTable(),
   min = MinLayer,
   diff = nn.CSubTable(),
   forget = nn.Sequential():add(nn.SelectTable(1)):add(nn.MulConstant(0.0)),
   sqrt_diff = SqrtDiffLayer
}

function MuFuRu:__init(inputSize, outputSize, ops, rho)
   -- Use all ops by default. To replicate GRU, use keep and replace only.
   self.ops = ops or {'keep', 'replace', 'mul', 'diff', 'forget', 'sqrt_diff', 'max', 'min'}
   self.num_ops = #self.ops
   self.operations = {}
   for i=1,self.num_ops do
      self.operations[i] = _operations[self.ops[i]]
   end
   self.inputSize = inputSize
   self.outputSize = outputSize
   parent.__init(self, inputSize, outputSize, rho or 9999)
end

-------------------------- factory methods -----------------------------
function MuFuRu:buildModel()
   -- input : {input, prevOutput}
   -- output : output

   local nonBatchDim = 2
   -- resetGate takes {input, prevOutput} to resetGate
   local resetGate = nn.Sequential()
      :add(nn.ParallelTable()
         :add(nn.Linear(self.inputSize, self.outputSize), false)
         :add(nn.Linear(self.outputSize, self.outputSize))
      )
      :add(nn.CAddTable())
      :add(nn.Sigmoid())

   -- Feature takes {input, prevOutput, reset} to feature
   local featureVec = nn.Sequential()
      :add(nn.ConcatTable()
         :add(nn.SelectTable(1))
         :add(nn.Sequential()
            :add(nn.NarrowTable(2,2))
            :add(nn.CMulTable())
         )
      )
      :add(nn.JoinTable(nonBatchDim)) -- [x_t, r dot s_t-1]
      :add(nn.Linear(self.inputSize + self.outputSize, self.outputSize))
      :add(nn.Sigmoid())

   -- opWeights takes {input, prevOutput, reset} to opWeights.
   -- Note that reset is not used
   local opWeights = nn.Sequential()
      :add(nn.NarrowTable(1,2))
      :add(nn.JoinTable(nonBatchDim)) -- k_t
      :add(nn.Linear(self.inputSize + self.outputSize, self.num_ops * self.outputSize)) --p^_t
      :add(nn.View(self.num_ops, self.outputSize):setNumInputDims(1))
      :add(nn.Transpose({1,2}))
      :add(nn.SoftMax()) --p_t

   -- all_ops takes {oldState, newState} to {newState1, newState2, ...newStateN}
   local all_ops = nn.ConcatTable()
   for i=1,self.num_ops do
      -- an operation is any layer taking {prevHidden, featureVec} to newState
      all_ops:add(self.operations[i])
   end

   local all_op_activations = nn.Sequential()
      :add(nn.NarrowTable(1,2))
      :add(all_ops)
      :add(nn.MapTable(nn.Unsqueeze(1)))
      :add(nn.JoinTable(1,3))

   -- combine_ops takes {prevHidden, featureVec, opWeights} to nextHidden
   local combine_ops = nn.Sequential()
      :add(nn.ConcatTable()
         :add(all_op_activations)
         :add(nn.SelectTable(3))
      )
      :add(nn.CMulTable())
      :add(nn.Sum(1,3))

   local cell = nn.Sequential()
      :add(nn.ConcatTable()
         :add(nn.SelectTable(1))
         :add(nn.SelectTable(2))
         :add(resetGate)
      ) -- {input,prevOutput,reset}
      :add(nn.ConcatTable()
         :add(nn.SelectTable(2))
         :add(featureVec)
         :add(opWeights)
      ) -- {prevOutput, v_t, opWeights}
      :add(combine_ops)
   return cell
end

-- Factory methods are inherited from GRU

function MuFuRu:__tostring__()
   local op_str = '{ '
   for i=1,self.num_ops do
      op_str = op_str .. self.ops[i] .. ' '
   end
   op_str = op_str .. '}'
   return (string.format('%s(%d -> %d) ', torch.type(self), self.inputSize, self.outputSize)) .. op_str
end

function MuFuRu:migrate(params)
   error"Migrate not supported for MuFuRu"
end
