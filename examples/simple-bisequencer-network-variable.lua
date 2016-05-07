-- Example BLSTM for variable-length sequences
require 'rnn'

torch.manualSeed(0)
math.randomseed(0)

-- hyper-parameters 
batchSize = 8
rho = 10 -- sequence length
hiddenSize = 5
nIndex = 10
lr = 0.1
maxIter = 100

local sharedLookupTable = nn.LookupTableMaskZero(nIndex, hiddenSize)

-- forward rnn
local fwd = nn.Sequential()
   :add(sharedLookupTable)
   :add(nn.FastLSTM(hiddenSize, hiddenSize):maskZero(1))

-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
fwdSeq = nn.Sequencer(fwd)

-- backward rnn (will be applied in reverse order of input sequence)
local bwd = nn.Sequential()
   :add(sharedLookupTable:sharedClone())
   :add(nn.FastLSTM(hiddenSize, hiddenSize):maskZero(1))
bwdSeq = nn.Sequencer(bwd)

-- merges the output of one time-step of fwd and bwd rnns.
-- You could also try nn.AddTable(), nn.Identity(), etc.
local merge = nn.JoinTable(1, 1) 
mergeSeq = nn.Sequencer(merge)

-- Assume that two input sequences are given (original and reverse, both are right-padded).
-- Instead of ConcatTable, we use ParallelTable here.
local parallel = nn.ParallelTable()
parallel:add(fwdSeq):add(bwdSeq)
local brnn = nn.Sequential()
   :add(parallel)
   :add(nn.ZipTable())
   :add(mergeSeq)

local rnn = nn.Sequential()
   :add(brnn) 
   :add(nn.Sequencer(nn.MaskZero(nn.Linear(hiddenSize*2, nIndex), 1))) -- times two due to JoinTable
   :add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(), 1)))

print(rnn)

-- build criterion

criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1))

-- build dummy dataset (task is to predict next item, given previous)
sequence_ = torch.LongTensor():range(1,10) -- 1,2,3,4,5,6,7,8,9,10
sequence = torch.LongTensor(100,10):copy(sequence_:view(1,10):expand(100,10))
sequence:resize(100*10) -- one long sequence of 1,2,3...,10,1,2,3...10...

offsets = {}
maxStep = {}
for i=1,batchSize do
   table.insert(offsets, math.ceil(math.random()*sequence:size(1)))
   -- variable length for each sample
   table.insert(maxStep, math.random(rho))
end
offsets = torch.LongTensor(offsets)

-- training
for iteration = 1, maxIter do
   -- 1. create a sequence of rho time-steps
   
   local inputs, inputs_rev, targets = {}, {}, {}
   for step=1,rho do
      -- a batch of inputs
      inputs[step] = sequence:index(1, offsets)
      -- increment indices
      offsets:add(1)
      for j=1,batchSize do
         if offsets[j] > sequence:size(1) then
            offsets[j] = 1
         end
      end
      targets[step] = sequence:index(1, offsets)
      -- padding
      for j=1,batchSize do
         if step > maxStep[j] then
            inputs[step][j] = 0
            targets[step][j] = 0
         end
      end
   end

   -- reverse
   for step=1,rho do
      inputs_rev[step] = torch.LongTensor(batchSize)
      for j=1,batchSize do
         if step <= maxStep[j] then
            inputs_rev[step][j] = inputs[maxStep[j]-step+1][j]
         else
            inputs_rev[step][j] = 0
         end
      end
   end
   
   -- 2. forward sequence through rnn
   
   rnn:zeroGradParameters() 

   local outputs = rnn:forward({inputs, inputs_rev})
   local err = criterion:forward(outputs, targets)
   
   local correct = 0
   local total = 0
   for step=1,rho do
      probs = outputs[step]
      _, preds = probs:max(2)
      for j=1,batchSize do
         local cur_x = inputs[step][j]
         local cur_y = targets[step][j]
         local cur_t = preds[j][1]
         -- print(string.format("x=%d ; y=%d ; pred=%d", cur_x, cur_y, cur_t))
         if step <= maxStep[j] then
             if cur_y == cur_t then correct = correct + 1 end
             total = total + 1
         end
      end
   end

   local acc = correct*1.0/total
   print(string.format("Iteration %d ; NLL err = %f ; ACC = %.2f ", iteration, err, acc))

   -- 3. backward sequence through rnn (i.e. backprop through time)
   
   local gradOutputs = criterion:backward(outputs, targets)
   local gradInputs = rnn:backward({inputs, inputs_rev}, gradOutputs)
   
   -- 4. update
   
   rnn:updateParameters(lr)
   
end
