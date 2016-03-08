-- example use of nn.Recurrence
require 'rnn'

-- hyper-parameters 
batchSize = 8
rho = 5 -- sequence length
hiddenSize = 7
nIndex = 10
lr = 0.1

-- the internal recurrentModule used by Recurrence
local rm = nn.Sequential() -- input is {x[t], h[t-1]}
   :add(nn.ParallelTable()
      :add(nn.LookupTable(nIndex, hiddenSize)) -- input layer
      :add(nn.Linear(hiddenSize, hiddenSize))) -- recurrent layer
   :add(nn.CAddTable()) -- merge
   :add(nn.Sigmoid()) -- transfer

local rnn = nn.Sequential()
   :add(nn.Recurrence(rm, hiddenSize, 0)) -- similar to nn.Recurrent, but more general, and no startModule
   :add(nn.Linear(hiddenSize, nIndex))
   :add(nn.LogSoftMax())

-- all following code is exactly the same as the simple-sequencer-network.lua script
-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
rnn = nn.Sequencer(rnn)

print(rnn)

-- build criterion

criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

-- build dummy dataset (task is to predict next item, given previous)
sequence_ = torch.LongTensor():range(1,10) -- 1,2,3,4,5,6,7,8,9,10
sequence = torch.LongTensor(100,10):copy(sequence_:view(1,10):expand(100,10))
sequence:resize(100*10) -- one long sequence of 1,2,3...,10,1,2,3...10...

offsets = {}
for i=1,batchSize do
   table.insert(offsets, math.ceil(math.random()*sequence:size(1)))
end
offsets = torch.LongTensor(offsets)

-- training
local iteration = 1
while true do
   -- 1. create a sequence of rho time-steps
   
   local inputs, targets = {}, {}
   for step=1,rho do
      -- a batch of inputs
      inputs[step] = sequence:index(1, offsets)
      -- incement indices
      offsets:add(1)
      for j=1,batchSize do
         if offsets[j] > sequence:size(1) then
            offsets[j] = 1
         end
      end
      targets[step] = sequence:index(1, offsets)
   end
   
   -- 2. forward sequence through rnn
   
   rnn:zeroGradParameters() 
   
   local outputs = rnn:forward(inputs)
   local err = criterion:forward(outputs, targets)
   
   print(string.format("Iteration %d ; NLL err = %f ", iteration, err))

   -- 3. backward sequence through rnn (i.e. backprop through time)
   
   local gradOutputs = criterion:backward(outputs, targets)
   local gradInputs = rnn:backward(inputs, gradOutputs)
   
   -- 4. update
   
   rnn:updateParameters(lr)
   
   iteration = iteration + 1
end
