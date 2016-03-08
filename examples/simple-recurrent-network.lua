require 'rnn'

-- hyper-parameters 
batchSize = 8
rho = 5 -- sequence length
hiddenSize = 7
nIndex = 10
lr = 0.1


-- build simple recurrent neural network
local r = nn.Recurrent(
   hiddenSize, nn.LookupTable(nIndex, hiddenSize), 
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
   rho
)

local rnn = nn.Sequential()
   :add(r)
   :add(nn.Linear(hiddenSize, nIndex))
   :add(nn.LogSoftMax())

-- wrap the non-recurrent module (Sequential) in Recursor.
-- This makes it a recurrent module
-- i.e. Recursor is an AbstractRecurrent instance
rnn = nn.Recursor(rnn, rho)

print(rnn)

-- build criterion

criterion = nn.ClassNLLCriterion()

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
   rnn:forget() -- forget all past time-steps
   
   local outputs, err = {}, 0
   for step=1,rho do
      outputs[step] = rnn:forward(inputs[step])
      err = err + criterion:forward(outputs[step], targets[step])
   end
   
   print(string.format("Iteration %d ; NLL err = %f ", iteration, err))

   -- 3. backward sequence through rnn (i.e. backprop through time)
   
   local gradOutputs, gradInputs = {}, {}
   for step=rho,1,-1 do -- reverse order of forward calls
      gradOutputs[step] = criterion:backward(outputs[step], targets[step])
      gradInputs[step] = rnn:backward(inputs[step], gradOutputs[step])
   end
   
   -- 4. update
   
   rnn:updateParameters(lr)
   
   iteration = iteration + 1
end
