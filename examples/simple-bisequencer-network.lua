require 'rnn'

-- hyper-parameters 
batchSize = 8
rho = 5 -- sequence length
hiddenSize = 7
nIndex = 10
lr = 0.1


-- forward rnn
-- build simple recurrent neural network
local fwd = nn.Recurrent(
   hiddenSize, nn.LookupTable(nIndex, hiddenSize), 
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
   rho
)

-- backward rnn (will be applied in reverse order of input sequence)
local bwd = fwd:clone()
bwd:reset() -- reinitializes parameters

-- merges the output of one time-step of fwd and bwd rnns.
-- You could also try nn.AddTable(), nn.Identity(), etc.
local merge = nn.JoinTable(1, 1) 

-- we use BiSequencerLM because this is a language model (previous and next words to predict current word).
-- If we used BiSequencer, x[t] would be used to predict y[t] = x[t] (which is cheating).
-- Note that bwd and merge argument are optional and will default to the above.
local brnn = nn.BiSequencerLM(fwd, bwd, merge)

local rnn = nn.Sequential()
   :add(brnn) 
   :add(nn.Sequencer(nn.Linear(hiddenSize*2, nIndex))) -- times two due to JoinTable
   :add(nn.Sequencer(nn.LogSoftMax()))

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
