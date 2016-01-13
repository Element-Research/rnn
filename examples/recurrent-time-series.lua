-- This is a multi-variate version of the time-series example 
-- at https://github.com/Element-Research/rnn#rnn.Recurrent
require 'rnn'
--require 'dp'

rho = 5 -- maximum number of time steps for BPTT
inputSize = 6
hiddenSize = 10
outputSize = 6
nIndex = 100
-- RNN
r = nn.Recurrent(
   hiddenSize, -- size of output
   nn.Linear(inputSize, hiddenSize), -- input layer
   nn.Linear(hiddenSize, hiddenSize), -- recurrent layer
   nn.Sigmoid(), -- transfer function
   rho
)

rnn = nn.Sequential()
   :add(r)
   :add(nn.Linear(hiddenSize, outputSize))

criterion = nn.MSECriterion() 

-- dummy dataset (task is to predict next vector, given previous)
-- evaluate normal distribution function, vX is used as both input X and output Y to save memory
local function evalPDF(vMean, vSigma, vX)
   for i=1,vMean:size(1) do
      local b = (vX[i]-vMean[i])/vSigma[i]
      vX[i] = math.exp(-b*b/2)/(vSigma[i]*math.sqrt(2*math.pi))
   end
   return vX
end
vBias = torch.randn(inputSize)
vMean = torch.Tensor(inputSize):fill(5)
vSigma = torch.linspace(1,inputSize/2.0,inputSize)
sequence = torch.Tensor(nIndex, inputSize)
j = 0
for i=1,nIndex do
  sequence[{i,{}}]:fill(j)
  evalPDF(vMean, vSigma, sequence[{i,{}}])
  sequence[{i,{}}]:add(vBias)
  j = j + 1
  if j>10 then j = 0 end
end
print('Sequence:'); print(sequence)

-- batch mode
batchSize = 1 --8
offsets = {}
for i=1,batchSize do
   --table.insert(offsets, 1)
   -- randomize batch input
   table.insert(offsets, math.ceil(math.random()*batchSize))
end
offsets = torch.LongTensor(offsets)
print(offsets)

-- wrap rnn in to a Recursor
rnn = nn.Recursor(rnn, rho)
-- rnn uses backwardOnline by default
--rnn:backwardOnline()
print(rnn)

updateInterval = 4 -- note that updateInterval < rho
lr = 0.001 -- learning rate
step = 0 -- step counter
minErr = outputSize
inputs, outputs, targets = {}, {}, {}
kErrs = torch.Tensor(sequence:size(1)-1):fill(0)
minK = 0
nEpochs = 1
avgErrs = torch.Tensor(nEpochs):fill(0)
for k = 1, nEpochs do --while true do
   for j = 1, sequence:size(1)-1 do
      step = step + 1

      -- forward
      local rinput = sequence:index(1, offsets)
      --print(rinput)
      local routput = rnn:forward(rinput)
      --print(#routput)

      -- increase indices by 1
      offsets:add(1)
      for i=1,batchSize do
         if offsets[i] > nIndex then
            offsets[i] = 1
         end
      end

      -- target
      local target = sequence:index(1, offsets)

      -- report errors
      local err = criterion:forward(routput, target)
      print('Step: ' .. step .. ' Err: '.. err)
      print(' Input:  ', rinput); print(' Output: ', routput); print(' Target: ', target)
      kErrs[j] = err

      -- save these for BPTT
      table.insert(inputs, rinput)
      table.insert(outputs, routput)
      table.insert(targets, target)

      -- backward
      if step % updateInterval == 0 then
         -- backpropagates through time (BPTT) :
         for istep = updateInterval,1,-1 do
            local gradOutput = criterion:backward(outputs[istep], targets[istep])
            print(gradOutput)
            print(r.step, r.updateGradInputStep)
            rnn:backward(inputs[istep], gradOutput)
         end
         -- 2. updates parameters
         rnn:updateParameters(lr)
         rnn:zeroGradParameters()
         -- 3. reset the internal time-step counter
         --rnn:forget()
         inputs, outputs, targets = {}, {}, {}
      end
   end -- #sequence
   avgErrs[k] = kErrs:mean()
   if avgErrs[k] < minErr then
      minErr = avgErrs[k]
      minK = k
   end
end -- #epoches

--print(avgErrs)
print('min err: ' .. minErr .. ' on iteration ' .. minK)
