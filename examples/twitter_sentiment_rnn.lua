--[[
   Script to train twitter sentiment classifier using the Twitter Sentiment
   data loader.
-]]

require 'paths'
require 'optim'
require 'rnn'
require 'nngraph'
require 'cutorch'
require 'cunn'
local dl = require 'dataload'

torch.setdefaulttensortype("torch.FloatTensor")

--[[ Command line arguments --]]
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a LSTM based sentiments classifier on Twitter dataset.')
cmd:text('Options:')
-- Data
cmd:option('--datapath', '/data/Twitter/', 'Path to Twitter data.')
cmd:option('--seqLen', 25, 'Sequence Length. BPTT for this many time steps.')
cmd:option('--minFreq', 10, 'Min freq for a word to be considered in vocab.')
cmd:option('--validRatio', 0.2, 'Part of trainSet to be used as validSet.')
cmd:option('--lookupDim', 128, 'Lookup feature dimensionality.')
cmd:option('--lookupDropout', 0, 'Lookup feature dimensionality.')
cmd:option('--hiddenSizes', '{256, 256}', 'Hidden size for LSTM.')
cmd:option('--dropouts', '{0, 0}', 'Dropout on hidden representations.')
cmd:option('--useCuda', false, 'Use GPU for training.')
cmd:option('--deviceId', 1, 'Device Id.')
cmd:option('--batchSize', 128, 'number of examples per batch')
cmd:option('--epochs', 1000, 'maximum number of epochs to run')
cmd:option('--earlyStopThresh', 50, 'Early stopping threshold.')
cmd:option('--adam', false, 'Use Adaptive moment estimation optimizer.')
cmd:option('--learningRate', 0.001, 'Learning rate.')
cmd:option('--learningRateDecay', 1e-7, 'Learning rate decay.')
cmd:option('--momentum', 0, 'Momentum')

cmd:text()
local opt = cmd:parse(arg or {})

-- Data
datapath = opt.datapath
seqLen = opt.seqLen
minFreq = opt.minFreq
validRatio = opt.validRatio

classes = {'0', '2', '4'}

trainSet, validSet, testSet = dl.loadSentiment140(datapath, minFreq,
                                                  seqLen, validRatio)

-- Model
lookupDim = tonumber(opt.lookupDim)
lookupDropout = tonumber(opt.lookupDropout)
hiddenSizes = loadstring(" return " .. opt.hiddenSizes)()
dropouts = loadstring(" return " .. opt.dropouts)()

model = nn.Sequential()

-- Transpose, such that input is seqLen x batchSize
model:add(nn.Transpose({1,2}))

-- LookupTable
local lookup = nn.LookupTableMaskZero(#trainSet.ivocab, lookupDim)
model:add(lookup)
if lookupDropout ~= 0 then model:add(nn.Dropout(lookupDropout)) end

-- Recurrent layers
local inputSize = lookupDim
for i, hiddenSize in ipairs(hiddenSizes) do
   local rnn = nn.SeqLSTM(inputSize, hiddenSize)
   rnn.maskzero = true
   model:add(rnn)
   if dropouts[i] ~= 0 and dropouts[i] ~= nil then
      model:add(nn.Dropout(dropouts[i]))
   end
   inputSize = hiddenSize 
end
model:add(nn.Select(1, -1))

-- Output Layer
model:add(nn.Linear(hiddenSizes[#hiddenSizes], #classes))
model:add(nn.LogSoftMax())

-- Criterion 
criterion = nn.ClassNLLCriterion()

-- Training
useCuda = opt.useCuda
deviceId = opt.deviceId
batchSize = opt.batchSize
epochs = opt.epochs
earlyStopThresh = opt.earlyStopThresh
epochSize = trainSet:size()
adam = opt.adam
learningRate = opt.learningRate
learningRateDecay = opt.learningRateDecay
momentum = opt.momentum

if useCuda then
   print("Using GPU:"..deviceId)
   cutorch.setDevice(deviceId)
   print("GPU set")
   model:cuda()
   print("Model copied to CUDA")
   criterion:cuda()
   print("Criterion copied to CUDA")
else
   print("Not using GPU")
end
print(model)

-- Confusion Matrix
confusion = optim.ConfusionMatrix(classes)

-- Retrieve parameters and gradients
parameters, gradParameters = model:getParameters()

-- Optimizers: Using SGD/ADAM [Stocastic Gradient Descent]
optimState = {
               learningRate = learningRate,
               momentum = momentum,
               learningRateDecay = learningRateDecay
             }
if adam then
   print("Using Adaptive moment estimation.")
   optimMethod = optim.adam
else
   print("Using Stocastic gradient descent")
   optimMethod = optim.sgd
end
print(optimState)


-- Variables for intermediate data
trainInputs = useCuda and torch.CudaTensor() or torch.FloatTensor()
trainTargets = useCuda and torch.CudaTensor() or torch.FloatTensor()
local conTargets, conOutputs
best_valid_accu = 0
best_valid_model = nn.Sequential()
best_train_accu = 0
best_train_model = nn.Sequential()
trainLoss = 0
validLoss = 0
earlyStopCount = 0

for epoch=1, epochs do
   -- Single training epoch
   trainLoss = 0
   confusion:zero()
   model:training()
   for i, inputs, targets in trainSet:sampleiter(batchSize, epochSize) do
      xlua.progress(i, epochSize)
      trainInputs:resize(inputs:size()):copy(inputs)
      trainTargets:resize(targets:size()):copy(targets)

      local feval = function()
         gradParameters:zero()

         -- Forward
         local outputs = model:forward(trainInputs)
         local f = criterion:forward(outputs, trainTargets)
         trainLoss = trainLoss + f

         -- Backward
         local df_do = criterion:backward(outputs, trainTargets)
         model:backward(trainInputs, df_do)

         if useCuda then
            conOutputs = outputs:float()
            conTargets = trainTargets:float()
         else
            conOutputs = outputs
            conTargets = trainTargets
         end
         confusion:batchAdd(conOutputs, conTargets)
         return f, gradParameters
      end
      optimMethod(feval, parameters, optimState)
   end
   confusion:updateValids()
   if best_train_accu < confusion.totalValid then
      print("Best train accuracy: ".. best_train_accu ..
                  " current accu: ".. confusion.totalValid)
      best_train_accu = confusion.totalValid
      best_train_model = model:clone()
   end

   -- Validation accuracy
   validLoss = 0
   model:evaluate()
   confusion:zero()
   for i, inputs, targets in validSet:sampleiter(batchSize, validSet:size()) do
      trainInputs:resize(inputs:size()):copy(inputs)
      trainTargets:resize(targets:size()):copy(targets)
      local outputs = model:forward(trainInputs)
      local f = criterion:forward(outputs, trainTargets)
      validLoss = validLoss + f

      if useCuda then
         conOutputs = outputs:float()
         conTargets = trainTargets:float()
      else
         conOutputs = outputs
         conTargets = trainTargets
      end
      confusion:batchAdd(conOutputs, conTargets)
   end
   confusion:updateValids()
   if best_valid_accu < confusion.totalValid then
      print("Best valid accuracy: ".. best_valid_accu ..
                  " current accu: ".. confusion.totalValid)
      best_valid_accu = confusion.totalValid
      earlyStopCount = 0
      best_valid_model = model:clone()
   else
      earlyStopCount = earlyStopCount + 1
   end
   
   if earlyStopCount >= earlyStopThresh then
      print("Early stopping.")
      break
   end
end

-- Testing Accuracy
model = best_valid_model
model:evaluate()
confusion:zero()
for i, inputs, targets in testSet:sampleiter(batchSize, testSet:size()) do
   trainInputs:resize(inputs:size()):copy(inputs)
   trainTargets:resize(targets:size()):copy(targets)
   local outputs = model:forward(trainInputs)

   if useCuda then
      conOutputs = outputs:float()
      conTargets = trainTargets:float()
   else
      conOutputs = outputs
      conTargets = trainTargets
   end
   confusion:batchAdd(conOutputs, conTargets)
end
confusion:updateValids()
print("TestSet confusion")
print(confusion)
