--[[
   Script to train twitter sentiment classifier using the Twitter Sentiment
   data loader.
-]]

require 'paths'
require 'rnn'
require 'nngraph'
local dl = require 'dataload'

torch.setdefaulttensortype("torch.FloatTensor")

--[[ Command line arguments --]]
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a LSTM based sentiments classifier on Twitter dataset.')
cmd:text('Options:')
-- Data
cmd:option('--datapath', '/data/Twitter/', 'Path to Twitter data.')
cmd:option('seqLen', 25, 'Sequence Length. BPTT for this many time steps.')
cmd:option('minFreq', 10, 'Min frequency for a word to be considered in vocab.')
cmd:option('validRatio', 0.2, 'Part of trainSet to be used as validSet.')
cmd:option('lookupDim', 128, 'Lookup feature dimensionality.')
cmd:option('lookupDropout', 0, 'Lookup feature dimensionality.')
cmd:option('hiddenSizes', '{256, 256}', 'Hidden size for LSTM.')
cmd:option('dropouts', '{0, 0}', 'Dropout on hidden representations.')
cmd:option('--useCuda', false, 'Use GPU for training.')
cmd:option('--batchSize', 128, 'number of examples per batch')
cmd:option('--epochs', 1000, 'maximum number of epochs to run')
cmd:option('--earlyStopThresh', 50, 'Early stopping threshold.')

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
batchSize = opt.batchSize
epochs = opt.epochs
earlyStopThresh = opt.earlyStopThresh
