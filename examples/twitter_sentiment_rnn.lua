--[[
   Script to train twitter sentiment classifier using the Twitter Sentiment
   data loader.
-]]

require 'paths'
require 'rnn'
require 'nngraph'
local dl = require 'dataload'

torch.setdefaulttensortype("torch.FloatTensor")

op = xlua.OptionParser('%prog [options]')

-- Data
op:option{'-d', '--datapath', action='store', dest='datapath',
          help='path to Twitter Sentiment Analysis data.',
          default='/media/eos/private/twitter'}
op:option{'--seqLen', action='store', dest='seqLen', help='Sequence Length',
          default=25}
op:option{'--minFreq', action='store', dest='minFreq',
          help='Drop words with occurance less than minFreq', default=10}
op:option{'--validRatio', action='store', dest='validRatio',
          help='Proportion of data to be used for validation.', default=0.2}

-- Model
op:option{'--lookupDim', action='store', dest='lookupDim',
          help='Feature dimensionality of lookuptable.', default=128}
op:option{'--lookupDropout', action='store', dest='lookupDropout',
          help='Dropout on lookup representation.', default=0}
op:option{'--hiddenSizes', action='store', dest='hiddenSizes',
          help='Hidden Layers', default='{256, 256}'}
op:option{'--dropouts', action='store', dest='dropouts',
          help='Dropouts', default='{0, 0}'}

-- Command line arguments
opt = op:parse()
op:summarize()

-- Data
datapath = opt.datapath
seqLen = tonumber(opt.seqLen)
minFreq = tonumber(opt.minFreq)
validRatio = tonumber(opt.validRatio)

trainFile = paths.concat(datapath, "trainSet.dl")
validFile = paths.concat(datapath, "validSet.dl")
testFile = paths.concat(datapath, "testSet.dl")

if paths.filep(trainFile) and paths.filep(validFile)
   and paths.filep(testFile) then
   trainSet = torch.load(trainFile)
   validSet = torch.load(validFile)
   testSet = torch.load(testFile)
else
   trainSet, validSet, testSet = dl.loadTwitterSentiment(datapath, minFreq,
                                                         seqLen, validRatio)
   torch.save(trainFile, trainSet)
   torch.save(validFile, validSet)
   torch.save(testFile, testSet)
end

-- Model
lookupDim = tonumber(opt.lookupDim)
lookupDropout = tonumber(opt.lookupDropout)
hiddenSizes = loadstring(" return " .. opt.hiddenSizes)()
dropouts = loadstring(" return " .. opt.dropouts)()
model = nn.Sequential()

-- LookupTable
local lookup = nn.LookupTableMaskZero(#trainSet.ivocab, lookupDim)
-- FIXME: Ask about maxoutnorm
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
end
