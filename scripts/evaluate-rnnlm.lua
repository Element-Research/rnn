require 'nngraph'
require 'rnn'
local dl = require 'dataload'

-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a Recurrent Model for Visual Attention')
cmd:text('Options:')
cmd:option('--xplogpath', '', 'path to a previously saved xplog containing model')
cmd:option('--cuda', false, 'model was saved with cuda')
cmd:option('--device', 1, 'which GPU device to use')
cmd:option('--nsample', -1, 'sample this many words from the language model')
cmd:text()
local opt = cmd:parse(arg or {})

-- check that saved model exists
assert(paths.filep(opt.xplogpath), opt.xplogpath..' does not exist')

if opt.cuda then
   require 'cunn'
   cutorch.setDevice(opt.device)
end

local xplog = torch.load(opt.xplogpath)
local lm = xplog.model
local criterion = xplog.criterion
local targetmodule = xplog.targetmodule
print("Hyper-parameters (xplog.opt):")
print(xplog.opt)

local trainset, validset, testset = dl.loadPTB({50, 1, 1})

assert(trainset.vocab['the'] == xplog.vocab['the'])

print(lm)

lm:forget()
lm:evaluate()

if opt.nsample > 0 then
   local sampletext = {}
   local prevword = trainset.vocab['<eos>']
   assert(prevword)
   local inputs = torch.LongTensor(1,1) -- seqlen x batchsize
   if opt.cuda then inputs = inputs:cuda() end
   local buffer = torch.FloatTensor()
   for i=1,opt.nsample do
      inputs:fill(prevword)
      local output = lm:forward(inputs)[1][1]
      buffer:resize(output:size()):copy(output)
      buffer:exp()
      local sample = torch.multinomial(buffer, 1, true)
      local currentword = trainset.ivocab[sample[1]]
      table.insert(sampletext, currentword)
      prevword = sample[1]
   end
   print(table.concat(sampletext, ' '))
else
   local sumErr = 0
   
   for i, inputs, targets in testset:subiter(100) do
      local targets = targetmodule:forward(targets)
      local outputs = lm:forward(inputs)
      local err = criterion:forward(outputs, targets)
      sumErr = sumErr + err
   end

   print(sumErr, sumErr/testset:size())
   local ppl = torch.exp(sumErr/testset:size())
   print("Test PPL : "..ppl)
end

