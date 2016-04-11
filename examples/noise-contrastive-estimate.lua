require 'paths'
require 'rnn'
local dl = require 'dataload'

version = 2

--[[ command line arguments ]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Language Model on Google Billion Words dataset using RNN or LSTM or GRU')
cmd:text('Example:')
cmd:text('th recurrent-language-model.lua --cuda --device 2 --progress --cutoff 4 --seqlen 10')
cmd:text("th recurrent-language-model.lua --progress --cuda --lstm --seqlen 20 --hiddensize '{200,200}' --batchsize 20 --startlr 1 --cutoff 5 --maxepoch 13 --schedule '{[5]=0.5,[6]=0.25,[7]=0.125,[8]=0.0625,[9]=0.03125,[10]=0.015625,[11]=0.0078125,[12]=0.00390625}'")
cmd:text('Options:')
-- training
cmd:option('--startlr', 0.05, 'learning rate at t=0')
cmd:option('--minlr', 0.00001, 'minimum learning rate')
cmd:option('--saturate', 400, 'epoch at which linear decayed LR will reach minlr')
cmd:option('--schedule', '', 'learning rate schedule. e.g. {[5] = 0.004, [6] = 0.001}')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxnormout', -1, 'max l2-norm of each layer\'s output neuron weights')
cmd:option('--cutoff', -1, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--device', 1, 'sets the device (GPU) to use')
cmd:option('--maxepoch', 1000, 'maximum number of epochs to run')
cmd:option('--earlystop', 50, 'maximum number of epochs to wait to find a better local minima for early-stopping')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('--k', 25, 'how many noise samples to use for NCE')
-- rnn layer 
cmd:option('--lstm', false, 'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
cmd:option('--gru', false, 'use Gated Recurrent Units (nn.GRU instead of nn.Recurrent)')
cmd:option('--seqlen', 5, 'sequence length : back-propagate through time (BPTT) for this many time-steps')
cmd:option('--hiddensize', '{200}', 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
cmd:option('--dropout', 0, 'ancelossy dropout with this probability after each rnn layer. dropout <= 0 disables it.')
-- data
cmd:option('--batchsize', 32, 'number of examples per batch')
cmd:option('--trainsize', -1, 'number of train time-steps seen between each epoch')
cmd:option('--validsize', -1, 'number of valid time-steps used for early stopping and cross-validation') 
cmd:option('--savepath', paths.concat(dl.SAVE_PATH, 'rnnlm'), 'path to directory where experiment log (includes model) will be saved')
cmd:option('--id', '', 'id string of this experiment (used to name output file) (defaults to a unique id)')
cmd:option('--tiny', false, 'use train_tiny.th7 training file')

cmd:text()
local opt = cmd:parse(arg or {})
opt.hiddensize = loadstring(" return "..opt.hiddensize)()
opt.schedule = loadstring(" return "..opt.schedule)()
if not opt.silent then
   table.print(opt)
end
opt.id = opt.id == '' and ('gbw' .. ':' .. dl.uniqueid()) or opt.id

--[[ data set ]]--

local trainset, validset, testset = dl.loadGBW({opt.batchsize,1,1}, opt.tiny and 'train_tiny.th7' or nil)
if not opt.silent then 
   print("Vocabulary size : "..#trainset.ivocab) 
   print("Train set split into "..opt.batchsize.." sequences of length "..trainset:size())
end

--[[ language model ]]--

local lm = nn.Sequential()

-- input layer (i.e. word embedding space)
local lookup = nn.LookupTableMaskZero(#trainset.ivocab, opt.hiddensize[1])
lookup.maxnormout = -1 -- prevent weird maxnormout behaviour
lm:add(lookup) -- input is seqlen x batchsize
lm:add(nn.SplitTable(1)) -- tensor to table of tensors

if opt.dropout > 0 then
   lm:insert(nn.Dropout(opt.dropout), 1)
end

-- rnn layers
local stepmodule = nn.Sequential() -- ancelossied at each time-step
local inputsize = opt.hiddensize[1]
for i,hiddensize in ipairs(opt.hiddensize) do 
   local rnn
   
   if opt.gru then -- Gated Recurrent Units
      rnn = nn.GRU(inputsize, hiddensize)
   elseif opt.lstm then -- Long Short Term Memory units
      require 'nngraph'
      --nn.FastLSTM.usenngraph = true -- faster
      rnn = nn.FastLSTM(inputsize, hiddensize)
   else -- simple recurrent neural network
      local rm =  nn.Sequential() -- input is {x[t], h[t-1]}
         :add(nn.ParallelTable()
            :add(i==1 and nn.Identity() or nn.Linear(inputsize, hiddensize)) -- input layer
            :add(nn.Linear(hiddensize, hiddensize))) -- recurrent layer
         :add(nn.CAddTable()) -- merge
         :add(nn.Sigmoid()) -- transfer
      rnn = nn.Recurrence(rm, hiddensize, 1)
   end
   --rnn:maskZero(1)

   stepmodule:add(rnn)
   
   if opt.dropout > 0 then
      stepmodule:add(nn.Dropout(opt.dropout))
   end
   
   inputsize = hiddensize
end

-- output layer
local unigram = trainset.wordfreq:float()
local ncemodule = nn.NCEModule(inputsize, #trainset.ivocab, opt.k, unigram)
ncemodule:fastNoise()

-- NCE requires {input, target} as inputs
stepmodule = nn.Sequential()
   :add(nn.ParallelTable()
      :add(stepmodule):add(nn.Identity())) -- {input, target}
   :add(ncemodule)

lm = nn.Sequential()
   :add(nn.ParallelTable()
      :add(lm):add(nn.Identity()))
   :add(nn.ZipTable()) -- {input, target} -> {{x1,t1},{x2,t2},...}

-- encapsulate stepmodule into a Sequencer
lm:add(nn.Sequencer(stepmodule))

-- remember previous state between batches
lm:remember((opt.lstm or opt.gru) and 'both' or 'eval')

-- TEMP FIX
-- similar to apply, recursively goes over network and calls
-- a callback function which returns a new module replacing the old one
function nn.Module:replace(callback)
  local out = callback(self)
  if self.modules then
    for i, module in ipairs(self.modules) do
      self.modules[i] = module:replace(callback)
    end
   elseif self.recurrentModule then
    self.recurrentModule = callback(self.recurrentModule)
  end
  return out
end

lm:replace(function(module)
   if torch.type(module) ~= 'nn.NaN' then
      module = nn.NaN(module)
   end
   return module
end)

if not opt.silent then
   print"Language Model:"
   print(lm)
end

if opt.uniform > 0 then
   for k,param in ipairs(lm:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
   end
end

--[[ loss function ]]--

local crit = nn.MaskZeroCriterion(nn.NCECriterion(), 0)

-- target is also seqlen x batchsize.
local targetmodule = nn.SplitTable(1)
if opt.cuda then
   targetmodule = nn.Sequential()
      :add(nn.Convert())
      :add(targetmodule)
end
 
local criterion = nn.SequencerCriterion(crit)

--[[ CUDA ]]--

if opt.cuda then
   require 'cunn'
   cutorch.setDevice(opt.device)
   lm:cuda()
   criterion:cuda()
   targetmodule:cuda()
end

--[[ experiment log ]]--

-- is saved to file every time a new validation minima is found
local xplog = {}
xplog.opt = opt -- save all hyper-parameters and such
xplog.dataset = 'GoogleBillionWords'
xplog.vocab = trainset.vocab
-- will only serialize params
xplog.model = nn.Serial(lm)
xplog.model:mediumSerial()
xplog.criterion = criterion
xplog.targetmodule = targetmodule
-- keep a log of NLL for each epoch
xplog.trainnceloss = {}
xplog.valnceloss = {}
-- will be used for early-stopping
xplog.minvalnceloss = 99999999
xplog.epoch = 0
local ntrial = 0
paths.mkdir(opt.savepath)

local a_, b_ = torch.randn(3,4):cuda(), torch.CudaTensor():cuda()

function nn.NaN.updateOutput(self, input)
   print(string.format("updateOutput for module :\n%s", self:__tostring__()))
   a_.THNN.Sigmoid_updateOutput(
      a_:cdata(),
      b_:cdata()
   )
   print("stop")
   self.output = self.module:updateOutput(input)
   if self:recursiveIsNaN(self.output) then
      if self:recursiveIsNaN(input) then
         error(string.format("NaN found in input of module :\n%s", self:__tostring__()))
      elseif self:recursiveIsNaN(self:parameters()) then
         error(string.format("NaN found in parameters of module :\n%s", self:__tostring__()))
      end
      error(string.format("NaN found in output of module :\n%s", self:__tostring__()))
   end
   return self.output
end

function nn.Sigmoid:updateOutput(input)
   print("Sigmoid in", torch.type(input), torch.type(self.output))
   print(input:size(), input:sum(), input:isContiguous())
   self._input = self._input or input.new()
   if not input:isContiguous() then
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   print(input:size(), input:sum(), input:isContiguous(), self.output:isContiguous(), self.output:size())
   input.THNN.Sigmoid_updateOutput(
      input:cdata(),
      self.output:cdata()
   )
   print("Sigmoid out")
   return self.output
end

local epoch = 1
opt.lr = opt.startlr
opt.trainsize = opt.trainsize == -1 and trainset:size() or opt.trainsize
opt.validsize = opt.validsize == -1 and validset:size() or opt.validsize
while opt.maxepoch <= 0 or epoch <= opt.maxepoch do
   print("")
   print("Epoch #"..epoch.." :")

   -- 1. training
   
   local a = torch.Timer()
   lm:training()
   local sumErr = 0
   for i, inputs, targets in trainset:subiter(opt.seqlen, opt.trainsize) do
      print(1)
      local _ = require 'moses'
      assert(not _.isNaN(targets:sum()))
      targets = targetmodule:forward(targets)
      inputs = {inputs, targets}
      print(2)
      -- forward
      local outputs = lm:forward(inputs)
      print(3)
      local err = criterion:forward(outputs, targets)
      assert(not _.isNaN(err))
      sumErr = sumErr + err
      print(4)
      -- backward 
      local gradOutputs = criterion:backward(outputs, targets)
      print(5)
      assert(not _.isNaN(gradOutputs[1][1]:sum()))
      assert(not _.isNaN(gradOutputs[1][2]:sum()))
      lm:zeroGradParameters()
      lm:backward(inputs, gradOutputs)
      print(6)
      -- update
      if opt.cutoff > 0 then
         local norm = lm:gradParamClip(opt.cutoff) -- affects gradParams
         opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
      end
      print(7)
      lm:updateGradParameters(opt.momentum) -- affects gradParams
      print(8)
      lm:updateParameters(opt.lr) -- affects params
      print(9)
      lm:maxParamNorm(opt.maxnormout) -- affects params
      print(10)

      if opt.progress then
         xlua.progress(i, opt.trainsize)
      end

      if i % 2000 == 0 then
         collectgarbage()
      end

   end
   
   -- learning rate decay
   if opt.schedule then
      opt.lr = opt.schedule[epoch] or opt.lr
   else
      opt.lr = opt.lr + (opt.minlr - opt.startlr)/opt.saturate
   end
   opt.lr = math.max(opt.minlr, opt.lr)
   
   if not opt.silent then
      print("learning rate", opt.lr)
      if opt.meanNorm then
         print("mean gradParam norm", opt.meanNorm)
      end
   end

   if cutorch then cutorch.synchronize() end
   local speed = a:time().real/opt.trainsize
   print(string.format("Speed : %f sec/batch ", speed))

   local nceloss = sumErr/opt.trainsize
   print("Training error : "..nceloss)

   xplog.trainnceloss[epoch] = nceloss

   -- 2. cross-validation

   lm:evaluate()
   local sumErr = 0
   for i, inputs, targets in validset:subiter(opt.seqlen, opt.validsize) do
      targets = targetmodule:forward(targets)
      local outputs = lm:forward{inputs, targets}
      local err = criterion:forward(outputs, targets)
      sumErr = sumErr + err
      
      if opt.progress then
         xlua.progress(i, opt.validsize)
      end
   end

   local nceloss = sumErr/opt.validsize
   print("Validation error : "..nceloss)

   xplog.valnceloss[epoch] = nceloss
   ntrial = ntrial + 1

   -- early-stopping
   if nceloss < xplog.minvalnceloss then
      -- save best version of model
      xplog.minvalnceloss = nceloss
      xplog.epoch = epoch 
      local filename = paths.concat(opt.savepath, opt.id..'.t7')
      print("Found new minima. Saving to "..filename)
      torch.save(filename, xplog)
      ntrial = 0
   elseif ntrial >= opt.earlystop then
      print("No new minima found after "..ntrial.." epochs.")
      print("Stopping experiment.")
      print("Best model can be found in "..paths.concat(opt.savepath, opt.id..'.t7'))
      os.exit()
   end

   collectgarbage()
   epoch = epoch + 1
end

