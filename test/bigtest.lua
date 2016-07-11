local _ = require 'moses'
local rnnbigtest = {}
local precision = 1e-5
local mytester

function rnnbigtest.NCE_nan()
   local success, dl = pcall(function() return require 'dataload' end)
   if not success then
      return
   end
   if not pcall(function() require 'cunn' end) then
      return
   end
   
   local datapath = paths.concat(dl.DATA_PATH, 'BillionWords')
   local wordfreq = torch.load(paths.concat(datapath, 'word_freq.th7'))
   local unigram = wordfreq:float()--:add(0.0000001):log()
   print("U", unigram:min(), unigram:mean(), unigram:std(), unigram:max())
   
   local batchsize = 128
   local seqlen = 50
   local hiddensize = 200
   local vocabsize = unigram:size(1)
   local k = 400
   
   local tinyset = dl.MultiSequenceGBW(datapath, 'train_tiny.th7', batchsize, verbose)
   
   local lm = nn.Sequential()
   local lookup = nn.LookupTableMaskZero(vocabsize, hiddensize)
   lm:add(lookup)

   for i=1,2 do
      local rnn = nn.SeqLSTM(hiddensize, hiddensize)
      rnn.maskzero = true
      lm:add(rnn)
   end

   lm:add(nn.SplitTable(1))

   local ncemodule = nn.NCEModule(hiddensize, vocabsize, k, unigram, 1)

   lm = nn.Sequential()
      :add(nn.ParallelTable()
         :add(lm):add(nn.Identity()))
      :add(nn.ZipTable()) 

   lm:add(nn.Sequencer(nn.MaskZero(ncemodule, 1)))
   lm:remember()
   
   local crit = nn.MaskZeroCriterion(nn.NCECriterion(), 0)
   local targetmodule = nn.Sequential():add(nn.Convert()):add(nn.SplitTable(1))
   local criterion = nn.SequencerCriterion(crit)
   
   for k,param in ipairs(lm:parameters()) do
      param:uniform(-0.1, 0.1)
   end
   
   -- comment this out to see the difference
   ncemodule:reset()
   
   lm:training()
   
   lm:cuda()
   criterion:cuda()
   targetmodule:cuda()
   
   local sumErr = 0
   local _ = require 'moses'
   for k,inputs, targets in tinyset:subiter(seqlen, 512) do  
      local targets = targetmodule:forward(targets)
      local inputs = {inputs, targets}
      -- forward
      local outputs = lm:forward(inputs)
      for i,output in ipairs(outputs) do
         assert(not _.isNaN(output[1]:sum()), tostring(i))
         assert(not _.isNaN(output[2]:sum()), tostring(i))
         assert(not _.isNaN(output[3]:sum()), tostring(i))
         assert(not _.isNaN(output[4]:sum()), tostring(i))
      end
      local err = criterion:forward(outputs, targets)
      assert(not _.isNaN(err))
      sumErr = sumErr + err
      -- backward 
      local gradOutputs = criterion:backward(outputs, targets)
      
      for i,gradOutput in ipairs(gradOutputs) do
         assert(not _.isNaN(gradOutput[1]:sum()), tostring(i))
         assert(not _.isNaN(gradOutput[2]:sum()), tostring(i))
      end
      lm:zeroGradParameters()
      lm:backward(inputs, gradOutputs)   
      lm:updateParameters(0.7)
      local params, gradParams = lm:parameters()
      
      for i,param in ipairs(params) do
         assert(not _.isNaN(param:sum()), tostring(i))
         assert(not _.isNaN(gradParams[i]:sum()), tostring(i))
      end
      
      local counts = {}
      inputs[1]:float():apply(function(x)
         counts[x] = (counts[x] or 0) + 1
      end)
      
      print("Top freqs", unpack(_.last(_.sort(_.values(counts)), 5)))
      print("Batch : "..k..", err="..err)
      for name,module in pairs{LT=lookup, NCE=ncemodule} do
         print(name..".gradWeight : "..module.gradWeight:norm()..", .weight : "..module.weight:norm())
         if name == 'NCE' then
            print(name..".gradBias : "..module.gradBias:norm()..", .bias : "..module.bias:norm())
         end
      end
   end
   
end

function rnn.bigtest(tests)
   mytester = torch.Tester()
   mytester:add(rnnbigtest)
   math.randomseed(os.time())
   mytester:run(tests)
end
