require 'rnn'

-- hyper-parameters
cmd = torch.CmdLine()
cmd:text()
cmd:text('Simple Image Segmentation Training using Recurrent Convolution Neural Network')
cmd:text('Options:')
cmd:option('-lr', 0.1, 'learning rate at t=0')
cmd:option('-rcnnlayer', '{0,1,0}', 'each layer with a 1 will be a recurrent convlayer. Otherwise, a normal convolution.')
cmd:option('-channelsize', '{16,24,32}', 'Number of output channels for each convolution layer (excluding last layer which has fixed size of 1)')
cmd:option('-kernelsize', '{5,5,5}', 'kernel size of each convolution layer. Height = Width')
cmd:option('-kernelstride', '{1,1,1}', 'kernel stride of each convolution layer. Height = Width')
cmd:option('-poolsize', '{2,2,2}', 'size of the max pooling of each convolution layer. Height = Width')
cmd:option('-poolstride', '{2,2,2}', 'stride of the max pooling of each convolution layer. Height = Width')
cmd:option('-batchsize', 8, 'number of examples per batch')
cmd:option('-maxepoch', 100, 'maximum number of epochs to run')
cmd:option('-maxwait', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('-seqlen', 3, 'how many times to feed each image back into RCNN, i.e. back-propagate through time (BPTT) for rho time-steps')
cmd:option('-progress', false, 'print progress bar')
cmd:text()
local opt = cmd:parse(arg or {})
print(opt)

opt.rcnnlayer = loadstring(" return "..opt.rcnnlayer)()
opt.channelsize = loadstring(" return "..opt.channelsize)()
opt.kernelsize = loadstring(" return "..opt.kernelsize)()
opt.kernelstride = loadstring(" return "..opt.kernelstride)()
opt.poolsize = loadstring(" return "..opt.poolsize)()
opt.poolstride = loadstring(" return "..opt.poolstride)()

opt.imageSize = {1,28,28}

local stepmodule = nn.Sequential()

local inputsize = 1
for i=1,#opt.channelsize do
   -- convolution
   local conv = nn.SpatialConvolution(inputsize, opt.channelsize[i], opt.kernelsize[i], opt.kernelsize[i], opt.kernelstride[i], opt.kernelstride[i], math.floor(opt.kernelsize[i]/2))
   stepmodule:add(conv)
   
   -- pooling
   if opt.poolsize[i] then
      local pool = nn.SpatialMaxPooling(opt.poolsize[i], opt.poolsize[i], opt.poolstride[i], opt.poolstride[i]))
      stepmodule:add(pool)
   end
   
   if opt.rcnnlayer[i] == 1 then
      -- what is the output size of the stepmodule so far?
      local outputsize = unpack(stepmodule:forward(input):size():totable())
      table.remove(outputsize, 1)
      
      -- the recurrent module that will be applied to each time-step
      local rm =  nn.Sequential() -- input is {x[t], h[t-1]}
         :add(nn.ParallelTable()
            :add(nn.Identity()) -- input layer
            :add(nn.SpatialConvolution(opt.channelsize[i], opt.channelsize[i]))) -- recurrent layer
         :add(nn.CAddTable()) -- merge
         :add(nn[opt.transfer]()) -- transfer
      
      -- combine into rcnn layer
      local rcnn = nn.Recurrence(rm, outputsize, 3)
      stepmodule:add(rcnn)
   else
      stepmodule:add(nn[opt.transfer]())
   end
   
   inputsize = opt.channelsize[i]
end

-- 1x1 convolution with one output channel
stepmodule:add(nn.SpatialConvolution(inputsize, 1, 1, 1, 1, 1))
stepmodule:add(nn.Sigmoid())

-- encapsulate stepmodule into a Sequencer
local rcnn = nn.Sequential()
   :add(nn.Convert())
   :add(nn.Sequencer(stepmodule))
   :add(nn.SplitTable(1))

-- remember previous state between batches
rcnn:remember()

print(rcnn)

-- build criterion

-- target is also seqlen x batchsize.
local targetmodule = nn.SplitTable(1)
if opt.cuda then
   targetmodule = nn.Sequential()
      :add(nn.Convert())
      :add(targetmodule)
end

local criterion = nn.SequencerCriterion(nn.BCECriterion())

-- get data

-- each mnist digit will be morphed into a the next digit (0-1,1-2,3-4)
local dl = require 'dataload'
local train = dl.loadMNIST()
--TODO


-- training
for epoch=1,10 do
   
   local a = torch.Timer()
   rcnn:training()
   xplog.traincm:zero()
   for i, inputs, targets in train:subiter(opt.seqlen, opt.trainsize) do
      targets = targetmodule:forward(targets)
      
      -- forward
      local outputs = rcnn:forward(inputs)
      criterion:forward(outputs, targets)
      
      for step=1,#outputs do
         xplog.traincm:batchAddBinary(outputs[step]:view(-1), targets[step]:view(-1))
      end
      
      -- backward 
      local gradOutputs = criterion:backward(outputs, targets)
      rcnn:zeroGradParameters()
      rcnn:backward(inputs, gradOutputs)
      
      -- update
      rcnn:updateParameters(opt.lr) -- affects params

      if opt.progress then
         xlua.progress(math.min(i + opt.seqlen, opt.trainsize), opt.trainsize)
      end

      if i % 1000 == 0 then
         collectgarbage()
      end

   end
   
end
