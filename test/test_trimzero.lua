require 'rnn'
require 'dp'
require 'sys'

torch.manualSeed(123)

batch_size = 200
sentence_length = 26
vocabulary_size = 1000
word_embedding_size = 200
rnn_size = 300

x = torch.ceil(torch.rand(batch_size,sentence_length)*vocabulary_size)
t = torch.ceil(torch.rand(batch_size)*10)

-- variable sentence lengths
for i=1,batch_size do
   idx = torch.floor(torch.rand(1)[1]*(sentence_length))
   if idx > 0 then x[i][{{1,idx}}]:fill(0) end
end

rnns = {'FastLSTM','GRU'}
methods = {'maskZero', 'trimZero'}

for ir,arch in pairs(rnns) do
   local rnn = nn[arch](word_embedding_size, rnn_size)
   local model = nn.Sequential()
               :add(nn.LookupTableMaskZero(vocabulary_size, word_embedding_size))
               :add(nn.SplitTable(2))
               :add(nn.Sequencer(rnn))
               :add(nn.SelectTable(sentence_length))
               :add(nn.Linear(rnn_size, 10))
   model:getParameters():uniform(-0.1, 0.1)
   collectgarbage()
   criterion = nn.CrossEntropyCriterion()
   local models = {}
   for j=1,#methods do
      table.insert(models, model:clone())
   end
   collectgarbage()
   for im,method in pairs(methods) do
      print('-- '..arch..' with '..method)
      model = models[im]
      rnn = model:get(3).module
      rnn[method](rnn, 1)
      sys.tic()
      for i=1,3 do
         model:zeroGradParameters()
         y = model:forward(x)
         loss = criterion:forward(y,t)
         print('loss:', loss)
         collectgarbage()
         dy = criterion:backward(y,t)
         model:backward(x, dy)
         w,dw = model:parameters()
         model:updateParameters(.5)
         collectgarbage()
      end
      elapse = sys.toc()
      print('elapse time:', elapse)   
   end
end
