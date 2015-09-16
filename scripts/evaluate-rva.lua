require 'dp'
require 'rnn'
require 'optim'

-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a Recurrent Model for Visual Attention')
cmd:text('Options:')
cmd:option('--xpPath', '', 'path to a previously saved model')
cmd:option('--cuda', false, 'model was saved with cuda')
cmd:option('--evalTest', false, 'model was saved with cuda')
cmd:text()
local opt = cmd:parse(arg or {})

-- check that saved model exists
assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')

if opt.cuda then
   require 'cunn'
end

xp = torch.load(opt.xpPath)

model = xp:model().module

tester = xp:tester()

conf = tester:feedback()

cm = conf._cm

print("Last evaluation of test set :")
print(cm)

ds = dp.Mnist()

if opt.evalTest then
   conf:reset()
   tester:propagateEpoch(ds:testSet())

   print("Another evaluation of test set :")
   print(cm)

   conf:reset()
   tester:propagateEpoch(ds:testSet())

   print("And another (it's stochastic):")
   print(cm)
end

inputs = ds:get('test','inputs','bhwc')
targets = ds:get('test','targets','b')

input = inputs:narrow(1,1,10)
model:training() -- otherwise the rnn doesn't save intermediate time-step states
output = model:forward(input)

function drawBox(img, bbox, channel)
    channel = channel or 1

    local x1, y1 = torch.round(bbox[1]), torch.round(bbox[2])
    local x2, y2 = torch.round(bbox[1] + bbox[3]), torch.round(bbox[2] + bbox[4])

    x1, y1 = math.max(1, x1), math.max(1, y1)
    x2, y2 = math.min(img:size(3), x2), math.min(img:size(2), y2)

    local max = img:max()

    for i=x1,x2 do
        img[channel][y1][i] = max
        img[channel][y2][i] = max
    end
    for i=y1,y2 do
        img[channel][i][x1] = max
        img[channel][i][x2] = max
    end

    return img
end

ra = model:findModules('nn.RecurrentAttention')[1]
sg = model:findModules('nn.SpatialGlimpse')[1]

locations = ra.actions

input = nn.Convert('bhwc','bchw'):forward(input)
glimpses = {}
patches = {}

params = nil
for i=1,input:size(1) do
   local img = input[i]
   for j,location in ipairs(locations) do
      local glimpse = glimpses[j] or {}
      glimpses[j] = glimpse
      local patch = patches[j] or {}
      patches[j] = patch
      
      local xy = location[i]
      -- (-1,-1) top left corner, (1,1) bottom right corner of image
      local x, y = xy:select(1,1), xy:select(1,2)
      -- (0,0), (1,1)
      x, y = (x+1)/2, (y+1)/2
      x, y = x*input:size(3), y*input:size(4)
      
      local bbox = {x-(sg.size/2),y-(sg.size/2), sg.size, sg.size}
      glimpse[i] = drawBox(img:clone(), bbox, 1)
      local sg_
      if j == 1 then
         sg_ = ra.rnn.initialModule:get(1):get(1):get(2):get(1).module
      else
         sg_ = ra.rnn.sharedClones[j]:get(1):get(1):get(1):get(2):get(1)
      end
      patch[i] = image.scale(img:clone():float(), sg_.output[i]:float())
      
      collectgarbage()
   end
end

paths.mkdir('glimpse')
for j,glimpse in ipairs(glimpses) do
   local g = image.toDisplayTensor{input=glimpse,nrow=10,padding=3}
   local p = image.toDisplayTensor{input=patches[j],nrow=10,padding=3}
   image.save("glimpse/glimpse"..j..".png", g)
   image.save("glimpse/patch"..j..".png", p)
end


