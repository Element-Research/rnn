mlp = nn.Sequential()
mlp:add(nn.Convert('bchw', 'bf')) -- collapse 3D to 1D
mlp:add(nn.Linear(1*28*28, 200))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(200, 200))
mlp:add(nn.Tanh()) 
mlp:add(nn.Linear(200, 10))
mlp:add(nn.LogSoftMax()) -- for classification problems