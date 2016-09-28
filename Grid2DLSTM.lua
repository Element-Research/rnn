local Grid2DLSTM, parent = torch.class("nn.Grid2DLSTM", 'nn.AbstractRecurrent')

function lstm(h_t, h_d, prev_c, rnn_size)
  local all_input_sums = nn.CAddTable()({h_t, h_d})
  local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
  local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
  -- decode the gates
  local in_gate = nn.Sigmoid()(n1)
  local forget_gate = nn.Sigmoid()(n2)
  local out_gate = nn.Sigmoid()(n3)
  -- decode the write inputs
  local in_transform = nn.Tanh()(n4)
  -- perform the LSTM update
  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
    })
  -- gated cells form the output
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end


function Grid2DLSTM:__init(inputSize, outputSize, nb_layers, dropout, tie_weights, rho, cell2gate)
   parent.__init(self, rho or 9999)
   self.inputSize = inputSize
   self.outputSize = outputSize or inputSize
   self.should_tie_weights = tie_weights or true
   self.dropout = dropout or 0
   self.nb_layers = nb_layers
   -- build the model
   self.cell2gate = (cell2gate == nil) and true or cell2gate
   self.recurrentModule = self:buildModel()

   -- make it work with nn.Container
   self.modules[1] = self.recurrentModule
   self.sharedClones[1] = self.recurrentModule

   -- for output(0), cell(0) and gradCell(T)
   self.zeroTensor = torch.Tensor()

   self.cells = {}
   self.gradCells = {}

   -- initialization
  --  local net_params = self.recurrentModule:parameters()
   --
  --  for _, p in pairs(net_params) do
  --    p:uniform(-0.08, 0.08)
  --  end
   --
    -- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
    for layer_idx = 1, self.nb_layers do
        for _,node in ipairs(self.recurrentModule.forwardnodes) do
            if node.data.annotations.name == "i2h_" .. layer_idx then
                print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                node.data.module.bias[{{self.outputSize+1, 2*self.outputSize}}]:fill(1.0)
            end
        end
    end

end

function Grid2DLSTM:buildModel()
  require 'nngraph'
  assert(nngraph, "Missing nngraph package")

    -- There will be 2*n+1 inputs
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- input c for depth dimension
    table.insert(inputs, nn.Identity()()) -- input h for depth dimension
    for L = 1,self.nb_layers do
      table.insert(inputs, nn.Identity()()) -- prev_c[L] for time dimension
      table.insert(inputs, nn.Identity()()) -- prev_h[L] for time dimension
    end

    local shared_weights
    if self.should_tie_weights == true then shared_weights = {nn.Linear(self.outputSize, 4 * self.outputSize), nn.Linear(self.outputSize, 4 * self.outputSize)} end

    local outputs_t = {} -- Outputs being handed to the next time step along the time dimension
    local outputs_d = {} -- Outputs being handed from one layer to the next along the depth dimension

    for L = 1,self.nb_layers do
      -- Take hidden and memory cell from previous time steps
      local prev_c_t = inputs[L*2+1]
      local prev_h_t = inputs[L*2+2]

      if L == 1 then
        -- We're in the first layer
        prev_c_d = inputs[1] -- input_c_d: the starting depth dimension memory cell, just a zero vec.
        prev_h_d = nn.LookupTable(self.inputSize, self.outputSize)(inputs[2]) -- input_h_d: the starting depth dimension hidden state. We map a char into hidden space via a lookup table
      else
        -- We're in the higher layers 2...N
        -- Take hidden and memory cell from layers below
        prev_c_d = outputs_d[((L-1)*2)-1]
        prev_h_d = outputs_d[((L-1)*2)]
        if self.dropout > 0 then prev_h_d = nn.Dropout(self.dropout)(prev_h_d):annotate{name='drop_' .. L} end -- apply dropout, if any
      end

      -- Evaluate the input sums at once for efficiency
      local t2h_t = nn.Linear(self.outputSize, 4 * self.outputSize)(prev_h_t):annotate{name='i2h_'..L}
      local d2h_t = nn.Linear(self.outputSize, 4 * self.outputSize)(prev_h_d):annotate{name='h2h_'..L}

      -- Get transformed memory and hidden states pointing in the time direction first
      local next_c_t, next_h_t = lstm(t2h_t, d2h_t, prev_c_t, self.outputSize)

      -- Pass memory cell and hidden state to next timestep
      table.insert(outputs_t, next_c_t)
      table.insert(outputs_t, next_h_t)

      -- Evaluate the input sums at once for efficiency
      local t2h_d = nn.Linear(self.outputSize, 4 * self.outputSize)(next_h_t):annotate{name='i2h_'..L}
      local d2h_d = nn.Linear(self.outputSize, 4 * self.outputSize)(prev_h_d):annotate{name='h2h_'..L}

      -- See section 3.5, "Weight Sharing" of http://arxiv.org/pdf/1507.01526.pdf
      -- The weights along the temporal dimension are already tied (cloned many times in train.lua)
      -- Here we can tie the weights along the depth dimension. Having invariance in computation
      -- along the depth appears to be critical to solving the 15 digit addition problem w/ high accy.
      -- See fig 4. to compare tied vs untied grid lstms on this task.
      if self.should_tie_weights == true then
        print("tying weights along the depth dimension")
        t2h_d.data.module:share(shared_weights[1], 'weight', 'bias', 'gradWeight', 'gradBias')
        d2h_d.data.module:share(shared_weights[2], 'weight', 'bias', 'gradWeight', 'gradBias')
      end

      -- Create the lstm gated update pointing in the depth direction.
      -- We 'prioritize' the depth dimension by using the updated temporal hidden state as input
      -- instead of the previous temporal hidden state. This implements Section 3.2, "Priority Dimensions"
      local next_c_d, next_h_d = lstm(t2h_d, d2h_d, prev_c_d, self.outputSize)

      -- Pass the depth dimension memory cell and hidden state to layer above
      table.insert(outputs_d, next_c_d)
      table.insert(outputs_d, next_h_d)
    end

    -- set up the decoder
    local top_h = outputs_d[#outputs_d]
    table.insert(outputs_t, top_h)

    -- outputs_h contains
    -- nb_layers x (next_c_t, next_h_t)
    -- next-h

    return nn.gModule(inputs, outputs_t)

end

function Grid2DLSTM:updateOutput(input)
  -- if self.step == 1 then
  --   -- the initial state of the cell/hidden states
  --   self.cells = {[0] = {}}
  --
  --   for L=1,self.nb_layers do
  --     local h_init = torch.zeros(input:size(1), self.outputSize):cuda()
  --     table.insert(self.cells[0], h_init:clone())
  --     table.insert(self.cells[0], h_init:clone()) -- extra initial state for prev_c
  --   end
  -- end
  local input_mem_cell = torch.zeros(input:size(1),  self.outputSize):float():cuda()
  -- print(self.cells[self.step-1])
  local rnn_inputs = {input_mem_cell, input, unpack(self.cells[self.step-1])}
  local lst
  if self.train ~= false then
     self:recycle()
     local recurrentModule = self:getStepModule(self.step)
     -- the actual forward propagation
     lst = recurrentModule:updateOutput(rnn_inputs)
  else
     lst = self.recurrentModule:updateOutput(rnn_inputs)
  end

  self.cells[self.step] = {}
  for i=1,#(self.cells[0]) do table.insert(self.cells[self.step], lst[i]) end

  self.outputs[self.step] = lst[#lst]
  self.output = lst[#lst]

  self.step = self.step + 1
  self.gradPrevOutput = nil
  self.updateGradInputStep = nil
  self.accGradParametersStep = nil
  -- note that we don't return the cell, just the output
  return self.output

end

function Grid2DLSTM:_updateGradInput(input, gradOutput)
  assert(self.step > 1, "expecting at least one updateOutput")
  local step = self.updateGradInputStep - 1
  assert(step >= 1)

  -- set the output/gradOutput states of current Module
  local recurrentModule = self:getStepModule(step)

  -- backward propagate through this step
  if (step == self.step-1) then
    self.gradCells = {[step] = {}}
    for L=1,self.nb_layers do
      local h_init = torch.zeros(input:size(1), self.outputSize):cuda()
      table.insert(self.gradCells[step], h_init:clone())
      table.insert(self.gradCells[step], h_init:clone()) -- extra initial state for prev_c
    end
    local input_mem_cell = torch.zeros(input:size(1),  self.outputSize):float():cuda()
    self.rnn_inputs = {input_mem_cell, input, unpack(self.cells[step-1])}
  end

  table.insert(self.gradCells[step], gradOutput)



  local dlst = recurrentModule:updateGradInput(self.rnn_inputs, self.gradCells[step])
  self.gradCells[step-1] = {}
  local gradInput = {}
  for k,v in pairs(dlst) do
      if k > 2 then -- k <= skip_index is gradient on inputs, which we dont need
          -- note we do k-1 because first item is dembeddings, and then follow the
          -- derivatives of the state, starting at index 2. I know...
          self.gradCells[step-1][k-2] = v
      else
        table.insert(gradInput, v)
      end
  end
  return gradInput
end

function Grid2DLSTM:_accGradParameters(input, gradOutput, scale)
  local step = self.accGradParametersStep - 1
  assert(step >= 1)

  -- set the output/gradOutput states of current Module
  local recurrentModule = self:getStepModule(step)

  -- backward propagate through this step
  local input_mem_cell = torch.zeros(input:size(1),  self.outputSize):float():cuda()
  local rnn_inputs = {input_mem_cell, input, unpack(self.cells[step-1])}

  recurrentModule:accGradParameters( rnn_inputs, self.gradCells[step], scale)

end
