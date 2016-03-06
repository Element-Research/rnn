# rnn: recurrent neural networks (递归神经网络)#

这是Torch神经网络库的一个扩展库，实现了RNN。
你可以用它来创建RNNs, LSTMs, GRUs, BRNNs, BLSTMs等等。
这个库里包含下列对象的文档：

可以通过按照顺序连续的调用`forward`来实现不同时间步骤输入的模块：
 * [AbstractRecurrent](#rnn.AbstractRecurrent) : 一个被Recurrent 和 LSTM继承的抽象类;
 * [Recurrent](#rnn.Recurrent) : 一个通用的循环神经网络容器;
 * [LSTM](#rnn.LSTM) : 一个普通的Long-Short Term Memory模块;
  * [FastLSTM](#rnn.FastLSTM) : 一个更快的[LSTM](#rnn.LSTM);
 * [GRU](#rnn.GRU) : Gated Recurrent Units模块;
 * [Recursor](#rnn.Recursor) : 用来装饰一个模型来使它符合[AbstractRecurrent](#rnn.AbstractRecurrent)的接口;
 * [Recurrence](#rnn.Recurrence) : 装饰一个模块当输入`{input(t), output(t-1)}`时输出`output(t)`;

通过装饰`AbstractRecurrent`实例一次`forward`整个序列的模块:
 * [AbstractSequencer](#rnn.AbstractSequencer) : 一个被Sequencer, Repeater, RecurrentAttention等继承的抽象类.;
 * [Sequencer](#rnn.Sequencer) : 对一个输入序列里的所有元素应用模块封装;
 * [BiSequencer](#rnn.BiSequencer) : 用来实现双向的RNNs和LSTMs;
 * [BiSequencerLM](#rnn.BiSequencerLM) : 用来实现语言模型的双向RNNs和LSTMs;
 * [Repeater](#rnn.Repeater) : 对一个AbstractRecurrent实例重复的应用同一个输入;
 * [RecurrentAttention](#rnn.RecurrentAttention) : 一个广受关注的模型[REINFORCE modules](https://github.com/nicholas-leonard/dpnn#nn.Reinforce);

其他的模块和损失函数:
 * [MaskZero](#rnn.MaskZero) : 来装饰一个模块当它`input`的张量中的行为0时，使相应的`output`和`gradOutput`为0.
 * [LookupTableMaskZero](#rnn.LookupTableMaskZero) : 扩展`nn.LookupTable`使它可以支持0索引和填充.0索引被作为以0组成的张量前向传递.
 * [MaskZeroCriterion](#rnn.MaskZeroCriterion) : 来装饰一个损失模块，当`input`张量的行为0时，使它对应的`gradInput`和`err`行为0.

用于处理序列输入和目标的损失函数:
 * [SequencerCriterion](#rnn.SequencerCriterion) : 对一个输入序列和目标顺序的应用相同的损失函数;
 * [RepeaterCriterion](#rnn.RepeaterCriterion) : 对一个输入序列重复以同一个目标应用同一个损失函数;


<a name='rnn.examples'></a>
## 例子 ##

下面的是用这个包实现的训练脚本样例:

  * [RNN/LSTM/GRU](examples/recurrent-language-model.lua) 使用宾夕法尼亚树库;
  * [Recurrent Model for Visual Attention](examples/recurrent-visual-attention.lua) 使用MNIST数据集;
  * [Encoder-Decoder LSTM](examples/encoder-decoder-coupling.lua) 向你展示如何连接用作编码器和解码器的`LSTMs`实现一个序列到序列的网络;
  * [Simple Recurrent Network](examples/simple-recurrent-network.lua) 展示一个创建和训练一个简单的循环神经网络的简单例子;
  * [Simple Sequencer Network](examples/simple-recurrent-network.lua) 是上面脚本使用Sequencer来装饰`rnn`的另一个版本;
  * [Sequence to One](examples/sequence-to-one.lua) 展示如何来进行多输入单输出序列的学习这种情感分析要用到的情况;
  * [Multivariate Time Series](examples/recurrent-time-series.lua) 展示如何来训练一个简单的RNN来进行multi-variate time-series 预测.

### 扩展资源

  * [dpnn](https://github.com/Element-Research/dpnn) : 这是__rnn__包的依赖. 它包含有用的nn扩展, 模块和损失函数.
  * [RNN/LSTM/BRNN/BLSTM training script ](https://github.com/nicholas-leonard/dp/blob/master/examples/recurrentlanguagemodel.lua)使用 宾夕法尼亚树库或Google Billion Words数据集;
  * 一个简明的Torch7概述(1 小时), 包含__rnn__包的一些细节 (在最后), 可以通过这里获得[NVIDIA GTC Webinar video](http://on-demand.gputechconf.com/gtc/2015/webinar/torch7-applied-deep-learning-for-vision-natural-language.mp4). 总之, 这个展示包含一个使用Torch7处理逻辑回归，多层感知器，卷积神经网络和循环神经网络不错的概览;
  * [ConvLSTM](https://github.com/viorik/ConvLSTM) 是一个训练 [Spatio-temporal video autoencoder with differentiable memory](http://arxiv.org/abs/1511.06309)的目录.
  * An [time series example](https://github.com/rracinskij/rnntest01/blob/master/rnntest01.lua) 用于univariate timeseries预测.
  
## 引用 ##

如果你在你的工作中使用到了__rnn__, 如果你引用下面的论文的话我们会很感激:

Léonard, Nicholas, Sagar Waghmare, Yang Wang, and Jin-Hwa Kim. [rnn: Recurrent Library for Torch.](http://arxiv.org/abs/1511.07889) arXiv preprint arXiv:1511.07889 (2015).

这个库的显著贡献者也会被作为作者加入这篇论文.
[significant contributor](https://github.com/Element-Research/rnn/graphs/contributors) 
是对这个库添加最少300行代码的人.

## 故障排除 ##

大部分问题可以通过更新各种依赖的包解决:
```bash
luarocks install torch
luarocks install nn
luarocks install dpnn
```

如果你使用CUDA:
```bash
luarocks install cutorch
luarocks install cunn
luarocks install cunnx
```

也不要忘了更新这个包 :
```bash
luarocks install rnn
```

如果还不能解决，在github上发布一个问题.

<a name='rnn.AbstractRecurrent'></a>
## AbstractRecurrent ##
一个被[Recurrent](#rnn.Recurrent), [LSTM](#rnn.LSTM) 和 [GRU](#rnn.GRU)继承的抽象类.
构造函数获取一个参数 :
```lua
rnn = nn.AbstractRecurrent([rho])
```
参数`rho` 是反向传播过程中的记录步数（backpropagate through time (BPTT)）的最大值.
子类可以设置这个值为一个大的数字像 99999 (默认值) 如果它们
想对整个序列进行反向传播而不管它有多长. 设置较小的 rho 值
在很长的序列被前向传播时是很有用的, 但是我们只希望对前`rho`步
进行反向传播, 这就意味着剩余的序列不需要被保存
(所以没有额外的开销). 

### [recurrentModule] getStepModule(step) ###
返回一个 `step` 时间步长的模块. 这个函数被内部的子类用来
获取内部 `recurrentModule` 的副本. 这些副本共享 
`parameters` 和 `gradParameters` 但是每个都有它们自己的 `output`, `gradInput` 
和其它任何的中间变量. 

### setOutputStep(step) ###
这是一个被保留的内部方法由 [Recursor](#rnn.Recursor)
进行反向传播时使用. 这个方法设置对象的 `output` 属性
指向时间步长 `step` 时的输出. 
这个方法被引进来解决一个非常烦人的bug.

### maskZero(nInputDim) ###
用 [MaskZero](#rnn.MaskZero) 来装饰内部的 `recurrentModule`. 
`recurrentModule` 的输出张量 (或由表生成的) `output`
的每一行(样例)会被置零当 `input` 对应的输入行
是一个值为0的张量时. 

参数 `nInputDim` 必须指定第一个输入张量 `input` 中
非批量维度（non-batch dims）的数量. 当输入一个表 `input` 时,
第一个张量是做深度优先搜索时遇到的第一个张量.

调用这个方法使以相同批量不同长度的0向量填充序列成为可能.
警告: 填充必须在任何真实输入序列里数据输入之前 (
在真实的数据输入之后填充不被支持而且会产生不可预测的结果而没有错误发生).

### [output] updateOutput(input) ###
对当前步的输入进行前向传播. 输出或者之前步的中间 
状态被循环地使用. 因为存储之前的输出和中间状态
所以这对调用则来说是透明的. 这个方法
同时把 `step` 的值增加1.

<a name='rnn.AbstractRecurrent.updateGradInput'></a>
### updateGradInput(input, gradOutput) ###
像 `backward`, 这个方法应该用 `forward` 
调用传递序列相反的顺序调用. 例如 :

```lua
rnn = nn.LSTM(10, 10) -- AbstractRecurrent instance
local outputs = {}
for i=1,nStep do -- forward propagate sequence
   outputs[i] = rnn:forward(inputs[i])
end

for i=nStep,1,-1 do -- backward propagate sequence in reverse order
   gradInputs[i] = rnn:backward(inputs[i], gradOutputs[i])
end

rnn:forget()
``` 

以相反的顺序执行反向传播步骤（backpropagation through time (BPTT)）.

### accGradParameters(input, gradOutput, scale) ###
像 `updateGradInput`, 但是对 w.r.t. 参数增加梯度值.

### recycle(offset) ###
这个方法与 `forget`密切相关. 当前的输入步数大于`rho`时
这个方法很有用, 从这时开始回收 
最早产生的 `recurrentModule` `sharedClones`, 
这样它们就可以存储下一步时被重用. `offset` 
被例如`nn.Recurrent`这样第一步使用不同模块
的模型使用. 默认的偏置是0.

<a name='rnn.AbstractRecurrent.forget'></a>
### forget(offset) ###
这个方法把所有状态重置为初始时的序列缓存, 
即 它忘掉当前的序列. 它也重置 `step` 属性的值为1.
强烈建议在每次参数更新之后调用 `forget`. 
否则, 之前的状态会被用来产生下一个, 这
往往会导致不稳定. 这是由之前的状态被
现在用来改变参数使用造成的结果. 实践中在输入
每一个新的序列前调用`forget`也很好.

<a name='rnn.AbstractRecurrent.maxBPTTstep'></a>
###  maxBPTTstep(rho) ###
这个方法设置进行反向传播时（backpropagation through time (BPTT)）
记录的最大步数. 也就是说你把 `rho = 3` 步,
前向传播输入4步, 然后进行反向传播,只有后3步会被
反向传播使用. 如果你懂AbstractRecurrent实例被
一个[Sequencer](#rnn.Sequencer)包裹, 这会被Sequencer魔术般的自动处理.
否则, 设置这个值为一个大值 (例如 9999999), 对大部分情况, 都很好, 如果不是全部的话.

<a name='rnn.AbstractRecurrent.backwardOnline'></a>
### backwardOnline() ###
这个方法于 Jan 6, 2016 被弃用. 
从那时起, 默认的, `AbstractRecurrent` 实例使用
backwardOnline行为. 
具体细节参见 [updateGradInput](#rnn.AbstractRecurrent.updateGradInput).

### training() ###
In training mode, the network remembers all previous `rho` (number of time-steps)
states. This is necessary for BPTT. 

### evaluate() ###
During evaluation, since their is no need to perform BPTT at a later time, 
only the previous step is remembered. This is very efficient memory-wise, 
such that evaluation can be performed using potentially infinite-length 
sequence.
 
<a name='rnn.Recurrent'></a>
## Recurrent ##
References :
 * A. [Sutsekever Thesis Sec. 2.5 and 2.8](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
 * B. [Mikolov Thesis Sec. 3.2 and 3.3](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
 * C. [RNN and Backpropagation Guide](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.3.9311&rep=rep1&type=pdf)

A [composite Module](https://github.com/torch/nn/blob/master/doc/containers.md#containers) for implementing Recurrent Neural Networks (RNN), excluding the output layer. 

The `nn.Recurrent(start, input, feedback, [transfer, rho, merge])` constructor takes 5 arguments:
 * `start` : the size of the output (excluding the batch dimension), or a Module that will be inserted between the `input` Module and `transfer` module during the first step of the propagation. When `start` is a size (a number or `torch.LongTensor`), then this *start* Module will be initialized as `nn.Add(start)` (see Ref. A).
 * `input` : a Module that processes input Tensors (or Tables). Output must be of same size as `start` (or its output in the case of a `start` Module), and same size as the output of the `feedback` Module.
 * `feedback` : a Module that feedbacks the previous output Tensor (or Tables) up to the `transfer` Module.
 * `transfer` : a non-linear Module used to process the element-wise sum of the `input` and `feedback` module outputs, or in the case of the first step, the output of the *start* Module.
 * `rho` : the maximum amount of backpropagation steps to take back in time. Limits the number of previous steps kept in memory. Due to the vanishing gradients effect, references A and B recommend `rho = 5` (or lower). Defaults to 99999.
 * `merge` : a [table Module](https://github.com/torch/nn/blob/master/doc/table.md#table-layers) that merges the outputs of the `input` and `feedback` Module before being forwarded through the `transfer` Module.
 
An RNN is used to process a sequence of inputs. 
Each step in the sequence should be propagated by its own `forward` (and `backward`), 
one `input` (and `gradOutput`) at a time. 
Each call to `forward` keeps a log of the intermediate states (the `input` and many `Module.outputs`) 
and increments the `step` attribute by 1. 
Method `backward` must be called in reverse order of the sequence of calls to `forward` in 
order to backpropgate through time (BPTT). This reverse order is necessary 
to return a `gradInput` for each call to `forward`. 

The `step` attribute is only reset to 1 when a call to the `forget` method is made. 
In which case, the Module is ready to process the next sequence (or batch thereof).
Note that the longer the sequence, the more memory that will be required to store all the 
`output` and `gradInput` states (one for each time step). 

To use this module with batches, we suggest using different 
sequences of the same size within a batch and calling `updateParameters` 
every `rho` steps and `forget` at the end of the sequence. 

Note that calling the `evaluate` method turns off long-term memory; 
the RNN will only remember the previous output. This allows the RNN 
to handle long sequences without allocating any additional memory.


For a simple concise example of how to make use of this module, please consult the 
[simple-recurrent-network.lua](examples/simple-recurrent-network.lua)
training script.

<a name='rnn.Recurrent.Sequencer'></a>
### Decorate it with a Sequencer ###

Note that any `AbstractRecurrent` instance can be decorated with a [Sequencer](#rnn.Sequencer) 
such that an entire sequence (a table) can be presented with a single `forward/backward` call.
This is actually the recommended approach as it allows RNNs to be stacked and makes the 
rnn conform to the Module interface, i.e. each call to `forward` can be 
followed by its own immediate call to `backward` as each `input` to the 
model is an entire sequence, i.e. a table of tensors where each tensor represents
a time-step.

```lua
seq = nn.Sequencer(module)
```

The [simple-sequencer-network.lua](examples/simple-sequencer-network.lua) training script
is equivalent to the above mentionned [simple-recurrent-network.lua](examples/simple-recurrent-network.lua)
script, except that it decorates the `rnn` with a `Sequencer` which takes 
a table of `inputs` and `gradOutputs` (the sequence for that batch).
This lets the `Sequencer` handle the looping over the sequence.

You should only think about using the `AbstractRecurrent` modules without 
a `Sequencer` if you intend to use it for real-time prediction. 
Actually, you can even use an `AbstractRecurrent` instance decorated by a `Sequencer`
for real time prediction by calling `Sequencer:remember()` and presenting each 
time-step `input` as `{input}`.

Other decorators can be used such as the [Repeater](#rnn.Repeater) or [RecurrentAttention](#rnn.RecurrentAttention).
The `Sequencer` is only the most common one. 

<a name='rnn.LSTM'></a>
## LSTM ##
References :
 * A. [Speech Recognition with Deep Recurrent Neural Networks](http://arxiv.org/pdf/1303.5778v1.pdf)
 * B. [Long-Short Term Memory](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf)
 * C. [LSTM: A Search Space Odyssey](http://arxiv.org/pdf/1503.04069v1.pdf)
 * D. [nngraph LSTM implementation on github](https://github.com/wojzaremba/lstm)

This is an implementation of a vanilla Long-Short Term Memory module. 
We used Ref. A's LSTM as a blueprint for this module as it was the most concise.
Yet it is also the vanilla LSTM described in Ref. C. 

The `nn.LSTM(inputSize, outputSize, [rho])` constructor takes 3 arguments:
 * `inputSize` : a number specifying the size of the input;
 * `outputSize` : a number specifying the size of the output;
 * `rho` : the maximum amount of backpropagation steps to take back in time. Limits the number of previous steps kept in memory. Defaults to 9999.

![LSTM](doc/image/LSTM.png) 

The actual implementation corresponds to the following algorithm:
```lua
i[t] = σ(W[x->i]x[t] + W[h->i]h[t−1] + W[c->i]c[t−1] + b[1->i])      (1)
f[t] = σ(W[x->f]x[t] + W[h->f]h[t−1] + W[c->f]c[t−1] + b[1->f])      (2)
z[t] = tanh(W[x->c]x[t] + W[h->c]h[t−1] + b[1->c])                   (3)
c[t] = f[t]c[t−1] + i[t]z[t]                                         (4)
o[t] = σ(W[x->o]x[t] + W[h->o]h[t−1] + W[c->o]c[t] + b[1->o])        (5)
h[t] = o[t]tanh(c[t])                                                (6)
```
where `W[s->q]` is the weight matrix from `s` to `q`, `t` indexes the time-step,
`b[1->q]` are the biases leading into `q`, `σ()` is `Sigmoid`, `x[t]` is the input,
`i[t]` is the input gate (eq. 1), `f[t]` is the forget gate (eq. 2), 
`z[t]` is the input to the cell (which we call the hidden) (eq. 3), 
`c[t]` is the cell (eq. 4), `o[t]` is the output gate (eq. 5), 
and `h[t]` is the output of this module (eq. 6). Also note that the 
weight matrices from cell to gate vectors are diagonal `W[c->s]`, where `s` 
is `i`,`f`, or `o`.

As you can see, unlike [Recurrent](#rnn.Recurrent), this 
implementation isn't generic enough that it can take arbitrary component Module
definitions at construction. However, the LSTM module can easily be adapted 
through inheritance by overriding the different factory methods :
  * `buildGate` : builds generic gate that is used to implement the input, forget and output gates;
  * `buildInputGate` : builds the input gate (eq. 1). Currently calls `buildGate`;
  * `buildForgetGate` : builds the forget gate (eq. 2). Currently calls `buildGate`;
  * `buildHidden` : builds the hidden (eq. 3);
  * `buildCell` : builds the cell (eq. 4);
  * `buildOutputGate` : builds the output gate (eq. 5). Currently calls `buildGate`;
  * `buildModel` : builds the actual LSTM model which is used internally (eq. 6).
  
Note that we recommend decorating the `LSTM` with a `Sequencer` 
(refer to [this](#rnn.Recurrent.Sequencer) for details).
  
<a name='rnn.FastLSTM'></a>
## FastLSTM ##

A faster version of the [LSTM](#rnn.LSTM). 
Basically, the input, forget and output gates, as well as the hidden state are computed at one fell swoop.

Note that `FastLSTM` does not use peephole connections between cell and gates. The algorithm from `LSTM` changes as follows:
```lua
i[t] = σ(W[x->i]x[t] + W[h->i]h[t−1] + b[1->i])                      (1)
f[t] = σ(W[x->f]x[t] + W[h->f]h[t−1] + b[1->f])                      (2)
z[t] = tanh(W[x->c]x[t] + W[h->c]h[t−1] + b[1->c])                   (3)
c[t] = f[t]c[t−1] + i[t]z[t]                                         (4)
o[t] = σ(W[x->o]x[t] + W[h->o]h[t−1] + b[1->o])                      (5)
h[t] = o[t]tanh(c[t])                                                (6)
```
i.e. omitting the summands `W[c->i]c[t−1]` (eq. 1), `W[c->f]c[t−1]` (eq. 2), and `W[c->o]c[t]` (eq. 5).

### usenngraph ###
This is a static attribute of the `FastLSTM` class. The default value is `false`.
Setting `usenngraph = true` will force all new instantiated instances of `FastLSTM` 
to use `nngraph`'s `nn.gModule` to build the internal `recurrentModule` which is 
cloned for each time-step.

<a name='rnn.GRU'></a>
## GRU ##

References :
 * A. [Learning Phrase Representations Using RNN Encoder-Decoder For Statistical Machine Translation.](http://arxiv.org/pdf/1406.1078.pdf)
 * B. [Implementing a GRU/LSTM RNN with Python and Theano](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)
 * C. [An Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
 * D. [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555)

This is an implementation of Gated Recurrent Units module. 

The `nn.GRU(inputSize, outputSize, [rho])` constructor takes 3 arguments likewise `nn.LSTM`:
 * `inputSize` : a number specifying the size of the input;
 * `outputSize` : a number specifying the size of the output;
 * `rho` : the maximum amount of backpropagation steps to take back in time. Limits the number of previous steps kept in memory. Defaults to 9999.

![GRU](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/10/Screen-Shot-2015-10-23-at-10.36.51-AM.png) 

The actual implementation corresponds to the following algorithm:
```lua
z[t] = σ(W[x->z]x[t] + W[s->z]s[t−1] + b[1->z])            (1)
r[t] = σ(W[x->r]x[t] + W[s->r]s[t−1] + b[1->r])            (2)
h[t] = tanh(W[x->h]x[t] + W[hr->c](s[t−1]r[t]) + b[1->h])  (3)
s[t] = (1-z[t])h[t] + z[t]s[t-1]                           (4)
```
where `W[s->q]` is the weight matrix from `s` to `q`, `t` indexes the time-step, `b[1->q]` are the biases leading into `q`, `σ()` is `Sigmoid`, `x[t]` is the input and `s[t]` is the output of the module (eq. 4). Note that unlike the [LSTM](#rnn.LSTM), the GRU has no cells.

The GRU was benchmark on `PennTreeBank` dataset using [recurrent-language-model.lua](examples/recurrent-language-model.lua) script. 
It slightly outperfomed `FastLSTM`, however, since LSTMs have more parameters than GRUs, 
the dataset larger than `PennTreeBank` might change the performance result. 
Don't be too hasty to judge on which one is the better of the two (see Ref. C and D).

```
                Memory   examples/s
    FastLSTM      176M        16.5K 
    GRU            92M        15.8K
```

__Memory__ is measured by the size of `dp.Experiment` save file. __examples/s__ is measured by the training speed at 1 epoch, so, it may have a disk IO bias.

![GRU-BENCHMARK](doc/image/gru-benchmark.png) 

<a name='rnn.Recursor'></a>
## Recursor ##

This module decorates a `module` to be used within an `AbstractSequencer` instance.
It does this by making the decorated module conform to the `AbstractRecurrent` interface,
which like the `LSTM` and `Recurrent` classes, this class inherits. 

```lua
rec = nn.Recursor(module[, rho])
```

For each successive call to `updateOutput` (i.e. `forward`), this 
decorator will create a `stepClone()` of the decorated `module`. 
So for each time-step, it clones the `module`. Both the clone and 
original share parameters and gradients w.r.t. parameters. However, for 
modules that already conform to the `AbstractRecurrent` interface, 
the clone and original module are one and the same (i.e. no clone).

Examples :

Let's assume I want to stack two LSTMs. I could use two sequencers :

```lua
lstm = nn.Sequential()
   :add(nn.Sequencer(nn.LSTM(100,100)))
   :add(nn.Sequencer(nn.LSTM(100,100)))
```

Using a `Recursor`, I make the same model with a single `Sequencer` :

```lua
lstm = nn.Sequencer(
   nn.Recursor(
      nn.Sequential()
         :add(nn.LSTM(100,100))
         :add(nn.LSTM(100,100))
      )
   )
```

Actually, the `Sequencer` will wrap any non-`AbstractRecurrent` module automatically, 
so I could simplify this further to :

```lua
lstm = nn.Sequencer(
   nn.Sequential()
      :add(nn.LSTM(100,100))
      :add(nn.LSTM(100,100))
   )
```

I can also add a `Linear` between the two `LSTM`s. In this case,
a `Linear` will be cloned (and have its parameters shared) for each time-step,
while the `LSTM`s will do whatever cloning internally :

```lua
lstm = nn.Sequencer(
   nn.Sequential()
      :add(nn.LSTM(100,100))
      :add(nn.Linear(100,100))
      :add(nn.LSTM(100,100))
   )
``` 

`AbstractRecurrent` instances like `Recursor`, `Recurrent` and `LSTM` are 
expcted to manage time-steps internally. Non-`AbstractRecurrent` instances
can be wrapped by a `Recursor` to have the same behavior. 

Every call to `forward` on an `AbstractRecurrent` instance like `Recursor` 
will increment the `self.step` attribute by 1, using a shared parameter clone
for each successive time-step (for a maximum of `rho` time-steps, which defaults to 9999999).
In this way, `backward` can be called in reverse order of the `forward` calls 
to perform backpropagation through time (BPTT). Which is exactly what 
[AbstractSequencer](#rnn.AbstractSequencer) instances do internally.
The `backward` call, which is actually divided into calls to `updateGradInput` and 
`accGradParameters`, decrements by 1 the `self.udpateGradInputStep` and `self.accGradParametersStep`
respectively, starting at `self.step`.
Successive calls to `backward` will decrement these counters and use them to 
backpropagate through the appropriate internall step-wise shared-parameter clones.

Anyway, in most cases, you will not have to deal with the `Recursor` object directly as
`AbstractSequencer` instances automatically decorate non-`AbstractRecurrent` instances
with a `Recursor` in their constructors.

For a concrete example of its use, please consult the [simple-recurrent-network.lua](examples/simple-recurrent-network.lua)
training script for an example of its use.

<a name='rnn.Recurrence'></a>
## Recurrence ##

A extremely general container for implementing pretty much any type of recurrence.

```lua
rnn = nn.Recurrence(recurrentModule, outputSize, nInputDim, [rho])
```

Unlike [Recurrent](#rnn.Recurrent), this module doesn't manage a separate 
modules like `inputModule`, `startModule`, `mergeModule` and the like.
Instead, it only manages a single `recurrentModule`, which should 
output a Tensor or table : `output(t)` 
given an input table : `{input(t), output(t-1)}`.
Using a mix of `Recursor` (say, via `Sequencer`) with `Recurrence`, one can implement 
pretty much any type of recurrent neural network, including LSTMs and RNNs.

For the first step, the `Recurrence` forwards a Tensor (or table thereof)
of zeros through the recurrent layer (like LSTM, unlike Recurrent).
So it needs to know the `outputSize`, which is either a number or 
`torch.LongStorage`, or table thereof. The batch dimension should be 
excluded from the `outputSize`. Instead, the size of the batch dimension 
(i.e. number of samples) will be extrapolated from the `input` using 
the `nInputDim` argument. For example, say that our input is a Tensor of size 
`4 x 3` where `4` is the number of samples, then `nInputDim` should be `1`.
As another example, if our input is a table of table [...] of tensors 
where the first tensor (depth first) is the same as in the previous example,
then our `nInputDim` is also `1`.


As an example, let's use `Sequencer` and `Recurrence` 
to build a Simple RNN for language modeling :

```lua
rho = 5
hiddenSize = 10
outputSize = 5 -- num classes
nIndex = 10000

-- recurrent module
rm = nn.Sequential()
   :add(nn.ParallelTable()
      :add(nn.LookupTable(nIndex, hiddenSize))
      :add(nn.Linear(hiddenSize, hiddenSize)))
   :add(nn.CAddTable())
   :add(nn.Sigmoid())

rnn = nn.Sequencer(
   nn.Sequential()
      :add(nn.Recurrence(rm, hiddenSize, 1))
      :add(nn.Linear(hiddenSize, outputSize))
      :add(nn.LogSoftMax())
)
```

Note : We could very well reimplement the `LSTM` module using the
newer `Recursor` and `Recurrent` modules, but that would mean 
breaking backwards compatibility for existing models saved on disk.

<a name='rnn.AbstractSequencer'></a>
## AbstractSequencer ##
This abastract class implements a light interface shared by 
subclasses like : `Sequencer`, `Repeater`, `RecurrentAttention`, `BiSequencer` and so on.
  
<a name='rnn.Sequencer'></a>
## Sequencer ##

The `nn.Sequencer(module)` constructor takes a single argument, `module`, which is the module 
to be applied from left to right, on each element of the input sequence.

```lua
seq = nn.Sequencer(module)
```

This Module is a kind of [decorator](http://en.wikipedia.org/wiki/Decorator_pattern) 
used to abstract away the intricacies of `AbstractRecurrent` modules. While an `AbstractRecurrent` instance 
requires that a sequence to be presented one input at a time, each with its own call to `forward` (and `backward`),
the `Sequencer` forwards an `input` sequence (a table) into an `output` sequence (a table of the same length).
It also takes care of calling `forget`, `backwardOnline` and other such AbstractRecurrent-specific methods.

### Input/Output Format

The `Sequencer` requires inputs and outputs to be of shape `seqlen x batchsize x featsize` :

 * `seqlen` is the number of time-steps that will be fed into the `Sequencer`.
 * `batchsize` is the number of examples in the batch. Each example is its own independent sequence.
 * `featsize` is the size of the remaining non-batch dimensions. So this could be `1` for language models, or `c x h x w` for convolutional models, etc.
 
![Hello Fuzzy](doc/image/hellofuzzy.png)

Above is an example input sequence for a character level language model.
It has `seqlen` is 5 which means that it contains sequences of 5 time-steps. 
The openning `{` and closing `}` illustrate that the time-steps are elements of a Lua table.
The `batchsize` is 2 as their are two independent sequences : `{ H, E, L, L, O }` and `{ F, U, Z, Z, Y, }`.
The `featsize` is 1 as their is only one feature dimension per character and each such character is of size 1.
So the input in this case is a table of `seqlen` time-steps where each time-step is represented by a `batchsize x featsize` Tensor.

![Sequence](doc/image/sequence.png)

Above is another example of a sequence (input or output). 
It has a `seqlen` of 4 time-steps. 
The `batchsize` is again 2 which means there are two sequences.
The `featsize` is 3 as each time-step of each sequence has 3 variables.
So each time-step (element of the table) is represented again as a tensor
of size `batchsize x featsize`. 
Note that while in both examples the `featsize` encodes one dimension, 
it could encode more. 


### Example

For example, `rnn` : an instance of nn.AbstractRecurrent, can forward an `input` sequence one forward at a time:
```lua
input = {torch.randn(3,4), torch.randn(3,4), torch.randn(3,4)}
rnn:forward(input[1])
rnn:forward(input[2])
rnn:forward(input[3])
```

Equivalently, we can use a Sequencer to forward the entire `input` sequence at once:

```lua
seq = nn.Sequencer(rnn)
seq:forward(input)
``` 

### Details

The `Sequencer` can also take non-recurrent Modules (i.e. non-AbstractRecurrent instances) and apply it to each 
input to produce an output table of the same length. 
This is especially useful for processing variable length sequences (tables).

Internally, the `Sequencer` expects the decorated `module` to be an 
`AbstractRecurrent` instance. When this is not the case, the `module` 
is automatically decorated with a [Recursor](#rnn.Recursor) module, which makes it 
conform to the `AbstractRecurrent` interface. 

Note : this is due a recent update (27 Oct 2015), as before this 
`AbstractRecurrent` and and non-`AbstractRecurrent` instances needed to 
be decorated by their own `Sequencer`. The recent update, which introduced the 
`Recursor` decorator, allows a single `Sequencer` to wrap any type of module, 
`AbstractRecurrent`, non-`AbstractRecurrent` or a composite structure of both types.
Nevertheless, existing code shouldn't be affected by the change.

For a concise example of its use, please consult the [simple-sequencer-network.lua](examples/simple-sequencer-network.lua)
training script.

### remember([mode]) ###
When `mode='both'` (the default), the Sequencer will not call [forget](#nn.AbstractRecurrent.forget) at the start of 
each call to `forward`, which is the default behavior of the class. 
This behavior is only applicable to decorated AbstractRecurrent `modules`.
Accepted values for argument `mode` are as follows :

 * 'eval' only affects evaluation (recommended for RNNs)
 * 'train' only affects training
 * 'neither' affects neither training nor evaluation (default behavior of the class)
 * 'both' affects both training and evaluation (recommended for LSTMs)

### forget() ###
Calls the decorated AbstractRecurrent module's `forget` method.

<a name='rnn.BiSequencer'></a>
## BiSequencer ##
Applies encapsulated `fwd` and `bwd` rnns to an input sequence in forward and reverse order.
It is used for implementing Bidirectional RNNs and LSTMs.

```lua
brnn = nn.BiSequencer(fwd, [bwd, merge])
```

The input to the module is a sequence (a table) of tensors
and the output is a sequence (a table) of tensors of the same length.
Applies a `fwd` rnn (an [AbstractRecurrent](#rnn.AbstractRecurrent) instance) to each element in the sequence in
forward order and applies the `bwd` rnn in reverse order (from last element to first element).
The `bwd` rnn defaults to:

```lua
bwd = fwd:clone()
bwd:reset()
```

For each step (in the original sequence), the outputs of both rnns are merged together using
the `merge` module (defaults to `nn.JoinTable(1,1)`). 
If `merge` is a number, it specifies the [JoinTable](https://github.com/torch/nn/blob/master/doc/table.md#nn.JoinTable)
constructor's `nInputDim` argument. Such that the `merge` module is then initialized as :

```lua
merge = nn.JoinTable(1,merge)
```

Internally, the `BiSequencer` is implemented by decorating a structure of modules that makes 
use of 3 Sequencers for the forward, backward and merge modules.

Similarly to a [Sequencer](#rnn.Sequencer), the sequences in a batch must have the same size.
But the sequence length of each batch can vary.

Note : make sure you call `brnn:forget()` after each call to `updateParameters()`. 
Alternatively, one could call `brnn.bwdSeq:forget()` so that only `bwd` rnn forgets.
This is the minimum requirement, as it would not make sense for the `bwd` rnn to remember future sequences.


<a name='rnn.BiSequencerLM'></a>
## BiSequencerLM ##

Applies encapsulated `fwd` and `bwd` rnns to an input sequence in forward and reverse order.
It is used for implementing Bidirectional RNNs and LSTMs for Language Models (LM).

```lua
brnn = nn.BiSequencerLM(fwd, [bwd, merge])
```

The input to the module is a sequence (a table) of tensors
and the output is a sequence (a table) of tensors of the same length.
Applies a `fwd` rnn (an [AbstractRecurrent](#rnn.AbstractRecurrent) instance to the 
first `N-1` elements in the sequence in forward order.
Applies the `bwd` rnn in reverse order to the last `N-1` elements (from second-to-last element to first element).
This is the main difference of this module with the [BiSequencer](#rnn.BiSequencer).
The latter cannot be used for language modeling because the `bwd` rnn would be trained to predict the input it had just be fed as input.

![BiDirectionalLM](doc/image/bidirectionallm.png)

The `bwd` rnn defaults to:

```lua
bwd = fwd:clone()
bwd:reset()
```

While the `fwd` rnn will output representations for the last `N-1` steps,
the `bwd` rnn will output representations for the first `N-1` steps.
The missing outputs for each rnn ( the first step for the `fwd`, the last step for the `bwd`)
will be filled with zero Tensors of the same size the commensure rnn's outputs.
This way they can be merged. If `nn.JoinTable` is used (the default), then the first 
and last output elements will be padded with zeros for the missing `fwd` and `bwd` rnn outputs, respectively.

For each step (in the original sequence), the outputs of both rnns are merged together using
the `merge` module (defaults to `nn.JoinTable(1,1)`). 
If `merge` is a number, it specifies the [JoinTable](https://github.com/torch/nn/blob/master/doc/table.md#nn.JoinTable)
constructor's `nInputDim` argument. Such that the `merge` module is then initialized as :

```lua
merge = nn.JoinTable(1,merge)
```

Similarly to a [Sequencer](#rnn.Sequencer), the sequences in a batch must have the same size.
But the sequence length of each batch can vary.

Note that LMs implemented with this module will not be classical LMs as they won't measure the 
probability of a word given the previous words. Instead, they measure the probabiliy of a word
given the surrounding words, i.e. context. While for mathematical reasons you may not be able to use this to measure the 
probability of a sequence of words (like a sentence), 
you can still measure the pseudo-likeliness of such a sequence (see [this](http://arxiv.org/pdf/1504.01575.pdf) for a discussion).

<a name='rnn.Repeater'></a>
## Repeater ##
This Module is a [decorator](http://en.wikipedia.org/wiki/Decorator_pattern) similar to [Sequencer](#rnn.Sequencer).
It differs in that the sequence length is fixed before hand and the input is repeatedly forwarded 
through the wrapped `module` to produce an output table of length `nStep`:
```lua
r = nn.Repeater(module, nStep)
```
Argument `module` should be an `AbstractRecurrent` instance.
This is useful for implementing models like [RCNNs](http://jmlr.org/proceedings/papers/v32/pinheiro14.pdf),
which are repeatedly presented with the same input.

<a name='rnn.RecurrentAttention'></a>
## RecurrentAttention ##
References :
  
  * A. [Recurrent Models of Visual Attention](http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf)
  * B. [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](http://incompleteideas.net/sutton/williams-92.pdf)
  
This module can be used to implement the Recurrent Attention Model (RAM) presented in Ref. A :
```lua
ram = nn.RecurrentAttention(rnn, action, nStep, hiddenSize)
```

`rnn` is an [AbstractRecurrent](#rnn.AbstractRecurrent) instance. 
Its input is `{x, z}` where `x` is the input to the `ram` and `z` is an 
action sampled from the `action` module. 
The output size of the `rnn` must be equal to `hiddenSize`.

`action` is a [Module](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module) 
that uses a [REINFORCE module](https://github.com/nicholas-leonard/dpnn#nn.Reinforce) (ref. B) like 
[ReinforceNormal](https://github.com/nicholas-leonard/dpnn#nn.ReinforceNormal), 
[ReinforceCategorical](https://github.com/nicholas-leonard/dpnn#nn.ReinforceCategorical), or 
[ReinforceBernoulli](https://github.com/nicholas-leonard/dpnn#nn.ReinforceBernoulli) 
to sample actions given the previous time-step's output of the `rnn`. 
During the first time-step, the `action` module is fed with a Tensor of zeros of size `input:size(1) x hiddenSize`.
It is important to understand that the sampled actions do not receive gradients 
backpropagated from the training criterion. 
Instead, a reward is broadcast from a Reward Criterion like [VRClassReward](https://github.com/nicholas-leonard/dpnn#nn.VRClassReward) Criterion to 
the `action`'s REINFORCE module, which will backprogate graidents computed from the `output` samples 
and the `reward`. 
Therefore, the `action` module's outputs are only used internally, within the RecurrentAttention module.

`nStep` is the number of actions to sample, i.e. the number of elements in the `output` table.

`hiddenSize` is the output size of the `rnn`. This variable is necessary 
to generate the zero Tensor to sample an action for the first step (see above).

A complete implementation of Ref. A is available [here](examples/recurrent-visual-attention.lua).

<a name='rnn.MaskZero'></a>
## MaskZero ##
This module zeroes the `output` rows of the decorated module 
for commensurate `input` rows which are tensors of zeros.

```lua
mz = nn.MaskZero(module, nInputDim)
```

The `output` Tensor (or table thereof) of the decorated `module`
will have each row (samples) zeroed when the commensurate row of the `input` 
is a tensor of zeros. 

The `nInputDim` argument must specify the number of non-batch dims 
in the first Tensor of the `input`. In the case of an `input` table,
the first Tensor is the first one encountered when doing a depth-first search.

This decorator makes it possible to pad sequences with different lengths in the same batch with zero vectors.

<a name='rnn.LookupTableMaskZero'></a>
## LookupTableMaskZero ##
This module extends `nn.LookupTable` to support zero indexes. Zero indexes are forwarded as zero tensors.

```lua
lt = nn.LookupTableMaskZero(nIndex, nOutput)
```

The `output` Tensor will have each row zeroed when the commensurate row of the `input` is a zero index. 

This lookup table makes it possible to pad sequences with different lengths in the same batch with zero vectors.

<a name='rnn.MaskZeroCriterion'></a>
## MaskZeroCriterion ##
This criterion zeroes the `err` and `gradInput` rows of the decorated criterion 
for commensurate `input` rows which are tensors of zeros.

```lua
mzc = nn.MaskZeroCriterion(criterion, nInputDim)
```

The `gradInput` Tensor (or table thereof) of the decorated `criterion`
will have each row (samples) zeroed when the commensurate row of the `input` 
is a tensor of zeros. The `err` will also disregard such zero rows.

The `nInputDim` argument must specify the number of non-batch dims 
in the first Tensor of the `input`. In the case of an `input` table,
the first Tensor is the first one encountered when doing a depth-first search.

This decorator makes it possible to pad sequences with different lengths in the same batch with zero vectors.

<a name='rnn.SequencerCriterion'></a>
## SequencerCriterion ##

This Criterion is a [decorator](http://en.wikipedia.org/wiki/Decorator_pattern):

```lua
c = nn.SequencerCriterion(criterion)
``` 

Both the `input` and `target` are expected to be a sequence (a table). 
For each step in the sequence, the corresponding elements of the input and target tables 
will be applied to the `criterion`.
The output of `forward` is the sum of all individual losses in the sequence.
This is useful when used in conjuction with a [Sequencer](#rnn.Sequencer).

<a name='rnn.RepeaterCriterion'></a>
## RepeaterCriterion ##

This Criterion is a [decorator](http://en.wikipedia.org/wiki/Decorator_pattern):

```lua
c = nn.RepeaterCriterion(criterion)
``` 

The `input` is expected to be a sequence (a table). A single `target` is 
repeatedly applied using the same `criterion` to each element in the `input` sequence.
The output of `forward` is the sum of all individual losses in the sequence.
This is useful for implementing models like [RCNNs](http://jmlr.org/proceedings/papers/v32/pinheiro14.pdf),
which are repeatedly presented with the same target.
