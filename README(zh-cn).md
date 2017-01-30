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
 * [Recursor](#rnn.Recursor) : 用来封装一个模型来使它符合[AbstractRecurrent](#rnn.AbstractRecurrent)的接口;
 * [Recurrence](#rnn.Recurrence) : 封装一个模块当输入`{input(t), output(t-1)}`时输出`output(t)`;

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
参数`rho` 是反向传播序列（backpropagate through time (BPTT)）的最大值.
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

反向传播序列（backpropagation through time (BPTT)）.

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
这个方法设置进行反向传播序列时（backpropagation through time (BPTT)）
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
在训练模式, 网络记忆之前所有 `rho` (输入时间步数)个
状态. 这对反向传播序列（BPTT）来说是必要的. 

### evaluate() ###
在评估过程中, 因为已经没有之后进行反向传播序列（BPTT）的需要, 
只有之前的一步被记录. 这是很有效的内存使用, 
评估具有处理无限长序列的 
潜能.
 
<a name='rnn.Recurrent'></a>
## Recurrent ##
参考 :
 * A. [Sutsekever Thesis Sec. 2.5 and 2.8](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
 * B. [Mikolov Thesis Sec. 3.2 and 3.3](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
 * C. [RNN and Backpropagation Guide](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.3.9311&rep=rep1&type=pdf)

实现循环神经网络(RNN)[composite Module](https://github.com/torch/nn/blob/master/doc/containers.md#containers), 不包括输出层. 

`nn.Recurrent(start, input, feedback, [transfer, rho, merge])` 的构造函数获得5个参数:
 * `start` : 输出的数量(除了批量维度), 或者一个在第一次传播时将被插入到 `input` 模块和 `transfer` 模块之间的模块. 当 `start` 是一个尺寸时 (一个数或者 `torch.LongTensor`), 那么这个 *start* 模块将会被初始化为 `nn.Add(start)` (参见 引用. A).
 * `input` : 一个处理输入张量(或表)的模块. 输出的尺寸必须与 `start` 相同(或者它以 `start` 模块的情况输出), 而且与 `feedback` 模块输出的尺寸相同.
 * `feedback` : 一个返回之前输出的张量(或表)到 `transfer` 模块的模块.
 * `transfer` : 一个用来处理 `input` 和 `feedback` 模块输出对应元素的和的非线性模块, 或者在第一步输入的情况下,  *start* 模块的输出.
 * `rho` : 反向传播时处理序列的最大长度. 限制存储在内存中的之前步骤的数量. 由于空梯度效应的原因, 参考 A 和 B 推荐 `rho = 5` (或者更小). 默认是 99999.
 * `merge` : 一个在前向传播通过 `transfer` 模块前合并 `input` 和 `feedback` 模块输出的[table Module](https://github.com/torch/nn/blob/master/doc/table.md#table-layers).
 
RNN用来处理一个序列的输入. 
序列的每一步都应该用它自己的 `forward` (和 `backward`)传播, 
依次进行 `input` (和 `gradOutput`) . 
每一个 `forward` 调用保存了一个中间状态( `input` 和很多 `Module.outputs`)的记录 
并且把 `step` 属性增加1. 
在反向传播序列期间(BPTT)
`backward`方法必须被以与 `forward` 方法相反的顺序调用. 相反的顺序
使每一个 `forward` 返回一个对应的 `gradInput`是必要的. 

`step` 属性只有在调用 `forget` 方法时会被复位为1. 
在这种情况下, 模块准备好了去处理下一个序列(或者由此的一批).
注意序列越长=, 需要用来存储全部 `output` 和 `gradInput`状态
(每一步产生一个)的内存也越多. 

要批量地使用这个模块, 我们建议在一个批量内使用
一个不同的同尺寸的序列, 并且每 `rho` 步
调用 `updateParameters`, 并在序列的最后 `forget` . 

注意调用 `evaluate` 方法会关闭长期的记忆; 
RNN 只会记录前一个输出. 这允许 RNN 
不需要额外的分配内存就可以处理长序列.


一个简单简明的样例来展示如何使用这个模块, 请翻阅
[simple-recurrent-network.lua](examples/simple-recurrent-network.lua)
训练脚本.

<a name='rnn.Recurrent.Sequencer'></a>
### 用一个Sequencer来封装它 ###

注意任何 `AbstractRecurrent` 实例都可以被一个 [Sequencer](#rnn.Sequencer) 封装
这样一个完整的序列 (表) 可以被一个 `forward/backward` 调用处理.
实际上这是被建议的方法因为它允许 RNNs 的堆叠而且使
rnn 符合模块的接口, 也就是每一个 `forward` 调用可以被
可以被它自身最接近的 `backward` 调用跟随只要每一个对模块的 `input`
是一个完整的序列, 也就是一个张量的表其中每一个张量表示
一步输入.

```lua
seq = nn.Sequencer(module)
```

[simple-sequencer-network.lua](examples/simple-sequencer-network.lua)训练脚本
与上面提到的[simple-recurrent-network.lua](examples/simple-recurrent-network.lua)是等价的
脚本, 除了它把 `rnn` 用一个从表获取 `inputs` 和 `gradOutputs`
(那一批的序列)的 `Sequencer` 来封装之外.
这让 `Sequencer` 处理序列的循环.

你只需要考虑使用 `AbstractRecurrent` 模块而没有
一个 `Sequencer` 如果你打算使用它来进行实时预测. 
实际上, 你可以使用一个被 `Sequencer` 封装的`AbstractRecurrent` 实例
只要每一个实时步骤预测调用 `Sequencer:remember()` 并把`input` 表示为 `{input}`
来作为每一步的传入 .

其它可能使用到的封装器例如 [Repeater](#rnn.Repeater) 或者 [RecurrentAttention](#rnn.RecurrentAttention).
`Sequencer` 只是最通用的一个. 

<a name='rnn.LSTM'></a>
## LSTM ##
参考:
 * A. [Speech Recognition with Deep Recurrent Neural Networks](http://arxiv.org/pdf/1303.5778v1.pdf)
 * B. [Long-Short Term Memory](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf)
 * C. [LSTM: A Search Space Odyssey](http://arxiv.org/pdf/1503.04069v1.pdf)
 * D. [nngraph LSTM implementation on github](https://github.com/wojzaremba/lstm)

这是一个普通长短期记忆模块的实现. 
我们使用引用 A 的 LSTM 作为这个模型的蓝本因为它最简明.
而然在引用 C 中描述的也是普通的 LSTM. 

`nn.LSTM(inputSize, outputSize, [rho])` 的构造函数获得3个参数:
 * `inputSize` : 一个用来指定输入尺寸的数;
 * `outputSize` : 一个用来指定输出尺寸的数;
 * `rho` : 反向传播时处理序列的最大长度. 限制存储在内存中的之前步骤的数量. 默认是 9999.

![LSTM](doc/image/LSTM.png) 

实际的实现与下面的算法相应:
```lua
i[t] = σ(W[x->i]x[t] + W[h->i]h[t−1] + W[c->i]c[t−1] + b[1->i])      (1)
f[t] = σ(W[x->f]x[t] + W[h->f]h[t−1] + W[c->f]c[t−1] + b[1->f])      (2)
z[t] = tanh(W[x->c]x[t] + W[h->c]h[t−1] + b[1->c])                   (3)
c[t] = f[t]c[t−1] + i[t]z[t]                                         (4)
o[t] = σ(W[x->o]x[t] + W[h->o]h[t−1] + W[c->o]c[t] + b[1->o])        (5)
h[t] = o[t]tanh(c[t])                                                (6)
```
`W[s->q]` 是从 `s` 到 `q`的权重矩阵, `t` 索引时间步长,
`b[1->q]` 是引入到 `q` 中的偏差, `σ()` 是 `Sigmoid`, `x[t]` 是输入,
`i[t]` 是输入门 (公式. 1), `f[t]` 是遗忘门 (公式. 2), 
`z[t]` 是到单元 (我们称之为隐藏) 的输入 (公式. 3), 
`c[t]` 是单元 (公式. 4), `o[t]` 是输出门 (公式. 5), 
`h[t]` 是这个模块的输出 (公式. 6). 也要说明
从单元到们向量的权重矩阵 `W[c->s]` 是对角的, 这里 `s` 
可以是 `i`,`f`, 或者 `o`.

就像你可以看到的, 不像 [Recurrent](#rnn.Recurrent), 这个
实现不是足够的通用以至于可以使用任意通过构造
定义模块组件. 然而, LSTM模块可以轻松的被适用
通过继承并重写不同的默认函数 :
  * `buildGate` : 创造一个被用做输入,遗忘和输出的通用门;
  * `buildInputGate` : 创造一个输入门 (公式. 1). 现在调用 `buildGate`;
  * `buildForgetGate` : 创造一个遗忘门 (公式. 2). 现在调用 `buildGate`;
  * `buildHidden` : 创造隐藏 (公式. 3);
  * `buildCell` : 创造单元 (公式. 4);
  * `buildOutputGate` : 创造输出门 (公式. 5). 现在调用 `buildGate`;
  * `buildModel` : 创造一个内部使用的实际的 LSTM 模块 (公式. 6).
  
注意我们推荐用一个 `Sequencer` 来封装 `LSTM`
(具体细节参考 [这个](#rnn.Recurrent.Sequencer) ).
  
<a name='rnn.FastLSTM'></a>
## FastLSTM ##

一个快速的 [LSTM](#rnn.LSTM) 版本. 
从根本上说, 输入, 遗忘和输出门, 和隐藏状态被一举计算.

注意 `FastLSTM` 没有在单元和门之间使用窥视孔连接. 这个算法与 `LSTM` 相比变成了下面:
```lua
i[t] = σ(W[x->i]x[t] + W[h->i]h[t−1] + b[1->i])                      (1)
f[t] = σ(W[x->f]x[t] + W[h->f]h[t−1] + b[1->f])                      (2)
z[t] = tanh(W[x->c]x[t] + W[h->c]h[t−1] + b[1->c])                   (3)
c[t] = f[t]c[t−1] + i[t]z[t]                                         (4)
o[t] = σ(W[x->o]x[t] + W[h->o]h[t−1] + b[1->o])                      (5)
h[t] = o[t]tanh(c[t])                                                (6)
```
也就是说遗漏了这个和项 `W[c->i]c[t−1]` (公式. 1), `W[c->f]c[t−1]` (公式. 2), 和 `W[c->o]c[t]` (公式. 5).

### usenngraph ###
这是 `FastLSTM` 类的一个静态属性. 默认值是 `false`.
设置 `usenngraph = true` 将会强制 `FastLSTM` 所有新的实例化的实例
来使用 `nngraph` 的 `nn.gModule` 来创建内部的每一个时间步长
克隆的 `recurrentModule` .

<a name='rnn.GRU'></a>
## GRU ##

参考 :
 * A. [Learning Phrase Representations Using RNN Encoder-Decoder For Statistical Machine Translation.](http://arxiv.org/pdf/1406.1078.pdf)
 * B. [Implementing a GRU/LSTM RNN with Python and Theano](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)
 * C. [An Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
 * D. [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555)

这是 Gated Recurrent Units 模块的一个实现. 

`nn.GRU(inputSize, outputSize, [rho])` 的构造函数和 `nn.LSTM` 一样获取 3 个参数:
 * `inputSize` : 一个指定输入尺寸的数字;
 * `outputSize` : 一个指定输出尺寸的数字;
 * `rho` : 反向传播时处理序列的最大长度. 限制存储在内存中的之前步骤的数量. 默认是 9999.

![GRU](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/10/Screen-Shot-2015-10-23-at-10.36.51-AM.png) 

实际的实现与下面的算法对应:
```lua
z[t] = σ(W[x->z]x[t] + W[s->z]s[t−1] + b[1->z])            (1)
r[t] = σ(W[x->r]x[t] + W[s->r]s[t−1] + b[1->r])            (2)
h[t] = tanh(W[x->h]x[t] + W[hr->c](s[t−1]r[t]) + b[1->h])  (3)
s[t] = (1-z[t])h[t] + z[t]s[t-1]                           (4)
```
`W[s->q]` 是从 `s` 到 `q` 的权重矩阵, `t` 索引时间步长, `b[1->q]` 是引入到 `q` 中的偏差, `σ()` 是 `Sigmoid`, `x[t]` 是输入 `s[t]` 是这个模块 (公式. 4) 的输出. 注意不像 [LSTM](#rnn.LSTM), GRU 没有单元.

GRU 被在 `PennTreeBank` 数据集上使用 [recurrent-language-model.lua](examples/recurrent-language-model.lua) 脚本检测. 
它的性能比 `FastLSTM` 出色一点, 然而, 既然 LSTMs 比 GRUs 有更多的参数, 
比 `PennTreeBank` 大的数据集可能改变性能的结果. 
不要太草率的去判断两个中的哪一个更好 (参加 参考. C 和 D).

```
                Memory   examples/s
    FastLSTM      176M        16.5K 
    GRU            92M        15.8K
```

__Memory__ 用 `dp.Experiment` 保存文件的尺寸来衡量. __examples/s__ 用1轮训练的速度来衡量, 所以, 这可能存在硬盘 IO 的偏差.

![GRU-BENCHMARK](doc/image/gru-benchmark.png) 

<a name='rnn.Recursor'></a>
## Recursor ##

这一个模块封装一个 `module` 以在一个 `AbstractSequencer` 实例内使用.
它通过装饰模块使其符合 `AbstractRecurrent` 的接口来实现这个,
就像 `LSTM` 和 `Recurrent` 类, 这个类继承. 

```lua
rec = nn.Recursor(module[, rho])
```

对每一个成功的 `updateOutput` (也就是说 `forward`)调用, 这个
封装器将会产生一个被封装 `module` 的 `stepClone()`. 
所以对每一个时间步骤, 它克隆了 `module`. 克隆和
原始共享参数和下降 w.r.t. 参数. 然而, 对于 
已经符合模块 `AbstractRecurrent` 的接口, 
克隆和原始模型是同一个 (也就是说没有克隆).

例子 :

让我们假设我想堆两个 LSTMs. 我可以使用两个序列器 :

```lua
lstm = nn.Sequential()
   :add(nn.Sequencer(nn.LSTM(100,100)))
   :add(nn.Sequencer(nn.LSTM(100,100)))
```

使用一个 `Recursor`, 我用一个单一的 `Sequencer` 创造了相同的模型 :

```lua
lstm = nn.Sequencer(
   nn.Recursor(
      nn.Sequential()
         :add(nn.LSTM(100,100))
         :add(nn.LSTM(100,100))
      )
   )
```

实际上, `Sequencer` 会自动地包裹任何非-`AbstractRecurrent` 模块, 
所以我可以进一步简化为:

```lua
lstm = nn.Sequencer(
   nn.Sequential()
      :add(nn.LSTM(100,100))
      :add(nn.LSTM(100,100))
   )
```

我也可以在两个 `LSTM` 中间添加一个 `Linear`. 这种情况下,
一个 `Linear` 将会在每一个时间步骤被克隆 (并且它的参数被共享),
`LSTM`会做任何内部克隆:

```lua
lstm = nn.Sequencer(
   nn.Sequential()
      :add(nn.LSTM(100,100))
      :add(nn.Linear(100,100))
      :add(nn.LSTM(100,100))
   )
``` 

`AbstractRecurrent` 实例就像 `Recursor`, `Recurrent` 和 `LSTM` 一样
被期待内部的时间步骤管理. 非-`AbstractRecurrent` 实例
可以被一个 `Recursor` 包裹来获得相同的行为. 

每一个对一个 `AbstractRecurrent` 的 `forward` 调用 例如 `Recursor` 
将会增加 `self.step` 属性1, 对每一个成功的时间步骤(对一个最大时间步长为 `rho` ,
默认为 9999999)使用一个共享的参数来克隆.
在这种方式下, `backward` 可以被以 `forward` 调用相反的顺序
来进行反向传播序列 (BPTT). 这实际上是
[AbstractSequencer](#rnn.AbstractSequencer) 实例内部进行的.
`backward` 调用, 实际上被分别的调用为 `updateGradInput` 和
`accGradParameters`, 对分别地从 self.step 开始的 `self.udpateGradInputStep` 
和 `self.accGradParametersStep` 减少1. 
成功得 `backward` 调用将会递减这些计数器并利用他们来
适当地对内部的对应步数的共享参数的克隆进行反向传播.

无论如何, 对大多数情况来说, 你不需要直接地处理 `Recursor` 对象因为
`AbstractSequencer` 实例自动地在构造中将 非-`AbstractRecurrent` 
实例与一个 `Recursor` 封装.

对一个具体它的使用样例, 请参考一个使用它的例子
训练脚本 [simple-recurrent-network.lua](examples/simple-recurrent-network.lua) .

<a name='rnn.Recurrence'></a>
## Recurrence ##

一个非常通用的容器用来实现非常多种类的循环.

```lua
rnn = nn.Recurrence(recurrentModule, outputSize, nInputDim, [rho])
```

不像[Recurrent](#rnn.Recurrent), 这个模块不管理分离的
模块 像 `inputModule`, `startModule`, `mergeModule` 和相像的.
替代的, 它只管理一个 `recurrentModule`, 应该
输出一个张量或者表 : `output(t)` 
给定一个输入表 : `{input(t), output(t-1)}`.
使用一个 `Recursor` (或者说, 通过 `Sequencer`) 和 `Recurrence` 的混合, 可以实现
非常多种类型的循环神经网络, 包括 LSTMs 和 RNNs.

第一步, `Recurrence` 前向传播一个0张量 (或者它的表)
通过循环层 (像 LSTM, 不像 Recurrent).
所以这需要知道 `outputSize`, 可以是一个数或者
`torch.LongStorage`, 或者它的表. 批的维度应该被从
`outputSize` 中排除. 代替的, 批量的维度
(换句话说样例的数量) 将会从 `input` 中使用
`nInputDim` 参数推断. 举个例子, 假设我们的输入是一个
`4 x 3` 的张量 其中 `4` 是样本的数量, 那么 `nInputDim` 应该是 `1`.
另一个例子, 如果我们的输入时一个张量的表 [...] 的一个表
其中第一个张量 (深度优先) 与前一个例子相同,
那么我们的 `nInputDim` 也应该是 `1`.


作为一个例子, 使用 `Sequencer` 和 `Recurrence` 
来为语言模型创建一个简单的 RNN :

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

注意 : 我们可以使用新的 `Recursor` 和 `Recurrent` 模块
来很好地重新实现 `LSTM` 模块, 但是这意味着
打破与磁盘上现存模型的向后兼容性.

<a name='rnn.AbstractSequencer'></a>
## AbstractSequencer ##
这个抽象类实现一个轻量接口
被: `Sequencer`, `Repeater`, `RecurrentAttention`, `BiSequencer` 等子类共享.
  
<a name='rnn.Sequencer'></a>
## Sequencer ##

`nn.Sequencer(module)` 构造函数获取一个参数, `module`, 是从左
到右被应用的模块, 对输入序列的每一个元素.

```lua
seq = nn.Sequencer(module)
```

这个模块是一种 [decorator](http://en.wikipedia.org/wiki/Decorator_pattern) 
用来抽象地隔离错综复杂的 `AbstractRecurrent` 模块. 当一个 `AbstractRecurrent` 实例
需要一个序列来在每一个时刻产生一个输入, 来调用它自己的 `forward` 调用(和 `backward`调用),
`Sequencer` 前向传播一个 `input` 序列 (一个表) 产生一个 `output` 序列 (一个相同长度的表).
它也处理好 `forget`, `backwardOnline` 和其它 AbstractRecurrent-特定 方法的调用.

### 输入/输出格式

`Sequencer` 需要输入和输出的形状是 `seqlen x batchsize x featsize` :

 * `seqlen` 是需要被送入 `Sequencer` 的序列长度.
 * `batchsize` 是每一批的样例数. 每一个样例有它自己的独立序列.
 * `featsize` 是剩余非批量维度的尺寸. 所以对语言模型来说这可以是 `1` , 或者对卷积模型来说是 `c x h x w` , 等等.
 
![Hello Fuzzy](doc/image/hellofuzzy.png)

上面是一个字节级语言模型输入序列的样例.
它的`seqlen` 是 5 表明它包含一个序列长度为5的序列. 
开 `{` 和闭 `}` 表明序列的元素是一个 Lua 表.
`batchsize` 是 2 因为它有两个独立的序列 : `{ H, E, L, L, O }` and `{ F, U, Z, Z, Y, }`.
`featsize` 是 1 因为它每个字符只有一个特征维度而且字符的尺寸是 1.
所以这种情况下输入的表的序列长度是 `seqlen` 而序列的每一步都由一个 `batchsize x featsize` 的张量来表示.

![Sequence](doc/image/sequence.png)

上面是另一个序列 (输入或输出) 的样本. 
它有一个 `seqlen` 为 4 的序列长度. 
`batchsize` 又是 2 表明有两个序列.
`featsize` 是 3 因为序列长度里的每一个序列有 3 个变量.
所以每一个时间序列长 (元素组成的表) 再次被表示为一个
尺寸为 `batchsize x featsize` 的张量. 
注意在两个样例里 `featsize` 都只编码了一个维度, 
它可以编码更多. 


### 例子

举个例子, `rnn` : 一个 nn.AbstractRecurrent 的实例, 可以一次前向传播一个 `input` 序列:
```lua
input = {torch.randn(3,4), torch.randn(3,4), torch.randn(3,4)}
rnn:forward(input[1])
rnn:forward(input[2])
rnn:forward(input[3])
```

相等的, 我们可以使用一个 Sequencer 来一次性地前向传播整个 `input` 序列:

```lua
seq = nn.Sequencer(rnn)
seq:forward(input)
``` 

### 细节

`Sequencer` 也可以封装非循环的模块 (也就是 非-AbstractRecurrent 实例) 并应用它到每一个
输入来产生一个同样长度的输出表. 
这对处理一个变长的序列(表)时尤其有用.

内部地, `Sequencer` 期望封装的 `module` 是一个
`AbstractRecurrent` 实例. 当不是这种情况时, `module`
被自动地和一个[Recursor](#rnn.Recursor)模块封装, 这使它
符合 `AbstractRecurrent` 接口. 

注意 : 这是由于一个近期的更新 (2015年10月27日), 像以前一样
`AbstractRecurrent` 和 非-`AbstractRecurrent` 实例需要被
它们自己的 `Sequencer`封装. 最近的更新, 介绍了
`Recursor` 封装器, 允许一个单独的 `Sequencer` 来包裹任何类型的模块, 
`AbstractRecurrent`, 非-`AbstractRecurrent` 或者两种类型的混合结构.
虽然如此, 已经存在的代码应该不会被改变影响.

它使用的一个简洁的例子, 请参考 [simple-sequencer-network.lua](examples/simple-sequencer-network.lua)
训练脚本.

### remember([mode]) ###
当 `mode='neither'` (类的默认行为), Sequencer将会在每一个`forward`调用之前
另外调用[forget](#nn.AbstractRecurrent.forget), 这是这个类的默认行为. 
这种行为只对封装 AbstractRecurrent `modules`适用.
`mode` 参数可以接受下面的参数 :

 * 'eval' 只影响评测 (对RNNs推荐)
 * 'train' 只影响训练
 * 'neither' 既不影响训练也不影响评测 (这个类的默认行为)
 * 'both' 同时影响训练和评测 (对LSTMs建议)

### forget() ###
调用封装的 AbstractRecurrent 模块的 `forget` 方法.

<a name='rnn.BiSequencer'></a>
## BiSequencer ##
对一个输入序列以前向和反向应用封装的 `fwd` 和 `bwd` rnns.
这被用来实现 双向的 RNNs 和 LSTMs.

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
