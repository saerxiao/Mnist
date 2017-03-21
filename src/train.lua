require 'torch'
require 'optim'
require 'nnModel'
require 'gnuplot'
require 'lfs'
require 'pl'

local mnist = require 'Loader'

cmd = torch.CmdLine()
cmd:option('-learning_rate',1e-3,'learning rate')
cmd:option('-epochs',10,'number of iterations')
cmd:option('-nCheckpoints',20,'number of checkpoints')
cmd:option('-train_max_load',1000,'number of train and validate data to load')
cmd:option('-batch_size',100,'batch size')
cmd:option('-checkpoint_dir','checkpoint/','dir to save checkpoints')
cmd:option('-model','convnet','neural net model')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-seed',123,'torch manual random number generator seed')
opt = cmd:parse(arg)

-- load lib for gpu
local ok, cunn = pcall(require, 'cunn')
local ok2, cutorch = pcall(require, 'cutorch')
if not ok then print('package cunn not found!') end
if not ok2 then print('package cutorch not found!') end
if ok and ok2 then
  print('using CUDA on GPU ' .. opt.gpuid .. '...')
  cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
  cutorch.manualSeed(opt.seed)
else
  print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
  print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
  print('Falling back on CPU mode')
  opt.gpuid = -1 -- overwrite user setting
end

-- load dataset
--local loader = mnist.create{batch_size = 100}
local loader = mnist.create{train_max_load = opt.train_max_load, batch_size = opt.batch_size}

-- create the model and trained parameters
local net, criterion
if opt.model == 'mlp' then
  net, criterion = mlp(32*32, 10)
elseif opt.model == 'convnet' then
  net, criterion = convnet(10)
else
  print('Unknown model type')
  cmd:text()
  error()
end

-- ship the model to the GPU if desired
if opt.gpuid == 0 then
--  for _, m in pairs(net.modules) do
--    m = m:cuda()
--  end
--  net.gradInput = net.gradInput:cuda()
--  net.output = net.output:cuda()
  net = net:cuda()
  criterion = criterion:cuda()
end

local params, grads = net:getParameters()
local tmpparams, tmpgrads = net:parameters()

-- initialize weights
params:uniform(-0.1, 0.1)
--prams = params:normal()*math.sqrt(2/trainset.size)

-- ship data to GPU for compute
local function ship2gpu(x, y)
  if opt.gpuid == 0 then
    x = x:cuda()
    y = y:cuda()
  end
  return x,y
end

local trainIter = loader:iterator("train")

--return loss, grad
local feval = function(w)
  if w ~= params then
    params:copy(w)
  end
  grads:zero()

--  local data, labels = loader:next_batch("train")
  local data, labels = trainIter.next_batch()
  data, labels = ship2gpu(data, labels)
--  print(#data)
--  print(labels:size())

  --forward
  local outputs = net:forward(data)
--  print(outputs)
  local loss = criterion:forward(outputs, labels)
--  print(loss)
  
  --backward
  local dloss = criterion:backward(outputs, labels)
  net:backward(data, dloss)
--  print(#dloss)
  
  return loss, grads
end

local function accuracy(data, labels)
  data, labels = ship2gpu(data, labels)
  local outputs = net:forward(data)
  local _, predict = torch.max(outputs, 2)
  if opt.gpuid < 0 then
    predict = predict:squeeze()
  end
  local hit = 0
  if labels:size(1) == 1 then
    if predict == labels[1] then
      hit = 1
    end
  else
    for i = 1, labels:size(1) do
      if opt.gpuid < 0 then 
        if predict[i] == labels[i] then
          hit = hit + 1
        end
      else
        if predict[i][1] == labels[i] then
          hit = hit + 1
        end
      end
    end
  end
  return hit, labels:size(1)
end

local function cal_accuracy(split)
  local allhit, alln = 0, 0
  local batchIterator = loader:iterator(split)
  for i = 1, loader.split_size[split] do
--    local data, labels = loader:next_batch(split)
    local data, labels = batchIterator:next_batch()
    local hit, n = accuracy(data, labels)
    allhit = allhit + hit
    alln = alln + n
  end
  return allhit / alln
end

-- gradient check
--local nchecks = 10
--local eps = 1e-4
--for i = 1,10 do
----  local iparam = math.random(params:size(1))
--  local iparam = i
--  trainIter.reset()
--  local lossOld, gradsOld = feval(params)
--  local analytical = gradsOld[iparam]
--  local oldparam = params[iparam]
--  params[iparam] = oldparam + eps
--  trainIter.reset()
--  local loss1, _ = feval(params)
--  params[iparam] = oldparam - eps
--  trainIter.reset()
--  local loss2, _ = feval(params)
--  local numerical = (loss1 - loss2) / 2 / eps
----  print(numerical, analytical)
--  if numerical ~=0 then
--    print('diff: '.. math.abs((analytical - numerical) / numerical)*100 .. '%')
--  end
--  params[iparam] = oldparam
--end

trainIter.reset()

--local trainData, trainLabels = loader:get_train_data()
--local validateData, validateLabels = loader:get_validate_data()

-- training
--optim.opt = {learningRate = opt.learning_rate}
local optim_opt = {learningRate = opt.learning_rate} 
local niterations = loader.training_size / loader.batch_size * opt.epochs
print(niterations)
--local epochs = torch.Tensor(niterations)
--local losses = torch.Tensor(niterations)
--local train_accuracy = torch.Tensor(opt.epochs)
--local validate_accuracy = torch.Tensor(opt.epochs)
--local i_epoch = 1

if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end
lfs.chdir(opt.checkpoint_dir)
if not path.exists(opt.model) then lfs.mkdir(opt.model) end

local print_every = math.ceil(niterations / opt.nCheckpoints)
local checkpoint = {}
for i = 1, niterations do
  local _, loss = optim.adagrad(feval, params, optim_opt)
--  local _, loss = optim.sgd(feval, params, optim.opt)
--  epochs[i] = math.ceil(i * loader.batch_size / loader.training_size)
--  losses[i] = loss[1]

  if i % print_every == 0 or i == niterations then
--    print(loss[1])
    local epoch = math.ceil(i * loader.batch_size / loader.training_size)
    local savefile = string.format('%s/batch_%d_iter%.2f_%.4f.t7', opt.model, loader.batch_size, i, loss[1])
    checkpoint.epoch = epoch
    checkpoint.iteration = i
    checkpoint.opt = opt
    checkpoint.model = net
    checkpoint.loss = loss[1]
    checkpoint.train_accuracy = cal_accuracy("train")
    checkpoint.validate_accuracy = cal_accuracy("validate")
    print("i = ", i, " epoch = ", epoch, "train_loss = ", loss[1], "val_accuracy = ", checkpoint.validate_accuracy)
    torch.save(savefile, checkpoint)
  end
  
  xlua.progress(i,niterations)
  
--  if epochs[i] > i_epoch then
--    train_accuracy[i_epoch] = accuracy(trainData, trainLabels)
--    validate_accuracy[i_epoch] = accuracy(validateData, validateLabels)
--    i_epoch = i_epoch + 1
--  end
end

--if i_epoch == opt.epochs then
--  train_accuracy[i_epoch] = accuracy(trainData, trainLabels)
--  validate_accuracy[i_epoch] = accuracy(validateData, validateLabels)
--end

-- make plots
--gnuplot.raw('set multiplot layout 1,2')
--gnuplot.plot(epochs, losses)
--local i_epochs = torch.Tensor(opt.epochs)
--for i = 1, opt.epochs do
--  i_epochs[i] = i
--end
--gnuplot.plot({i_epochs, train_accuracy}, {i_epochs, validate_accuracy})



