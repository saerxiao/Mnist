require 'nn'
require 'paths'
require 'lfs'
require 'pl'
require 'gnuplot'
require 'inspectCheckpoints'

local savefile = string.format('%s/epoch%d_i%d.t7', "h1", 1,100)
print(savefile)
--plot_accuracy("checkpoint/convnet")
--plot_mlp_firstlayer_feature("lm_mlp512_iter44.00_0.4037.t7", 10)
--inspect_model("lm_mlp512_iter900.00_0.2798.t7")
--plot_activation("lm_mlp512_iter900.00_0.2798.t7", 3)
--plot_input("lm_convnet_batch_50_iter18.00_0.5972.t7")

--x = torch.ones(3)
--x[2] = 2
--x[3] = 3
--y = torch.ones(3)
--y[2] = 3
--gnuplot.plot(x,y,'+')
--print('done')

-- "plot  '-' title '' with lines\n"
--"plot  '-' title '' with dots\n"
--"1 1\n2 3\n3 1\ne\n"

--local current = lfs.currentdir()
--local dir = current .. "/checkpoint/convnet"
--for file in lfs.dir(dir) do
--  print(file)
--end
--local dir = current .. "/test/model1"
--lfs.chdir(dir)
--lfs.chdir("test")
--print(lfs.currentdir())
--lfs.chdir("model1")
--print(lfs.currentdir())
--local dir = 'test'
--if not path.exists(dir) then lfs.mkdir(dir) end
--lfs.chdir(dir)
--local subdir = 'model2'
--if not path.exists(subdir) then lfs.mkdir(subdir) end
--
--local checkpoint = {}
--checkpoint.data = torch.ones(3)
--local filename = string.format('%s/test.t7', subdir)
--torch.save(filename, checkpoint)

--local filename = 'checkpoint/lm_mlp_epoch1.00_1.4183.t7'
--local f = torch.load(filename)
--print(f.iteration)

--local files = {}
--local dir = 'checkpoint/'
--for file in lfs.dir(dir) do
----  print(file, lfs.attributes(dir .. file, "mode"))
--  if lfs.attributes(dir .. file, "mode") == "file" then
--    print(file, lfs.attributes(dir .. file, "mode"))
--    table.insert(files,dir .. file)
----    local f = torch.load(dir .. file)
----    print(f.iteration)
--  end
--end
--
--local function find_iter(str)
--  local iter_index = string.find(str, 'iter')
--  return tonumber(string.sub(str, iter_index + 4, string.find(str,'_',iter_index) - 1))
--end
--
--table.sort(files, function (a,b) return find_iter(a) < find_iter(b) end)
--for i = 1, #files do
--  print(files[i])
--end

--local mnist = require 'Loader'
--local m1 = require 'TestModule'
--print(m1)
--local m1loader = m1.create{}
--m1loader:public()

--local loader = mnist.create{train_max_load = 10, batch_size = 1}
--print(loader.trainingData)
--print(loader.trainingData.labels[1])

--local path_remote = 'https://s3.amazonaws.com/torch7/data/mnist.t7.tgz'
--local path_dataset = 'mnist.t7'
--local path_trainset = paths.concat(path_dataset, 'train_32x32.t7')
--trainset = loadDataset(path_trainset)

--trainset = mnist.loadTrainSet()
--testset = mnist.loadTestSet()
--print(testset)

--cmd = torch.CmdLine()
--cmd:option('-learning_rate',1e-1,'learning rate')
--cmd:option('-iterations',10,'number of iterations')
--cmd:option('-print_every',1,'print every')
--opt = cmd:parse(arg)
--
--print(opt.learning_rate)

--local mnist = require 'mnist'
--local trainset = mnist.traindataset()
--local testset = mnist.testdataset()
--print(trainset)

--reshape = nn.View(10000, 28*28):forward(testset)