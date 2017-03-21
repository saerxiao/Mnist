require 'lfs'
require 'torch'
require 'gnuplot'
require 'nn'
require 'cunn'
require 'cutorch'

function plot_accuracy(dir)
  local current = lfs.currentdir()
  dir = current .. "/" .. dir
  lfs.chdir(dir)
  print(lfs.currentdir())
--  local dir = 'checkpoint/'
  local files = {}
  for file in lfs.dir(dir) do
    if lfs.attributes(dir .. "/" .. file, "mode") == "file" then
      table.insert(files,dir .. "/" .. file)
    end
  end

  -- sort files by iteration number
  local function find_iter(str)
    local iter_index = string.find(str, 'iter')
    return tonumber(string.sub(str, iter_index + 4, string.find(str,'_',iter_index) - 1))
  end

  table.sort(files, function (a,b) return find_iter(a) < find_iter(b) end)

  local epochs = torch.Tensor(#files)
  local losses = torch.Tensor(#files)
  local train_accuracy = torch.Tensor(#files)
  local validate_accuracy = torch.Tensor(#files)
  for i = 1, #files do
    local f = torch.load(files[i])
    epochs[i] = f.epoch
    losses[i] = f.loss
    train_accuracy[i] = f.train_accuracy
    validate_accuracy[i] = f.validate_accuracy
    
--    if i == 10 then
--      local weights = f.model.modules[2].weight:view(f.model.modules[2].weight:size(1), 32, 32)
--      gnuplot.raw('set multiplot layout 2,2')
--      for j = 1, 4 do
--        gnuplot.imagesc(weights[j])
--      end
--    end
  end

-- make plots
  gnuplot.raw('set multiplot layout 1,2')
  gnuplot.plot(epochs, losses, '+-')
  gnuplot.plot({epochs, train_accuracy, '+'}, {epochs, validate_accuracy, '+'})
end

function plot_mlp_firstlayer_feature(file, ifeature)
  local f = torch.load("checkpoint/" .. file)
  local weights = f.model.modules[2].weight:view(f.model.modules[2].weight:size(1), 32, 32)
  gnuplot.imagesc(weights[ifeature])
end

function inspect_model(file)
  local f = torch.load("checkpoint/" .. file)
  for i = 1, #f.model.modules do
    print(#f.model.modules[i].output)
  end
end

function plot_input(file)
  local f = torch.load("checkpoint/" .. file)
  print(#f.model.modules[1].finput)
  print(#f.model.modules[1].output)
  print(#f.model.modules[1].weight)
--  gnuplot.imagesc(f.model.modules[1].output[1][12])
  local weight = f.model.modules[1].weight:view(f.model.modules[1].weight:size(1),5,5)
  gnuplot.raw('set multiplot layout 2,2')
  for i = 5, 8 do
    gnuplot.imagesc(weight[i])
  end
end

function plot_activation(file, layer)
  local f = torch.load("checkpoint/" .. file)
  print(#f.model.modules[layer].output)
  -- plot activation hist for one sample
  gnuplot.hist(f.model.modules[layer].output[1], 50, -1, 1)
end



