require 'nn'

function mlp(input_size, n_classes)
--  local L1, L2 = 2048, 256
  local L1 = 512
  local net = nn.Sequential()
  net:add(nn.Reshape(input_size))
  net:add(nn.Linear(input_size, L1))
  net:add(nn.Tanh())
--  net:add(nn.Linear(L1, L2))
--  net:add(nn.Sigmoid())
  net:add(nn.Linear(L1, n_classes))
--  net:add(nn.LogSoftMax())
--  local criterion = nn.ClassNLLCriterion()
  local criterion = nn.CrossEntropyCriterion()
  
  return net, criterion
end

function convnet(n_classes)
  local net = nn.Sequential()
  -- conv -> relu -> pool -> conv -> relu -> pool -> fc
  net:add(nn.SpatialConvolutionMM(1,32,5,5,1,1,2,2)) -- output: 32x32x32, 32 filters, output dim = (32 - 5 + 2 * 2) / 1 + 1 = 32
  net:add(nn.ReLU())
  net:add(nn.SpatialMaxPooling(2,2)) -- output: 32x16x16
  net:add(nn.SpatialConvolutionMM(32,16,5,5,1,1,2,2)) -- output: 16x16x16
  net:add(nn.ReLU())
  net:add(nn.SpatialMaxPooling(2,2)) -- output: 16x8x8
  net:add(nn.Reshape(16*8*8))
  net:add(nn.Linear(16*8*8, 256))
  net:add(nn.ReLU())
  net:add(nn.Linear(256, n_classes))
--  net:add(nn.LogSoftMax())
--  
--  local criterion = nn.ClassNLLCriterion()
  local criterion = nn.CrossEntropyCriterion()
  
  return net, criterion
end