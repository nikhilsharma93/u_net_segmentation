------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------

require 'torch'
require 'nn'
local nninit = require 'nninit'
require 'nngraph'
--require 'MultiCrossEntropy'

----------------------------------------------------------------------
-- parameters:
local filtsize = 3

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> construct model')



function convolutional(no_inp, no_op, filtsize, out_batchNorm, out_Dropout)
  local out_Dropout = out_Dropout or false

  local modelConv = nn.Sequential()

  modelConv:add(nn.SpatialConvolutionMM(no_inp, no_op, filtsize, filtsize, 1,1,0,0)
            :init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
  modelConv:add(nn.SpatialBatchNormalization(no_op))
  modelConv:add(nn.ReLU())

  modelConv:add(nn.SpatialConvolutionMM(no_op, no_op, filtsize, filtsize, 1,1,0,0)
            :init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
  if out_batchNorm==true then
    modelConv:add(nn.SpatialBatchNormalization(no_op))
  end
  modelConv:add(nn.ReLU())
  if out_Dropout==true then
    modelConv:add(nn.Dropout(0.5))
  end

  return modelConv
end

function blockBranch(block, no_inp, no_op,cropW)
  local subBlock = nn.Sequential()
    subBlock:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
    subBlock:add(block)
    subBlock:add(nn.SpatialFullConvolution(no_inp, no_op, 2, 2, 2, 2)
            :init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))

  local crop = nn.Sequential()
    crop:add(nn.SpatialZeroPadding(-cropW,-cropW,-cropW,-cropW))

  local parallel = nn.ConcatTable(2)
    parallel:add(crop)
    parallel:add(subBlock)

  local modelBlock = nn.Sequential()
    modelBlock:add(parallel)
    modelBlock:add(nn.JoinTable(1,3))

  return modelBlock
end



local block0 = convolutional(512,1024,filtsize,true,true)

local block1 = nn.Sequential()
  block1:add(convolutional(256,512,filtsize,true,true))
  block1:add(blockBranch(block0,1024,512,4))
  block1:add(convolutional(1024,512,filtsize,true))

local block2 = nn.Sequential()
  block2:add(convolutional(128,256,filtsize,true))
  block2:add(blockBranch(block1,512,256,16))
  block2:add(convolutional(512,256,filtsize,true))

local block3 = nn.Sequential()
  block3:add(convolutional(64,128,filtsize,true))
  block3:add(blockBranch(block2,256,128,40))
  block3:add(convolutional(256,128,filtsize,true))

local block4 = nn.Sequential()
  block4:add(convolutional(1,64,filtsize,true))
  block4:add(blockBranch(block3,128,64,88))
  block4:add(convolutional(128,64,filtsize,true))
  block4:add(nn.SpatialConvolutionMM(64, 2, 1, 1, 1, 1, 0, 0)
        :init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))


local model = nn.Sequential()
model:add(block4)
model:add(nn.LogSoftMax())


for _,layer in ipairs(block4.modules) do
   if layer.bias then
      layer.bias:fill(.2)
      if i == #block4.modules-1 then
         layer.bias:zero()
      end
   end
end


loss = nn.SpatialClassNLLCriterion()


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> here is the CNN:')
print(model)


-- return package:
return {
   model = model,
   loss = loss,
}
