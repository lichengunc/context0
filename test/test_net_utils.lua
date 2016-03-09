package.path = '../?.lua;' .. package.path

require 'torch'
require 'nn'
require 'cutorch'
require 'cudnn'
require 'nngraph'
-- exotic things
require 'loadcaffe'
-- local imports
require 'misc.DataLoader'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'

local cnn_proto = '../model/vgg/VGG_ILSVRC_16_layers_deploy.prototxt'
local cnn_model = '../model/vgg/VGG_ILSVRC_16_layers.caffemodel'
local cnn_raw = loadcaffe.load(cnn_proto, cnn_model, 'cudnn')


local opt = {}
opt.init_norm = 10
opt.dropout = 0.0
opt.visual_encoding_size = 512
opt.use_context = 1
opt.use_location = 1
opt.use_dif = 1
local cxt_feats = torch.randn(5, 4096):cuda()
local ann_feats = torch.randn(5, 4096):cuda()
local lfeats = torch.randn(5, 5):cuda()
local nb_ann_feats = torch.randn(5, 4096):cuda()
local nb_lfeats = torch.randn(5, 25):cuda()

-- local dif_encoder = net_utils.build_dif_encoder(cnn_raw, opt):cuda()
-- print(dif_encoder:forward({nb_ann_feats, nb_lfeats}))

-- local jemb = net_utils.build_visual_encoder(cnn_raw, cnn_raw, opt):cuda()
-- local out = jemb:forward{cxt_feats, ann_feats, lfeats}
-- print(out:size())
-- jemb:backward({cxt_feats, ann_feats, lfeats}, out)

local jemb = net_utils.build_jemb(cnn_raw, cnn_raw, opt):cuda()
local out = jemb:forward{cxt_feats, ann_feats, lfeats, nb_ann_feats, nb_lfeats}
print(out:size())
jemb:backward({cxt_feats, ann_feats, lfeats, nb_ann_feats, nb_lfeats}, out)

