import torch
import torchvision.models as models
import matplotlib.pylab as plt

from PIL import Image
from network.deeplabv3.deeplabv3 import *
from build_data import *
from module_list import *

aug_semi_save_dir = ""
aug_sup_save_dir = ""

im_size = [513, 513]
root = 'dataset/pascal'
with open(root + '/test_aug.txt') as f:
    idx_list = f.read().splitlines()

num_segments = 5
device = torch.device("cpu")
model = DeepLabv3Plus(models.resnet101(), num_classes=num_segments).to(device)
colormap = create_pascal_label_colormap()

id_list = [1]
for id in range(len(idx_list)):
    im_id = idx_list[id]
    im = Image.open(root + '/JPEGImages/{}.jpg'.format(im_id))
    gt_label = Image.open(root + '/SegmentationClassAug/{}.png'.format(im_id))
    im_tensor, label_tensor = transform(im, gt_label, None, crop_size=im_size, scale_size=(1.0, 1.0), augmentation=False)
    im_w, im_h = im.size

    model.load_state_dict(torch.load('model_weights/pascal_label0_sup_reco_0.pth', map_location=torch.device('cpu')))
    model.eval() 
    logits, _ = model(im_tensor.unsqueeze(0))
    logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=True)
    max_logits, label_sup = torch.max(torch.softmax(logits, dim=1), dim=1)
    label_sup[label_tensor == -1] = -1

    model.load_state_dict(torch.load('model_weights/pascal_label0_semi_classmix_reco_0.pth', map_location=torch.device('cpu')))
    model.eval()
    logits, rep = model(im_tensor.unsqueeze(0))
    logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=True)
    max_logits, label_reco = torch.max(torch.softmax(logits, dim=1), dim=1)
    label_reco[label_tensor == -1] = -1


    gt_blend = Image.blend(im, Image.fromarray(color_map(label_tensor[0].numpy(), colormap)[:im_h, :im_w]), alpha=0.7)
    sup_blend = Image.blend(im, Image.fromarray(color_map(label_sup[0].numpy(), colormap)[:im_h, :im_w]), alpha=0.7)
    reco_blend = Image.blend(im, Image.fromarray(color_map(label_reco[0].numpy(), colormap)[:im_h, :im_w]), alpha=0.7)


    im_id = im_id.replace('_test','')
    print('Creating... ', im_id)
    im.save(aug_semi_save_dir + str(im_id) + '_1_im.png')
    gt_blend.save(aug_semi_save_dir + str(im_id) + '_gt_blend.png')
    reco_blend.save(aug_semi_save_dir + str(im_id) + '_4_reco_blend.png')