import torch
import torchvision
from tool import imutils

import argparse
import importlib
import numpy as np

import voc12.data
from torch.utils.data import DataLoader
import scipy.misc
import torch.nn.functional as F
import os.path

def get_indices_in_radius(height, width, radius):

    search_dist = []
    for x in range(1, radius):
        search_dist.append((0, x))

    for y in range(1, radius):
        for x in range(-radius+1, radius):
            if x*x + y*y < radius*radius:
                search_dist.append((y, x))

    full_indices = np.reshape(np.arange(0, height * width, dtype=np.int64),
                              (height, width))
    radius_floor = radius-1
    cropped_height = height - radius_floor
    cropped_width = width - 2 * radius_floor

    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor], [-1])

    indices_from_to_list = []

    for dy, dx in search_dist:

        indices_to = full_indices[dy:dy + cropped_height, radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])

        indices_from_to = np.stack((indices_from, indices_to), axis=1)

        indices_from_to_list.append(indices_from_to)

    concat_indices_from_to = np.concatenate(indices_from_to_list, axis=0)

    return concat_indices_from_to


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="network.resnet38_aff", type=str)
    parser.add_argument("--infer_list", default="voc12/val.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--cam_dir", required=True, type=str)
    parser.add_argument("--voc12_root", default='VOC2012', type=str)
    parser.add_argument("--alpha", default=6, type=float)
    parser.add_argument("--out_rw", default='out_rw', type=str)
    parser.add_argument("--beta", default=8, type=int)
    parser.add_argument("--logt", default=6, type=int)
    parser.add_argument("--crf", default=False, type=bool)

    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), 'Net')()

    model.load_state_dict(torch.load(args.weights), strict=False)

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ImageDataset(args.infer_list, voc12_root=args.voc12_root,
                                               transform=torchvision.transforms.Compose(
        [np.asarray,
         model.normalize,
         imutils.HWC_to_CHW]))
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    for iter, (name, img) in enumerate(infer_data_loader):

        name = name[0]
        print(iter)

        orig_shape = img.shape
        padded_size = (int(np.ceil(img.shape[2]/8)*8), int(np.ceil(img.shape[3]/8)*8))

        p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
        img = F.pad(img, p2d)

        dheight = int(np.ceil(img.shape[2]/8))
        dwidth = int(np.ceil(img.shape[3]/8))

        cam = np.load(os.path.join(args.cam_dir, name + '.npy'), allow_pickle=True).item()

        cam_full_arr = np.zeros((21, orig_shape[2], orig_shape[3]), np.float32)
        for k, v in cam.items():
            cam_full_arr[k+1] = v
        cam_full_arr[0] = (1 - np.max(cam_full_arr[1:], (0), keepdims=False))**args.alpha
        #cam_full_arr[0] = 0.2
        cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')

        with torch.no_grad():
            aff_mat = torch.pow(model.forward(img.cuda(), True), args.beta)

            trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
            for _ in range(args.logt):
                trans_mat = torch.matmul(trans_mat, trans_mat)

            cam_full_arr = torch.from_numpy(cam_full_arr)
            cam_full_arr = F.avg_pool2d(cam_full_arr, 8, 8)

#            if args.crf:    
#                img_8 = F.interpolate(img, (dheight,dwidth), mode='bilinear')[0].numpy().transpose((1,2,0))
#                img_8 = img[0].numpy().transpose((1,2,0))
#                img_8 = np.ascontiguousarray(img_8)
#                mean = (0.485, 0.456, 0.406)
#                std = (0.229, 0.224, 0.225)
#                img_8[:,:,0] = (img_8[:,:,0]*std[0] + mean[0])*255
#                img_8[:,:,1] = (img_8[:,:,1]*std[1] + mean[1])*255
#                img_8[:,:,2] = (img_8[:,:,2]*std[2] + mean[2])*255
#                img_8[img_8 > 255] = 255
#                img_8[img_8 < 0] = 0
#                img_8 = img_8.astype(np.uint8)
#                cam_full_arr = cam_full_arr.cpu().numpy()
#                cam_full_arr = imutils.crf_inference(img_8, cam_full_arr, t=1)
#                cam_full_arr = torch.from_numpy(cam_full_arr).view(1, 21, dheight, dwidth).cuda()
            cam_vec = cam_full_arr.view(21, -1)

            cam_rw = torch.matmul(cam_vec.cuda(), trans_mat)
            cam_rw = cam_rw.view(1, 21, dheight, dwidth)
       
            cam_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw)

            if args.crf:    
                img_8 = img[0].numpy().transpose((1,2,0))#F.interpolate(img, (dheight,dwidth), mode='bilinear')[0].numpy().transpose((1,2,0))
                img_8 = np.ascontiguousarray(img_8)
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
                img_8[:,:,0] = (img_8[:,:,0]*std[0] + mean[0])*255
                img_8[:,:,1] = (img_8[:,:,1]*std[1] + mean[1])*255
                img_8[:,:,2] = (img_8[:,:,2]*std[2] + mean[2])*255
                img_8[img_8 > 255] = 255
                img_8[img_8 < 0] = 0
                img_8 = img_8.astype(np.uint8)
                cam_rw = cam_rw[0].cpu().numpy()
                cam_rw = imutils.crf_inference(img_8, cam_rw, t=1)
                cam_rw = torch.from_numpy(cam_rw).view(1, 21, img.shape[2], img.shape[3]).cuda()


            _, cam_rw_pred = torch.max(cam_rw, 1)

            res = np.uint8(cam_rw_pred.cpu().data[0])[:orig_shape[2], :orig_shape[3]]

            scipy.misc.imsave(os.path.join(args.out_rw, name + '.png'), res)
