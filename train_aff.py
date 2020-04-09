
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils
import argparse
import importlib



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    parser.add_argument("--network", default="network.resnet38_aff", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="resnet38_aff", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--voc12_root", default='VOC2012', type=str)
    parser.add_argument("--la_crf_dir", required=True, type=str)
    parser.add_argument("--ha_crf_dir", required=True, type=str)
    args = parser.parse_args()

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))

    model = getattr(importlib.import_module(args.network), 'Net')()

    print(model)


    train_dataset = voc12.data.VOC12AffDataset(args.train_list, label_la_dir=args.la_crf_dir, label_ha_dir=args.ha_crf_dir,
                                               voc12_root=args.voc12_root, cropsize=args.crop_size, radius=5,
                    joint_transform_list=[
                        None,
                        None,
                        imutils.RandomCrop(args.crop_size),
                        imutils.RandomHorizontalFlip()
                    ],
                    img_transform_list=[
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        np.asarray,
                        model.normalize,
                        imutils.HWC_to_CHW
                    ],
                    label_transform_list=[
                        None,
                        None,
                        None,
                        imutils.AvgPool2d(8)
                    ])
    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        import network.resnet38d
        assert args.network == "network.resnet38_aff"
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss', 'bg_loss', 'fg_loss', 'neg_loss', 'bg_cnt', 'fg_cnt', 'neg_cnt')

    timer = pyutils.Timer("Session started: ")

    for ep in range(args.max_epoches):

        for iter, pack in enumerate(train_data_loader):

            aff = model.forward(pack[0])

            bg_label = pack[1][0].cuda(non_blocking=True)
            fg_label = pack[1][1].cuda(non_blocking=True)
            neg_label = pack[1][2].cuda(non_blocking=True)

            bg_count = torch.sum(bg_label) + 1e-5
            fg_count = torch.sum(fg_label) + 1e-5
            neg_count = torch.sum(neg_label) + 1e-5

            bg_loss = torch.sum(- bg_label * torch.log(aff + 1e-5)) / bg_count
            fg_loss = torch.sum(- fg_label * torch.log(aff + 1e-5)) / fg_count
            neg_loss = torch.sum(- neg_label * torch.log(1. + 1e-5 - aff)) / neg_count

            loss = bg_loss/4 + fg_loss/4 + neg_loss/2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({
                'loss': loss.item(),
                'bg_loss': bg_loss.item(), 'fg_loss': fg_loss.item(), 'neg_loss': neg_loss.item(),
                'bg_cnt': bg_count.item(), 'fg_cnt': fg_count.item(), 'neg_cnt': neg_count.item()
            })

            if (optimizer.global_step - 1) % 50 == 0:

                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step-1, max_step),
                      'loss:%.4f %.4f %.4f %.4f' % avg_meter.get('loss', 'bg_loss', 'fg_loss', 'neg_loss'),
                      'cnt:%.0f %.0f %.0f' % avg_meter.get('bg_cnt', 'fg_cnt', 'neg_cnt'),
                      'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                avg_meter.pop()


        else:
            print('')
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.session_name + '.pth')
