import time, torchvision, argparse, logging, sys, os
import torch, random
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
from utils.UTILS1 import compute_psnr
from utils.UTILS import AverageMeters, print_args_parameters, compute_ssim
from datasets.datasets_pairs import my_dataset_eval
from networks.aaln_arch import ShadowRemovalUNet
from networks.image_utils import splitimage, mergeimage

sys.path.append(os.getcwd())


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

# arch
parser.add_argument('--base_channel', type=int, default=32)
parser.add_argument('--num_res', type=int, default=6)
parser.add_argument('--img_channel', type=int, default=3)
parser.add_argument('--enc_blks', nargs='+', type=int, default=[1, 1, 1, 28])
parser.add_argument('--dec_blks', nargs='+', type=int, default=[1, 1, 1, 1])
parser.add_argument('--kernel_size', type=int, default=3)

# test setting
parser.add_argument('--Crop_patches', type=int, default=512)
parser.add_argument('--overlap_size', type=int, default=256)
parser.add_argument('--experiment_name', type=str, default="test")
parser.add_argument('--result_path', type=str, default='./results/')
parser.add_argument('--eval_in_path', type=str, default='./test_images/')
parser.add_argument('--eval_gt_path', type=str, default='')

# model path
parser.add_argument('--pre_model', type=str, default='./ckpt/best_model.pth')

# input ensemble
parser.add_argument('--inputs_ensemble', type=str2bool, default=True)

args = parser.parse_args()
if not args.eval_gt_path:
    args.eval_gt_path = args.eval_in_path
print_args_parameters(args)

log_dir = args.result_path + '/log_file/'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(args.result_path, exist_ok=True)

trans_eval = transforms.Compose([transforms.ToTensor()])
results_metrics = log_dir + args.experiment_name + '.txt'

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def input_ensemble(net, inp):
    """Input ensemble: normal + flipH + flipW + flipHW, average results."""
    out0 = net(inp)

    inp_h = torch.flip(inp, dims=[3])
    out_h = torch.flip(net(inp_h), dims=[3])

    inp_v = torch.flip(inp, dims=[2])
    out_v = torch.flip(net(inp_v), dims=[2])

    inp_hv = torch.flip(inp, dims=[2, 3])
    out_hv = torch.flip(net(inp_hv), dims=[2, 3])

    return (out0 + out_h + out_v + out_hv) / 4.0


def test(net, eval_loader, Dname='test', save_result=True):
    net.eval()
    with torch.no_grad():
        eval_meters = AverageMeters()
        st = time.time()

        for index, (data_in, label, name) in enumerate(eval_loader, 0):
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)
            B, C, H, W = inputs.shape

            if index == 0:
                print(f"  input: {data_in.size()}, gt: {label.size()}")

            # split into overlapped patches for large images
            if H > args.Crop_patches or W > args.Crop_patches:
                split_data, starts = splitimage(inputs, crop_size=args.Crop_patches,
                                                overlap_size=args.overlap_size)
                for i, patch in enumerate(split_data):
                    if args.inputs_ensemble:
                        split_data[i] = input_ensemble(net, patch)
                    else:
                        split_data[i] = net(patch)
                outputs = mergeimage(split_data, starts, crop_size=args.Crop_patches,
                                     resolution=(B, C, H, W), is_mean=True)
            else:
                if args.inputs_ensemble:
                    outputs = input_ensemble(net, inputs)
                else:
                    outputs = net(inputs)

            out_psnr = compute_psnr(outputs, labels)
            out_psnr_clip = compute_psnr(torch.clamp(outputs, 0., 1.), labels)
            out_ssim = compute_ssim(outputs, labels)
            in_psnr = compute_psnr(inputs, labels)
            in_ssim = compute_ssim(inputs, labels)

            eval_meters.update({
                'out_psnr': out_psnr,
                'out_psnr_clip': out_psnr_clip,
                'in_psnr': in_psnr,
                'out_ssim': out_ssim,
                'in_ssim': in_ssim,
            })

            content = (f"index:{index} | {name[0]} | in_psnr:{in_psnr:.3f} in_ssim:{in_ssim:.4f} | "
                       f"out_psnr:{out_psnr:.3f} out_psnr_clip:{out_psnr_clip:.3f} out_ssim:{out_ssim:.4f}")
            print(content)
            with open(results_metrics, 'a') as f:
                f.write(content + '\n')

            if save_result:
                save_path = args.result_path + '/'
                os.makedirs(save_path, exist_ok=True)
                torchvision.utils.save_image(
                    [torch.clamp(outputs, 0., 1.).cpu().detach()[0]],
                    save_path + name[0], nrow=1, padding=0)

        # summary
        summary = (f"Dataset:{Dname} | Num:{len(eval_loader)} | "
                   f"In_PSNR:{eval_meters['in_psnr']:.3f} In_SSIM:{eval_meters['in_ssim']:.4f} | "
                   f"Out_PSNR:{eval_meters['out_psnr']:.3f} Out_PSNR_clip:{eval_meters['out_psnr_clip']:.3f} "
                   f"Out_SSIM:{eval_meters['out_ssim']:.4f} | time:{time.time()-st:.1f}s")
        print(summary)
        with open(results_metrics, 'a') as f:
            f.write(summary + '\n')


def get_eval_data(val_in_path, val_gt_path, trans=trans_eval):
    eval_data = my_dataset_eval(root_in=val_in_path, root_label=val_gt_path,
                                transform=trans, fix_sample=500)
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=4)
    return eval_loader


if __name__ == '__main__':
    # build shadow removal model (no global residual)
    net = ShadowRemovalUNet(
        img_channel=args.img_channel,
        width=args.base_channel,
        middle_blk_num=args.num_res,
        enc_blk_nums=args.enc_blks,
        dec_blk_nums=args.dec_blks,
        kernel_size=args.kernel_size,
    )

    # load weights
    net.load_state_dict(torch.load(args.pre_model, map_location=device), strict=True)
    print('-----' * 20, 'loaded model weights!')

    net.to(device)
    print('#parameters:', sum(p.numel() for p in net.parameters()))

    eval_loader = get_eval_data(val_in_path=args.eval_in_path, val_gt_path=args.eval_gt_path)

    test(net=net, eval_loader=eval_loader, Dname=args.experiment_name, save_result=True)

    with open(results_metrics, 'a') as f:
        f.write('-=-=' * 50 + '\n')
