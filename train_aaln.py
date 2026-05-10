import time, torchvision, argparse, logging, sys, os, gc
import torch, random
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from utils.UTILS1 import compute_psnr
from utils.UTILS import AverageMeters, print_args_parameters
import loss.losses as losses
from torch.utils.tensorboard import SummaryWriter
from datasets.datasets_pairs import my_dataset, my_dataset_eval, my_dataset_wTxt
from networks.aaln_arch import ShadowRemovalUNet
from networks.diffusion_shadow import DiffusionShadowGenerator, apply_shadow_mask
from networks.image_utils import splitimage, mergeimage

sys.path.append(os.getcwd())


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

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

# path
parser.add_argument('--experiment_name', type=str, default="train_aaln")
parser.add_argument('--unified_path', type=str, default='./experiments/')
parser.add_argument('--training_path', type=str, default='./data/')
parser.add_argument('--training_path_txt', nargs='*', default=['./data/train_list.txt'])
parser.add_argument('--writer_dir', type=str, default='./tf-logs/')
parser.add_argument('--eval_in_path', type=str, default='./data/val_input/')
parser.add_argument('--eval_gt_path', type=str, default='./data/val_gt/')

# training
parser.add_argument('--EPOCH', type=int, default=1000)
parser.add_argument('--T_period', type=int, default=50)
parser.add_argument('--BATCH_SIZE', type=int, default=12)
parser.add_argument('--Crop_patches', type=int, default=512)
parser.add_argument('--learning_rate', type=float, default=4e-4)
parser.add_argument('--print_frequency', type=int, default=50)
parser.add_argument('--SAVE_Inter_Results', type=str2bool, default=False)
parser.add_argument('--max_psnr', type=int, default=40)
parser.add_argument('--fix_sampleA', type=int, default=30000)
parser.add_argument('--debug', type=str2bool, default=False)
parser.add_argument('--Aug_regular', type=str2bool, default=False)
parser.add_argument('--grad_accum_steps', type=int, default=1)

# arch
parser.add_argument('--base_channel', type=int, default=32)
parser.add_argument('--num_res', type=int, default=6)
parser.add_argument('--img_channel', type=int, default=3)
parser.add_argument('--enc_blks', nargs='+', type=int, default=[1, 1, 1, 28])
parser.add_argument('--dec_blks', nargs='+', type=int, default=[1, 1, 1, 1])
parser.add_argument('--kernel_size', type=int, default=3)

# loss
parser.add_argument('--base_loss', type=str, default='char')
parser.add_argument('--addition_loss', type=str, default='fft')
parser.add_argument('--addition_loss_coff', type=float, default=0.02)

# load pre-trained model
parser.add_argument('--load_pre_model', type=str2bool, default=False)
parser.add_argument('--pre_model', type=str, default='')

# optimizer
parser.add_argument('--optim', type=str, default='adam')

# training stage
parser.add_argument('--training_stage', type=int, default=1)

# diffusion shadow generation (VP-SDE)
parser.add_argument('--use_shadow_gen', type=str2bool, default=True)
parser.add_argument('--shadow_model_path', type=str, default='./ckpt/diffusion_shadow.pth')
parser.add_argument('--shadow_prob', type=float, default=0.3)
parser.add_argument('--shadow_width', type=int, default=32)
parser.add_argument('--shadow_sample_steps', type=int, default=50)
parser.add_argument('--shadow_sample_method', type=str, default='sde', choices=['sde', 'ode'])

args = parser.parse_args()
print_args_parameters(args)

if args.debug:
    fix_sampleA = 400
else:
    fix_sampleA = args.fix_sampleA

exper_name = args.experiment_name
writer = SummaryWriter(args.writer_dir + exper_name)
os.makedirs(args.writer_dir, exist_ok=True)

unified_path = args.unified_path
SAVE_PATH = unified_path + exper_name + '/'
os.makedirs(SAVE_PATH, exist_ok=True)

if args.SAVE_Inter_Results:
    SAVE_Inter_Results_PATH = SAVE_PATH + 'Inter_Temp_results/'
    os.makedirs(SAVE_Inter_Results_PATH, exist_ok=True)

logging.basicConfig(filename=SAVE_PATH + args.experiment_name + '.log', level=logging.INFO)
logging.info('=======================' * 2 + 'args' + '=======================' * 2)
for k in args.__dict__:
    logging.info(k + ": " + str(args.__dict__[k]))
logging.info('=======================' * 2 + '====' + '=======================' * 2)

trans_eval = transforms.Compose([transforms.ToTensor()])

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
print("==" * 50)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def test(net, eval_loader, epoch=1, max_psnr_val=26, Dname='val'):
    net.eval()
    with torch.no_grad():
        eval_meters = AverageMeters()
        st = time.time()
        for index, (data_in, label, name) in enumerate(eval_loader, 0):
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)

            if index == 0:
                print(f"  val_input: {data_in.size()}, gt: {label.size()}")

            outputs = net(inputs)

            eval_meters.update({
                'out_psnr': compute_psnr(outputs, labels),
                'in_psnr': compute_psnr(inputs, labels),
            })

        out_psnr = eval_meters['out_psnr']
        in_psnr = eval_meters['in_psnr']

        writer.add_scalars(exper_name + '/testing', {
            'out_PSNR': out_psnr, 'in_PSNR': in_psnr,
        }, epoch)

        if out_psnr > max_psnr_val:
            max_psnr_val = out_psnr
            torch.save(net.state_dict(), SAVE_PATH + 'best_model.pth')

        print(f"epoch:{epoch} [{Dname}] num:{len(eval_loader)} in_psnr:{in_psnr:.2f} "
              f"out_psnr:{out_psnr:.2f} best:{max_psnr_val:.2f} time:{time.time()-st:.1f}s")
        logging.info(f"epoch:{epoch} [{Dname}] num:{len(eval_loader)} in_psnr:{in_psnr:.2f} "
                     f"out_psnr:{out_psnr:.2f} best:{max_psnr_val:.2f}")

    return max_psnr_val


def save_imgs_for_visual(path, inputs, labels, outputs):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0]], path, nrow=3, padding=0)


def get_training_data(Crop_patches=args.Crop_patches):
    rootA = args.training_path
    rootA_txt_list = args.training_path_txt
    datasets_list = []
    for txt_path in rootA_txt_list:
        ds = my_dataset_wTxt(rootA, txt_path, crop_size=Crop_patches,
                             fix_sample_A=fix_sampleA, regular_aug=args.Aug_regular)
        datasets_list.append(ds)
    train_dataset = ConcatDataset(datasets_list)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.BATCH_SIZE,
                              num_workers=8, shuffle=True)
    print(f'len(train_loader): {len(train_loader)}')
    return train_loader


def get_eval_data(val_in_path=args.eval_in_path, val_gt_path=args.eval_gt_path):
    eval_data = my_dataset_eval(root_in=val_in_path, root_label=val_gt_path,
                                transform=trans_eval, fix_sample=500)
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=4)
    print(f'len(eval_loader): {len(eval_loader)}')
    return eval_loader


def print_param_number(net):
    print('#parameters:', sum(p.numel() for p in net.parameters()))


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

    # load pre-trained weights if resuming
    if args.load_pre_model and args.pre_model:
        net.load_state_dict(torch.load(args.pre_model, map_location=device), strict=True)
        print('-----' * 20, 'loaded pre-trained model!')
        logging.info('loaded pre-trained model from ' + args.pre_model)

    net.to(device)
    print_param_number(net)

    # load diffusion shadow generator (pre-trained, frozen)
    shadow_gen = None
    if args.use_shadow_gen and os.path.exists(args.shadow_model_path):
        shadow_gen = DiffusionShadowGenerator(width=args.shadow_width).to(device)
        shadow_gen.load_state_dict(torch.load(args.shadow_model_path, map_location=device))
        shadow_gen.eval()
        for p in shadow_gen.parameters():
            p.requires_grad = False
        print('loaded diffusion shadow generator (VP-SDE)')
    elif args.use_shadow_gen:
        print(f'[warning] shadow model not found at {args.shadow_model_path}, training without shadow gen')

    train_loader = get_training_data()
    eval_loader = get_eval_data()

    # optimizer
    if args.optim.lower() == 'sgd':
        optimizerG = optim.SGD(net.parameters(), lr=args.learning_rate)
    else:
        optimizerG = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))

    scheduler = CosineAnnealingWarmRestarts(optimizerG, T_0=args.T_period, T_mult=1)

    # base loss (Charbonnier)
    if args.base_loss.lower() == 'char':
        base_loss_fn = losses.CharbonnierLoss()
    else:
        base_loss_fn = nn.L1Loss()

    # additional loss
    if args.addition_loss.lower() == 'fft':
        addition_loss_fn = losses.fftLoss()
    elif args.addition_loss.lower() == 'ssim':
        addition_loss_fn = losses.SSIMLoss()
    elif args.addition_loss.lower() == 'lpips':
        addition_loss_fn = losses.LPIPSLoss()
    elif args.addition_loss.lower() == 'vgg':
        addition_loss_fn = losses.VGGLoss()
    else:
        addition_loss_fn = None

    running_results = {'iter_nums': 0, 'max_psnr_val': 0}
    train_meters = AverageMeters()
    optimizerG.zero_grad()

    for epoch in range(args.EPOCH):
        scheduler.step(epoch)
        st = time.time()

        for i, train_data in enumerate(train_loader, 0):
            data_in, label, img_name = train_data

            if i == 0:
                print(f"  train_input: {data_in.size()}, gt: {label.size()}")
                logging.info(f"  train_input: {data_in.size()}, gt: {label.size()}")

            running_results['iter_nums'] += 1
            net.train()

            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)

            # first pass: normal shadow removal
            outputs = net(inputs)
            loss_content = base_loss_fn(outputs, labels)
            total_loss = loss_content

            loss_add = torch.tensor(0.)
            if addition_loss_fn is not None:
                loss_add = args.addition_loss_coff * addition_loss_fn(outputs, labels)
                total_loss = total_loss + loss_add

            # Diffusion Shadow Generation: overlay generated shadows onto
            # restored images, then re-feed into the network (Fig.2)
            loss_shadow = torch.tensor(0.)
            if shadow_gen is not None and random.random() < args.shadow_prob:
                with torch.no_grad():
                    B, C, H, W = outputs.shape
                    shadow_mask = shadow_gen.sample((B, 1, H, W), device=device,
                                                    num_steps=args.shadow_sample_steps,
                                                    method=args.shadow_sample_method)
                    shadowed_output = apply_shadow_mask(
                        torch.clamp(outputs.detach(), 0., 1.), shadow_mask)

                # re-feed shadowed output into the network
                re_restored = net(shadowed_output)
                loss_shadow = base_loss_fn(re_restored, labels)
                total_loss = total_loss + 0.5 * loss_shadow

            # gradient accumulation
            total_loss = total_loss / args.grad_accum_steps
            total_loss.backward()

            if (i + 1) % args.grad_accum_steps == 0:
                optimizerG.step()
                optimizerG.zero_grad()

            in_psnr = compute_psnr(inputs, labels)
            out_psnr = compute_psnr(outputs, labels)

            train_meters.update({
                'loss': total_loss.item() * args.grad_accum_steps,
                'loss_content': loss_content.item(),
                'loss_add': loss_add.item(),
                'loss_shadow': loss_shadow.item(),
                'in_psnr': in_psnr,
                'out_psnr': out_psnr,
            })

            if (i + 1) % args.print_frequency == 0 and i > 1:
                writer.add_scalars(exper_name + '/training', {
                    'out_PSNR': train_meters['out_psnr'],
                    'in_PSNR': train_meters['in_psnr'],
                    'loss': train_meters['loss'],
                }, running_results['iter_nums'])

                print("epoch:%d [%d/%d] lr:%.7f loss:%.5f(char:%.4f,add:%.4f,shd:%.4f) "
                      "in:%.2f out:%.2f t:%.1f" % (
                    epoch, i+1, len(train_loader), optimizerG.param_groups[0]["lr"],
                    train_meters['loss'], loss_content.item(), loss_add.item(),
                    loss_shadow.item(), in_psnr, out_psnr, time.time()-st))
                logging.info("epoch:%d [%d/%d] lr:%.7f loss:%.5f in:%.2f out:%.2f" % (
                    epoch, i+1, len(train_loader), optimizerG.param_groups[0]["lr"],
                    train_meters['loss'], in_psnr, out_psnr))
                st = time.time()

                if args.SAVE_Inter_Results:
                    save_path = SAVE_Inter_Results_PATH + str(running_results['iter_nums']) + '.jpg'
                    save_imgs_for_visual(save_path, inputs, labels, outputs)

        # eval
        running_results['max_psnr_val'] = test(
            net=net, eval_loader=eval_loader, epoch=epoch,
            max_psnr_val=running_results['max_psnr_val'], Dname='val')

        torch.save(net.state_dict(), SAVE_PATH + 'latest_model.pth')
