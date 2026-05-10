
import argparse, os, time, random, logging, sys, copy
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())

from networks.diffusion_shadow import DiffusionShadowGenerator


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShadowMaskDataset(Dataset):
    """Load shadow masks for diffusion training.

    Two modes:
      1. mask_dir — load pre-computed single-channel mask images.
      2. shadow_dir + clean_dir — extract masks from paired images.
         mask = clamp(1 - mean(shadow / (clean + eps)), 0, 1)
    """
    EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    def __init__(self, mask_dir=None, shadow_dir=None, clean_dir=None, crop_size=256):
        self.crop_size = crop_size
        if mask_dir:
            self.mode = 'mask'
            self.paths = sorted([
                os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                if os.path.splitext(f)[1].lower() in self.EXTS])
        elif shadow_dir and clean_dir:
            self.mode = 'pair'
            files = sorted([
                f for f in os.listdir(shadow_dir)
                if os.path.splitext(f)[1].lower() in self.EXTS])
            self.shadow_paths = [os.path.join(shadow_dir, f) for f in files]
            self.clean_paths = [os.path.join(clean_dir, f) for f in files]
        else:
            raise ValueError('Provide either --mask_dir or both --shadow_dir and --clean_dir')

    def __len__(self):
        return len(self.paths) if self.mode == 'mask' else len(self.shadow_paths)

    def _random_crop_flip(self, *imgs):
        w, h = imgs[0].size
        cs = self.crop_size
        x = random.randint(0, max(w - cs, 0))
        y = random.randint(0, max(h - cs, 0))
        crops = [img.crop((x, y, x + cs, y + cs)) for img in imgs]
        if random.random() > 0.5:
            crops = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in crops]
        if random.random() > 0.5:
            crops = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in crops]
        return crops

    def __getitem__(self, idx):
        to_tensor = transforms.ToTensor()

        if self.mode == 'mask':
            img = Image.open(self.paths[idx]).convert('L')
            [img] = self._random_crop_flip(img)
            return to_tensor(img)
        else:
            shadow = Image.open(self.shadow_paths[idx]).convert('RGB')
            clean = Image.open(self.clean_paths[idx]).convert('RGB')
            shadow, clean = self._random_crop_flip(shadow, clean)
            shadow_t = to_tensor(shadow)
            clean_t = to_tensor(clean)
            ratio = shadow_t / (clean_t + 1e-6)
            mask = 1.0 - ratio.mean(dim=0, keepdim=True)
            return torch.clamp(mask, 0., 1.)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()

# data
parser.add_argument('--mask_dir', type=str, default='')
parser.add_argument('--shadow_dir', type=str, default='')
parser.add_argument('--clean_dir', type=str, default='')
parser.add_argument('--crop_size', type=int, default=256)

# training
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--T_period', type=int, default=50)
parser.add_argument('--ema_decay', type=float, default=0.9999)
parser.add_argument('--grad_accum_steps', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--vis_every', type=int, default=20)
parser.add_argument('--save_every', type=int, default=50)

# model
parser.add_argument('--width', type=int, default=32)
parser.add_argument('--middle_blk_num', type=int, default=1)
parser.add_argument('--enc_blks', nargs='+', type=int, default=[1, 1, 1, 4])
parser.add_argument('--dec_blks', nargs='+', type=int, default=[1, 1, 1, 1])
parser.add_argument('--kernel_size', type=int, default=3)

# SDE
parser.add_argument('--beta_min', type=float, default=0.1)
parser.add_argument('--beta_max', type=float, default=20.0)
parser.add_argument('--N', type=int, default=1000)

# save
parser.add_argument('--save_path', type=str, default='./ckpt/')
parser.add_argument('--experiment_name', type=str, default='train_diffusion')
parser.add_argument('--writer_dir', type=str, default='./tf-logs/')

# resume
parser.add_argument('--resume', type=str, default='')

args = parser.parse_args()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualise_samples(generator, epoch, save_dir, num_samples=8, size=128, num_steps=100):
    generator.eval()
    masks_sde = generator.sample((num_samples, 1, size, size), device,
                                 num_steps=num_steps, method='sde')
    masks_ode = generator.sample((num_samples, 1, size, size), device,
                                 num_steps=num_steps, method='ode')
    torchvision.utils.save_image(masks_sde,
        os.path.join(save_dir, f'samples_sde_ep{epoch:04d}.png'), nrow=4, padding=2)
    torchvision.utils.save_image(masks_ode,
        os.path.join(save_dir, f'samples_ode_ep{epoch:04d}.png'), nrow=4, padding=2)
    generator.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.writer_dir, exist_ok=True)
    vis_dir = os.path.join(args.save_path, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    writer = SummaryWriter(args.writer_dir + args.experiment_name)

    logging.basicConfig(
        filename=os.path.join(args.save_path, args.experiment_name + '.log'),
        level=logging.INFO)
    for k, v in vars(args).items():
        logging.info(f'{k}: {v}')

    # dataset
    if args.mask_dir:
        dataset = ShadowMaskDataset(mask_dir=args.mask_dir, crop_size=args.crop_size)
    else:
        dataset = ShadowMaskDataset(shadow_dir=args.shadow_dir, clean_dir=args.clean_dir,
                                    crop_size=args.crop_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, drop_last=True)
    print(f'dataset: {len(dataset)} samples, {len(dataloader)} batches')

    # model
    generator = DiffusionShadowGenerator(
        beta_min=args.beta_min, beta_max=args.beta_max, N=args.N,
        width=args.width, middle_blk_num=args.middle_blk_num,
        enc_blk_nums=args.enc_blks, dec_blk_nums=args.dec_blks,
        kernel_size=args.kernel_size,
    ).to(device)
    print(f'#parameters: {sum(p.numel() for p in generator.model.parameters()) / 1e6:.2f}M')

    # EMA
    ema = EMA(generator, decay=args.ema_decay)

    # optimizer
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_period, T_mult=1)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        generator.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        ema.load_state_dict(ckpt['ema'])
        start_epoch = ckpt['epoch'] + 1
        print(f'resumed from epoch {start_epoch}')

    # training
    optimizer.zero_grad()
    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        generator.train()
        st = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for i, masks in enumerate(dataloader):
            masks = masks.to(device)
            loss = generator.compute_loss(masks)
            loss = loss / args.grad_accum_steps
            loss.backward()

            if (i + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(generator)

            epoch_loss += loss.item() * args.grad_accum_steps
            num_batches += 1

            if (i + 1) % args.print_freq == 0:
                avg_loss = epoch_loss / num_batches
                print(f"epoch:{epoch} [{i+1}/{len(dataloader)}] lr:{optimizer.param_groups[0]['lr']:.7f} "
                      f"loss:{avg_loss:.5f} t:{time.time()-st:.1f}s")
                logging.info(f"epoch:{epoch} [{i+1}/{len(dataloader)}] loss:{avg_loss:.5f}")
                st = time.time()

        avg_loss = epoch_loss / max(num_batches, 1)
        writer.add_scalar('loss', avg_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # visualise
        if (epoch + 1) % args.vis_every == 0:
            ema.apply_shadow(generator)
            visualise_samples(generator, epoch, vis_dir, num_samples=8,
                              size=args.crop_size, num_steps=100)
            ema.restore(generator)
            print(f'  -> saved sample visualisation at epoch {epoch}')

        # save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            torch.save({
                'model': generator.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': vars(args),
            }, os.path.join(args.save_path, 'latest_diffusion.pth'))

            # save EMA weights as the deployable model
            ema.apply_shadow(generator)
            torch.save(generator.state_dict(),
                       os.path.join(args.save_path, 'diffusion_shadow.pth'))
            ema.restore(generator)

            print(f'  -> checkpoint saved at epoch {epoch}')
            logging.info(f'checkpoint saved at epoch {epoch}')

    print('training complete.')
