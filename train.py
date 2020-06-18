"""Main"""
import argparse
import os
import time
from datetime import datetime

import matplotlib
from matplotlib import pyplot as plt
import tensorboardX
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

plt.ion()

from dataset.dataset_class import PreprocessDataset
from dataset.dataset_class import VidDataSet
from dataset.video_extraction_conversion import *
from loss.loss_discriminator import *
from loss.loss_generator import *
from network.model import *

parser = argparse.ArgumentParser()
parser.add_argument('-k', default=8, type=int)
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--preprocessed')
parser.add_argument('--save-checkpoint', type=int, default=1000)
parser.add_argument('--train-dir', default='train')
parser.add_argument('--vggface-dir', default='.')
parser.add_argument('--data-dir', default='../image2image/ds_fa_vox')
parser.add_argument('--frame-shape', default=256, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--fa-device', default='cuda:0')

args = parser.parse_args()


def print_fun(s):
    print(s)
    sys.stdout.flush()


"""Create dataset and net"""
display_training = False
matplotlib.use('agg')
device = torch.device("cuda")
cpu = torch.device("cpu")
batch_size = args.batch_size
frame_shape = args.frame_shape
path_to_Wi = os.path.join(args.train_dir, 'wi_weights')
K = args.k
if not os.path.exists(path_to_Wi):
    os.makedirs(path_to_Wi)


if args.preprocessed:
    dataset = PreprocessDataset(K=K, path_to_preprocess=args.preprocessed, path_to_Wi=path_to_Wi, frame_shape=frame_shape)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
else:
    dataset = VidDataSet(
        K=K, path_to_mp4=args.data_dir,
        device=args.fa_device, path_to_wi=path_to_Wi, size=frame_shape
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers if 'cuda' not in args.fa_device else 0,
    )

path_to_chkpt = os.path.join(args.train_dir, 'model_weights.tar')

G = nn.DataParallel(Generator(frame_shape).to(device))
E = nn.DataParallel(Embedder(frame_shape).to(device))
D = nn.DataParallel(Discriminator(dataset.__len__(), path_to_Wi, args.batch_size).to(device))

G.train()
E.train()
D.train()

optimizerG = optim.Adam(
    params=list(E.parameters()) + list(G.parameters()),
    lr=5e-5,
    amsgrad=False
)
optimizerD = optim.Adam(
    params=D.parameters(),
    lr=2e-4,
    amsgrad=False)

"""Criterion"""
criterionG = LossG(
    VGGFace_body_path=os.path.join(args.vggface_dir, 'Pytorch_VGGFACE_IR.py'),
    VGGFace_weight_path=os.path.join(args.vggface_dir, 'Pytorch_VGGFACE.pth'),
    device=device
)
criterionDreal = LossDSCreal()
criterionDfake = LossDSCfake()

"""Training init"""
epochCurrent = epoch = i_batch = 0
lossesG = []
lossesD = []
i_batch_current = 0

num_epochs = args.epochs

# initiate checkpoint if inexistant
if not os.path.isfile(path_to_chkpt):
    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform(m.weight)


    G.apply(init_weights)
    D.apply(init_weights)
    E.apply(init_weights)

    print_fun('Initiating new checkpoint...')
    torch.save({
        'epoch': epoch,
        'lossesG': lossesG,
        'lossesD': lossesD,
        'E_state_dict': E.module.state_dict(),
        'G_state_dict': G.module.state_dict(),
        'D_state_dict': D.module.state_dict(),
        'num_vid': dataset.__len__(),
        'i_batch': i_batch,
        'optimizerG': optimizerG.state_dict(),
        'optimizerD': optimizerD.state_dict()
    }, path_to_chkpt)
    print_fun('...Done')

"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
E.module.load_state_dict(checkpoint['E_state_dict'])
G.module.load_state_dict(checkpoint['G_state_dict'], strict=False)
D.module.load_state_dict(checkpoint['D_state_dict'])
epochCurrent = checkpoint['epoch']
lossesG = checkpoint['lossesG']
lossesD = checkpoint['lossesD']
num_vid = checkpoint['num_vid']
optimizerG.load_state_dict(checkpoint['optimizerG'])
optimizerD.load_state_dict(checkpoint['optimizerD'])
prev_step = checkpoint['i_batch']

G.train()
E.train()
D.train()

"""Training"""
batch_start = datetime.now()

writer = tensorboardX.SummaryWriter(args.train_dir)
num_batches = len(dataset) / args.batch_size
log_step = int(round(0.005 * num_batches + 20))
log_epoch = 1
if num_batches <= 10:
    log_step = 100
    log_epoch = log_step
save_checkpoint = args.save_checkpoint
print_fun(f"Will log each {log_step} step.")
print_fun(f"Will save checkpoint each {save_checkpoint} step.")
if prev_step != 0:
    print_fun(f"Starting at {prev_step} step.")


for epoch in range(0, num_epochs):
    # if epochCurrent > epoch:
    #     pbar = tqdm(dataLoader, leave=True, initial=epoch, disable=None)
    #     continue
    # Reset random generator
    np.random.seed(int(time.time()))
    for i_batch, (f_lm, x, g_y, i, W_i) in enumerate(data_loader):

        f_lm = f_lm.to(device)
        x = x.to(device)
        g_y = g_y.to(device)
        W_i = W_i.squeeze(-1).transpose(0, 1).to(device).requires_grad_()

        D.module.load_W_i(W_i)

        with torch.autograd.enable_grad():
            # zero the parameter gradients
            optimizerG.zero_grad()
            optimizerD.zero_grad()

            # forward
            # Calculate average encoding vector for video
            f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2],
                                     f_lm.shape[-1])  # BxK,2,3,224,224

            e_vectors = E(f_lm_compact[:, 0, :, :, :], f_lm_compact[:, 1, :, :, :])  # BxK,512,1
            e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1)  # B,K,512,1
            e_hat = e_vectors.mean(dim=1)

            # train G and D
            x_hat = G(g_y, e_hat)
            r_hat, D_hat_res_list = D(x_hat, g_y, i)
            with torch.no_grad():
                r, D_res_list = D(x, g_y, i)
            """####################################################################################################################################################
            r, D_res_list = D(x, g_y, i)"""

            lossG = criterionG(
                x, x_hat, r_hat, D_res_list, D_hat_res_list, e_vectors, D.module.W_i[:, :batch_size], i
            )

            """####################################################################################################################################################
            lossD = criterionDfake(r_hat) + criterionDreal(r)
            loss = lossG + lossD
            loss.backward(retain_graph=False)
            optimizerG.step()
            optimizerD.step()"""

            lossG.backward(retain_graph=False)
            optimizerG.step()
            # optimizerD.step()

        with torch.autograd.enable_grad():
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            x_hat.detach_().requires_grad_()
            r_hat, D_hat_res_list = D(x_hat, g_y, i)
            lossDfake = criterionDfake(r_hat)

            r, D_res_list = D(x, g_y, i)
            lossDreal = criterionDreal(r)

            lossD = lossDfake + lossDreal
            lossD.backward(retain_graph=False)
            optimizerD.step()

            optimizerD.zero_grad()
            r_hat, D_hat_res_list = D(x_hat, g_y, i)
            lossDfake = criterionDfake(r_hat)

            r, D_res_list = D(x, g_y, i)
            lossDreal = criterionDreal(r)

            lossD = lossDfake + lossDreal
            lossD.backward(retain_graph=False)
            optimizerD.step()

        for enum, idx in enumerate(i):
            dataset.W_i[:, idx.item()] = D.module.W_i[:, enum]
            # torch.save({'W_i': D.module.W_i[:, enum].unsqueeze(-1)},
            #            path_to_Wi + '/W_' + str(idx.item()) + '/W_' + str(idx.item()) + '.tar')

        step = epoch * num_batches + i_batch + prev_step
        # Output training stats
        if step % log_step == 0:
            out = (x_hat[0] * 255).permute([1, 2, 0])
            out1 = out.type(torch.int32).to(cpu).numpy()

            out = (x[0] * 255).permute([1, 2, 0])
            out2 = out.type(torch.int32).to(cpu).numpy()

            out = (g_y[0] * 255).permute([1, 2, 0])
            out3 = out.type(torch.int32).to(cpu).numpy()
            accuracy = np.sum(np.squeeze((np.abs(out1 - out2) <= 1))) / np.prod(out.shape)
            print_fun(
                'Step %d [%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(y)): %.4f\tMatch: %.3f'
                % (step, epoch, num_epochs, i_batch, len(data_loader),
                   lossD.item(), lossG.item(), r.mean(), r_hat.mean(), accuracy)
            )

            image = np.hstack((out1, out2, out3)).astype(np.uint8).clip(0, 255)
            writer.add_image(
                'Result', image,
                global_step=step,
                dataformats='HWC'
            )
            writer.add_scalar('loss_g', lossG.item(), global_step=step)
            writer.add_scalar('loss_d', lossD.item(), global_step=step)
            writer.add_scalar('match', accuracy, global_step=step)
            writer.flush()

        if step != 0 and step % save_checkpoint == 0:
            print_fun('Saving latest...')
            torch.save({
                'epoch': epoch,
                'lossesG': lossesG,
                'lossesD': lossesD,
                'E_state_dict': E.module.state_dict(),
                'G_state_dict': G.module.state_dict(),
                'D_state_dict': D.module.state_dict(),
                'num_vid': dataset.__len__(),
                'i_batch': step,
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict()
            },
                path_to_chkpt
            )
            dataset.save_w_i()

    if epoch % log_epoch == 0:
        print_fun('Saving latest...')
        torch.save({
            'epoch': epoch,
            'lossesG': lossesG,
            'lossesD': lossesD,
            'E_state_dict': E.module.state_dict(),
            'G_state_dict': G.module.state_dict(),
            'D_state_dict': D.module.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': step,
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict()
        },
            path_to_chkpt
        )
        dataset.save_w_i()
        print_fun('...Done saving latest')
