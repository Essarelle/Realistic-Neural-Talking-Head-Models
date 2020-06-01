"""Main"""
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from matplotlib import pyplot as plt
import os

from dataset.dataset_class import VidDataSet
from dataset.video_extraction_conversion import *
from loss.loss_discriminator import *
from loss.loss_generator import *
from network.blocks import *
from network.model import *
from tqdm import tqdm
import tensorboardX

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint')
parser.add_argument('--backup')
parser.add_argument('--wi')
parser.add_argument('--batch-size')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--preprocessed')
parser.add_argument('--train-dir', default='train')
parser.add_argument('--vggface-dir', default='.')
parser.add_argument('--data-dir', default='../image2image/ds_fa_vox')
parser.add_argument('--frame-shape', default=224)

args = parser.parse_args()

"""Create dataset and net"""
device = torch.device("cuda:0")
cpu = torch.device("cpu")
path_to_chkpt = os.path.join(args.train_dir, 'model_weights.tar')
dataset = VidDataSet(K=8, path_to_mp4=args.data_dir, device=device)

dataLoader = DataLoader(dataset, batch_size=1, shuffle=True)

G = Generator(224).to(device)
E = Embedder(224).to(device)
D = Discriminator(dataset.__len__()).to(device)

G.train()
E.train()
D.train()

optimizerG = optim.Adam(params=list(E.parameters()) + list(G.parameters()), lr=5e-5)
optimizerD = optim.Adam(params=D.parameters(), lr=2e-4)

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
    os.makedirs(os.path.dirname(path_to_chkpt))
    print('Initiating new checkpoint...')
    torch.save({
        'epoch': epoch,
        'lossesG': lossesG,
        'lossesD': lossesD,
        'E_state_dict': E.state_dict(),
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'num_vid': dataset.__len__(),
        'i_batch': i_batch
    }, path_to_chkpt)
    print('...Done')

"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
E.load_state_dict(checkpoint['E_state_dict'])
G.load_state_dict(checkpoint['G_state_dict'])
D.load_state_dict(checkpoint['D_state_dict'])
optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
epochCurrent = checkpoint['epoch']
lossesG = checkpoint['lossesG']
lossesD = checkpoint['lossesD']
num_vid = checkpoint['num_vid']
i_batch_current = checkpoint['i_batch'] + 1

G.train()
E.train()
D.train()

"""Training"""
batch_start = datetime.now()
writer = tensorboardX.SummaryWriter(args.train_dir)
num_batches = 0

for epoch in range(epochCurrent, num_epochs):
    for i_batch, (f_lm, x, g_y, i) in enumerate(dataLoader, start=i_batch_current):
        if i_batch > len(dataLoader):
            i_batch_current = 0
            break
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
            r, D_res_list = D(x, g_y, i)

            lossG = criterionG(x, x_hat, r_hat, D_res_list, D_hat_res_list, e_vectors, D.W_i, i)
            lossDfake = criterionDfake(r_hat)
            lossDreal = criterionDreal(r)

            loss = lossDreal + lossDfake + lossG
            loss.backward(retain_graph=False)
            optimizerG.step()
            optimizerD.step()

            # train D again
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            x_hat.detach_()
            r_hat, D_hat_res_list = D(x_hat, g_y, i)
            r, D_res_list = D(x, g_y, i)

            lossDfake = criterionDfake(r_hat)
            lossDreal = criterionDreal(r)

            lossD = lossDreal + lossDfake
            lossD.backward(retain_graph=False)
            optimizerD.step()

        step = epoch * num_batches + i_batch
        # Output training stats
        if step % 20 == 0:
            batch_end = datetime.now()
            avg_time = (batch_end - batch_start) / 10
            print('\n\navg batch time for batch size of', x.shape[0], ':', avg_time)

            batch_start = datetime.now()

            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(y)): %.4f'
                  % (epoch, num_epochs, i_batch, len(dataLoader),
                     lossD.item(), lossG.item(), r.mean(), r_hat.mean()))

            out = x_hat.transpose(1, 3)[0]
            for img_no in range(1, x_hat.shape[0]):
                out = torch.cat((out, x_hat.transpose(1, 3)[img_no]), dim=1)
            out1 = out.type(torch.int32).to(cpu).numpy()

            out = x.transpose(1, 3)[0]
            for img_no in range(1, x.shape[0]):
                out = torch.cat((out, x.transpose(1, 3)[img_no]), dim=1)
            out2 = out.type(torch.int32).to(cpu).numpy()

            out = g_y.transpose(1, 3)[0]
            for img_no in range(1, g_y.shape[0]):
                out = torch.cat((out, g_y.transpose(1, 3)[img_no]), dim=1)
            out3 = out.type(torch.int32).to(cpu).numpy()

            writer.add_image(
                'Result', np.hstack((out1, out2, out3)).astype(np.uint8),
                global_step=step,
                dataformats='HWC'
            )
            writer.add_scalar('loss_g', lossG.item(), global_step=step)
            writer.add_scalar('loss_d', lossD.item(), global_step=step)
            # cv2.imshow('Result', np.hstack((out1, out2, out3)).astype(np.uint8)[:, :, ::-1])
            # cv2.waitKey(1)

    num_batches = i_batch
    print('Saving latest...')
    torch.save({
        'epoch': epoch,
        'lossesG': lossesG,
        'lossesD': lossesD,
        'E_state_dict': E.state_dict(),
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'num_vid': dataset.__len__(),
        'i_batch': i_batch
    }, path_to_chkpt)
    print('...Done saving latest')
