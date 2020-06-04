import argparse

import torch
import cv2
import face_alignment
from matplotlib import pyplot as plt
import numpy as np

from loss.loss_discriminator import *
from loss.loss_generator import *
from network.blocks import *
from network.model import *
from dataset import video_extraction_conversion

# from webcam_demo.webcam_extraction_conversion import *

"""Init"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--embedding')
    parser.add_argument('--video')
    parser.add_argument('--output')

    return parser.parse_args()


# Paths
args = parse_args()
path_to_model_weights = args.model
path_to_embedding = args.embedding

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
cpu = torch.device("cpu")

checkpoint = torch.load(path_to_model_weights, map_location=cpu)
e_hat = torch.load(path_to_embedding, map_location=cpu)
e_hat = e_hat['e_hat'].to(device)

G = Generator(256, finetuning=True, e_finetuning=e_hat)
G.eval()

"""Training Init"""
G.load_state_dict(checkpoint['G_state_dict'])
G.to(device)
G.finetuning_init()

"""Main"""
print('PRESS Q TO EXIT')
cap = cv2.VideoCapture(args.video if args.video else 0)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device.type)

if args.output:
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_format = video.get(cv2.CAP_PROP_FORMAT)
    video_writer = cv2.VideoWriter(
        args.output, fourcc, fps,
        frameSize=(256 * 3, 256)
    )

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_list = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)]
        l = video_extraction_conversion.generate_landmarks(frames_list, face_aligner=fa)
        x, g_y = l[0][0], l[0][1]
        x = torch.from_numpy(x.transpose([2, 0, 1])).type(dtype=torch.float)
        g_y = torch.from_numpy(g_y.transpose([2, 0, 1])).type(dtype=torch.float)
        if use_cuda:
            x, g_y = x.cuda(), g_y.cuda()

        g_y = g_y.unsqueeze(0) / 255
        x = x.unsqueeze(0) / 255

        x_hat = G(g_y, e_hat)

        out1 = x[0].to(cpu).numpy().transpose([1, 2, 0])
        out2 = g_y[0].to(cpu).numpy().transpose([1, 2, 0])
        out3 = x_hat[0].to(cpu).numpy().transpose([1, 2, 0])

        result = cv2.cvtColor(np.hstack((out1, out2, out3)), cv2.COLOR_BGR2RGB)
        cv2.imshow('Result', result)
        if args.output:
            video_writer.write(result)

        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
if args.output:
    video_writer.release()
