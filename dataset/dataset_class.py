import glob
import time

import torch
from torch.utils.data import Dataset
import face_alignment

from .video_extraction_conversion import *


class VidDataSet(Dataset):
    def __init__(self, K, path_to_mp4, device, path_to_wi, size=256):
        self.K = K
        self.size = size
        self.path_to_Wi = path_to_wi
        self.path_to_mp4 = path_to_mp4
        self.device = device
        self.face_aligner = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D,
            flip_input=False,
            device=device
        )
        self.video_paths = glob.glob(os.path.join(path_to_mp4, '*/*/*.mp4'))
        self.W_i = None
        if self.path_to_Wi is not None:
            if self.W_i is None:
                try:
                    # Load
                    W_i = torch.load(self.path_to_Wi + '/W_' + str(len(self.video_paths)) + '.tar',
                                     map_location='cpu')['W_i'].requires_grad_(False)
                    self.W_i = W_i
                except:
                    # print("\n\nerror loading: ", self.path_to_Wi + '/W_' + str(len(self.video_paths)) + '.tar')
                    w_i = torch.rand(512, len(self))
                    torch.save({'W_i': w_i}, self.path_to_Wi + '/W_' + str(len(self)) + '.tar')
                    self.W_i = w_i

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        vid_idx = idx
        path = self.video_paths[vid_idx]
        ok = False
        while not ok:
            try:
                frame_mark = select_frames(path, self.K)
                frame_mark = generate_landmarks(frame_mark, self.face_aligner, size=self.size)
                ok = True
            except Exception:
                vid_idx = torch.randint(low=0, high=len(self.video_paths), size=(1,))[0].item()
                path = self.video_paths[vid_idx]
        frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype=torch.float)  # K,2,224,224,3
        frame_mark = frame_mark.permute([0, 1, 4, 2, 3]) / 255.  # K,2,3,224,224

        g_idx = torch.randint(low=0, high=self.K, size=(1, 1))
        x = frame_mark[g_idx, 0].squeeze()
        g_y = frame_mark[g_idx, 1].squeeze()

        return frame_mark, x, g_y, vid_idx, self.W_i[:, vid_idx].unsqueeze(1)

    def save_w_i(self):
        torch.save({'W_i': self.W_i}, self.path_to_Wi + '/W_' + str(len(self)) + '.tar')


class PreprocessDataset(Dataset):
    def __init__(self, K, path_to_preprocess, path_to_Wi, frame_shape=224):
        self.K = K
        self.path_to_preprocess = path_to_preprocess
        self.path_to_Wi = path_to_Wi
        self.frame_shape = frame_shape

        self.video_dirs = glob.glob(os.path.join(path_to_preprocess, '*/*'))
        # self.W_i = None
        # if self.path_to_Wi is not None:
        #     if self.W_i is None:
        #         try:
        #             Load
                    # W_i = torch.load(self.path_to_Wi + '/W_' + str(len(self.video_dirs)) + '.tar',
                    #                  map_location='cpu')['W_i'].requires_grad_(False)
                    # self.W_i = W_i
                # except:
                #     print("error loading: ", self.path_to_Wi + '/W_' + str(len(self.video_dirs)) + '.tar')
                #     print("Initializing: ", self.path_to_Wi + '/W_' + str(len(self.video_dirs)) + '.tar')
                #     import sys
                #     sys.stdout.flush()
                #     w_i = torch.rand(512, len(self))
                #     torch.save({'W_i': w_i}, self.path_to_Wi + '/W_' + str(len(self)) + '.tar')
                #     self.W_i = w_i

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        vid_idx = idx
        video_dir = self.video_dirs[vid_idx]
        lm_path = os.path.join(video_dir, 'landmarks.npy')
        jpg_paths = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))
        if os.path.exists(lm_path):
            all_landmarks = np.load(lm_path)

        while not os.path.exists(lm_path) or len(all_landmarks) != len(jpg_paths):
            vid_idx = vid_idx // 2
            video_dir = self.video_dirs[vid_idx]
            lm_path = os.path.join(video_dir, 'landmarks.npy')
            if not os.path.exists(lm_path):
                continue
            jpg_paths = glob.glob(os.path.join(video_dir, '*.jpg'))
            all_landmarks = np.load(lm_path)
            if len(all_landmarks) != len(jpg_paths):
                continue

        # Select K paths
        random_indices = np.random.randint(0, len(jpg_paths), size=(self.K,))
        paths = np.array(jpg_paths)[random_indices]
        landmarks = all_landmarks[random_indices]

        frame_mark = []
        for i, path in enumerate(paths):
            frame = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            cur_landmark = landmarks[i].copy()
            if frame.shape[:2] != (self.frame_shape, self.frame_shape):
                x_factor, y_factor = frame.shape[1] / self.frame_shape, frame.shape[0] / self.frame_shape
                frame = cv2.resize(frame, (self.frame_shape, self.frame_shape), interpolation=cv2.INTER_AREA)
                cur_landmark /= [x_factor, y_factor]
            lmark = draw_landmark(cur_landmark, size=frame.shape)
            # cv2.imshow('img', lmark)
            # cv2.waitKey(0)
            # exit()
            frame_mark.append((frame, lmark))

        frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype=torch.float)  # K,2,224,224,3
        frame_mark = frame_mark.permute([0, 1, 4, 2, 3]) / 255.  # K,2,3,224,224
        frame_mark = frame_mark.requires_grad_(False)

        g_idx = np.random.randint(low=0, high=self.K, size=(1, 1))
        x = frame_mark[g_idx, 0].squeeze()
        g_y = frame_mark[g_idx, 1].squeeze()

        # w_i = self.W_i[:, vid_idx].unsqueeze(1)
        # w_i = w_i.detach()
        return frame_mark, x, g_y, vid_idx, torch.Tensor([])

    def save_w_i(self):
        # torch.save({'W_i': self.W_i}, self.path_to_Wi + '/W_' + str(len(self)) + '.tar')
        pass


class FineTuningImagesDataset(Dataset):
    def __init__(self, path_to_images, device):
        self.path_to_images = path_to_images
        self.device = device
        self.face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
                                                         device='cuda:0')

    def __len__(self):
        return len(os.listdir(self.path_to_images))

    def __getitem__(self, idx):
        frame_mark_images = select_images_frames(self.path_to_images)
        random_idx = torch.randint(low=0, high=len(frame_mark_images), size=(1, 1))
        frame_mark_images = [frame_mark_images[random_idx]]
        frame_mark_images = generate_cropped_landmarks(frame_mark_images, self.face_aligner, pad=50)
        frame_mark_images = torch.from_numpy(np.array(frame_mark_images)).type(dtype=torch.float)  # 1,2,256,256,3
        frame_mark_images = frame_mark_images.transpose(2, 4).to(self.device)  # 1,2,3,256,256

        x = frame_mark_images[0, 0].squeeze() / 255
        g_y = frame_mark_images[0, 1].squeeze() / 255

        return x, g_y


class FineTuningVideoDataset(Dataset):
    def __init__(self, path_to_video, device):
        self.path_to_video = path_to_video
        self.device = device
        self.face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
                                                         device='cuda:0')

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        path = self.path_to_video
        frame_has_face = False
        while not frame_has_face:
            try:
                frame_mark = select_frames(path, 1)
                frame_mark = generate_cropped_landmarks(frame_mark, self.face_aligner, pad=50)
                frame_has_face = True
            except:
                print('No face detected, retrying')
        frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype=torch.float)  # 1,2,256,256,3
        frame_mark = frame_mark.transpose(2, 4).to(self.device)  # 1,2,3,256,256

        x = frame_mark[0, 0].squeeze() / 255
        g_y = frame_mark[0, 1].squeeze() / 255
        return x, g_y
