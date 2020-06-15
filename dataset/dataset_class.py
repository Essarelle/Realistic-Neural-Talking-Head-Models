import glob

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


def draw_landmark(landmark, canvas=None, size=None):
    if canvas is None:
        canvas = (np.ones(size) * 255).astype(np.uint8)

    colors = [
        (0, 128, 0),
        (255, 165, 0),
        (255, 165, 0),
        (255, 0, 0),
        (255, 0, 0),
        (0, 0, 255),
        (0, 0, 255),
        (128, 0, 128),
        (255, 192, 203),
    ]

    chin = landmark[0:17]
    left_brow = landmark[17:22]
    right_brow = landmark[22:27]
    left_eye = landmark[36:42]
    left_eye = np.concatenate((left_eye, [landmark[36]]))
    right_eye = landmark[42:48]
    right_eye = np.concatenate((right_eye, [landmark[42]]))
    nose1 = landmark[27:31]
    nose2 = landmark[31:36]
    mouth = landmark[48:60]
    mouth = np.concatenate((mouth, [landmark[48]]))
    mouth_internal = landmark[60:68]
    mouth_internal = np.concatenate((mouth_internal, [landmark[60]]))
    lines = np.array([
        chin, left_brow, right_brow,
        left_eye, right_eye, nose1, nose2,
        mouth, mouth_internal
    ])
    for i, line in enumerate(lines):
        cur_color = colors[i]
        cv2.polylines(
            canvas,
            np.int32([line]), False,
            cur_color, thickness=2, lineType=cv2.LINE_AA
        )

    return canvas


class PreprocessDataset(Dataset):
    def __init__(self, K, path_to_preprocess, path_to_Wi):
        self.K = K
        self.path_to_preprocess = path_to_preprocess
        self.path_to_Wi = path_to_Wi

        self.video_dirs = glob.glob(os.path.join(path_to_preprocess, '*/*'))
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
        return len(self.video_dirs)

    def __getitem__(self, idx):
        vid_idx = idx
        video_dir = self.video_dirs[vid_idx]
        lm_path = os.path.join(video_dir, 'landmarks.npy')
        jpg_paths = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))
        if os.path.exists(lm_path):
            all_landmarks = np.load(lm_path)

        while not os.path.exists(lm_path) or len(all_landmarks) != len(jpg_paths):
            vid_idx = torch.randint(low=0, high=len(self.video_dirs), size=(1,))[0].item()
            video_dir = self.video_dirs[vid_idx]
            lm_path = os.path.join(video_dir, 'landmarks.npy')
            if not os.path.exists(lm_path):
                continue
            jpg_paths = glob.glob(os.path.join(video_dir, '*.jpg'))
            all_landmarks = np.load(lm_path)
            if len(all_landmarks) != len(jpg_paths):
                continue

        if len(jpg_paths) != len(all_landmarks):
            print('DELETE')
            print(lm_path)

        # Select K paths
        random_indices = np.random.randint(0, len(jpg_paths), size=(self.K,))
        paths = np.array(jpg_paths)[random_indices]
        landmarks = all_landmarks[random_indices]

        frame_mark = []
        for i, path in enumerate(paths):
            frame = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            lmark = draw_landmark(landmarks[i], size=frame.shape)
            # cv2.imshow('img', lmark)
            # cv2.waitKey(0)
            # exit()
            frame_mark.append((frame, lmark))

        frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype=torch.float)  # K,2,224,224,3
        frame_mark = frame_mark.permute([0, 1, 4, 2, 3]) / 255.  # K,2,3,224,224

        g_idx = torch.randint(low=0, high=self.K, size=(1, 1))
        x = frame_mark[g_idx, 0].squeeze()
        g_y = frame_mark[g_idx, 1].squeeze()

        return frame_mark, x, g_y, vid_idx, self.W_i[:, vid_idx].unsqueeze(1)


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
