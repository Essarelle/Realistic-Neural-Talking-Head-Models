import argparse
import glob
import queue
import os
import sys
import threading
import time

import cv2
import face_alignment
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir')
parser.add_argument('--output')

args = parser.parse_args()

path_to_mp4 = args.data_dir
device = torch.device('cuda:0')
saves_dir = args.output

if not os.path.isdir(saves_dir):
    os.mkdir(saves_dir)


def print_fun(s):
    print(s)
    sys.stdout.flush()


def generate_landmarks(frame, face_aligner):
    input = frame
    preds = face_aligner.get_landmarks(input)[0]

    return preds


def process_images(video_dir, lm_queue: queue.Queue):
    videos = sorted([os.path.join(video_dir, v) for v in os.listdir(video_dir)])
    frame_id = -1
    for video_path in videos:
        print_fun(f'Process {video_path}...')
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_id += 1
            lm_queue.put((rgb, video_path, frame_id))

        cap.release()

    lm_queue.put(os.path.join(video_dir, 'dummy'))


class LandmarksQueue(object):
    def __init__(self, q: queue.Queue, root_dir):
        self.landmarks = []
        self.q = q
        self.root_dir = root_dir
        self.save_q = queue.Queue(maxsize=q.maxsize)
        self.face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:0')
        self.save_frame_id = 0
        self.lock = threading.Lock()
        self.threads = []

    def start_process(self):
        t = threading.Thread(target=self.process_lm, daemon=True)
        t.start()
        self.threads.append(t)
        # t = threading.Thread(target=self.process_lm, daemon=True)
        # t.start()
        # self.threads.append(t)
        t = threading.Thread(target=self.process_save, daemon=True)
        t.start()
        self.threads.append(t)

    def get_new_video_dir(self, video_path):
        splitted = video_path.split('/')
        video_id = splitted[-2]
        person_id = splitted[-3]
        new_dir = os.path.join(self.root_dir, person_id, video_id)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        return new_dir

    def process_lm(self):
        self.start = time.time()
        while True:
            item = self.q.get()
            if isinstance(item, str):
                if item == 'stop':
                    break
                else:
                    # flush landmarks
                    if len(self.landmarks) == 0:
                        continue

                    with self.lock:
                        print_fun(f'Processed {len(self.landmarks)} landmarks: {time.time() - self.start}')
                        self.start = time.time()

                        dirname = self.get_new_video_dir(item)
                        save_path = os.path.join(dirname, 'landmarks.npy')
                        print_fun(f'save {save_path}')
                        np.save(save_path, np.stack(self.landmarks))
                        self.save_frame_id = 0
                        self.landmarks = []

            else:
                frame = item[0]
                video_dir = self.get_new_video_dir(item[1])
                try:
                    landmark = generate_landmarks(frame, self.face_aligner)

                    with self.lock:
                        self.landmarks.append(landmark)
                        self.save_q.put((frame, video_dir, self.save_frame_id))
                        self.save_frame_id += 1
                except Exception as e:
                    print_fun(e)

    def process_save(self):
        while True:
            item = self.save_q.get()
            if isinstance(item, str):
                if item == 'stop':
                    break
            else:
                bgr = cv2.cvtColor(item[0], cv2.COLOR_RGB2BGR)
                video_dir = item[1]
                frame_id = item[2]
                cv2.imwrite(os.path.join(video_dir, f'{frame_id:05d}.jpg'), bgr)

    def stop(self):
        self.q.put('stop')
        # self.q.put('stop')
        while not self.q.empty():
            time.sleep(0.1)

        self.save_q.put('stop')
        for t in self.threads:
            t.join(10)


video_paths = glob.glob(os.path.join(path_to_mp4, '**/*'))
lm_queue = queue.Queue(maxsize=200)
landmarks_queue = LandmarksQueue(lm_queue, args.output)
landmarks_queue.start_process()

for video_dir in video_paths:
    process_images(video_dir, lm_queue)

print_fun('Done.')
print_fun('Waiting stop threads...')
landmarks_queue.stop()
print_fun('Done.')

