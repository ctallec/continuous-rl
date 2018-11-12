"""Video from npz"""
import argparse
from moviepy.editor import ImageSequenceClip
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--npz_file', type=str)
parser.add_argument('--fps', type=int)
parser.add_argument('--output_file', type=str)
args = parser.parse_args()

imgs = np.load(args.npz_file)['arr_0']
imgs = [img.squeeze() for img in np.split(imgs, imgs.shape[0])]
clip = ImageSequenceClip(imgs, fps=args.fps)
if args.output_file.endswith('.gif'):
    clip.write_gif(args.output_file)
if args.output_file.endswith('.mp4'):
    clip.write_videofile(args.output_file)
