import os
import cv2
import torch
import numpy as np
import imageio
import torchvision
from einops import rearrange


def mask_to_xyxy_box(mask):
    rows, cols = np.where(mask == 255)
    xmin = min(cols)
    xmax = max(cols) + 1  # 加 1 包含边界像素
    ymin = min(rows)
    ymax = max(rows) + 1  # 加 1 包含边界像素
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, mask.shape[1])
    ymax = min(ymax, mask.shape[0])
    box = [xmin, ymin, xmax, ymax]
    box = [int(x) for x in box]
    return box


def encode_audio(wav2vec, audio_feats, num_frames=129):
    start_ts = [0]
    step_ts = [1]
    audio_feats = wav2vec.encoder(audio_feats.unsqueeze(0)[:, :, :3000], output_hidden_states=True).hidden_states
    audio_feats = torch.stack(audio_feats, dim=2)
    audio_feats = torch.cat([torch.zeros_like(audio_feats[:,:4]), audio_feats], 1)
    
    audio_prompts = []
    for bb in range(1):
        audio_feats_list = []
        for f in range(num_frames):
            cur_t = (start_ts[bb] + f * step_ts[bb]) * 2
            audio_clip = audio_feats[bb:bb+1, cur_t: cur_t+10]
            audio_feats_list.append(audio_clip)
        audio_feats_list = torch.stack(audio_feats_list, 1)
        audio_prompts.append(audio_feats_list)
    audio_prompts = torch.cat(audio_prompts)
    return audio_prompts


def get_audio_feature(feature_extractor, audio_path):
    import librosa
    audio_input, sampling_rate = librosa.load(audio_path, sr=16000)
    assert sampling_rate == 16000

    audio_features = []
    window = 750*640
    for i in range(0, len(audio_input), window):
        audio_feature = feature_extractor(audio_input[i:i+window], 
                                        sampling_rate=sampling_rate, 
                                        return_tensors="pt", 
                                        ).input_features
        audio_features.append(audio_feature)

    audio_features = torch.cat(audio_features, dim=-1)
    return audio_features, len(audio_input) // 640


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8, quality=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x,0,1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps, quality=quality)

def pad_image(crop_img, size, color=(255, 255, 255), resize_ratio=1):
    crop_h, crop_w = crop_img.shape[:2]
    target_w, target_h = size
    scale_h, scale_w = target_h / crop_h, target_w / crop_w
    if scale_w > scale_h:
        resize_h = int(target_h*resize_ratio)
        resize_w = int(crop_w / crop_h * resize_h)
    else:
        resize_w = int(target_w*resize_ratio)
        resize_h = int(crop_h / crop_w * resize_w)
    crop_img = cv2.resize(crop_img, (resize_w, resize_h))
    pad_left = (target_w - resize_w) // 2
    pad_top = (target_h - resize_h) // 2
    pad_right = target_w - resize_w - pad_left
    pad_bottom = target_h - resize_h - pad_top
    crop_img = cv2.copyMakeBorder(crop_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
    return crop_img
