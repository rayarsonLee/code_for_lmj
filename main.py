import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import cv2
from typing import List
from torch.nn import functional as F


def wav2spectrogram(audio_filepath):
    """
    将wav文件转化为音频谱图
    :param audio_filepath: 音频文件路径
    :return: （梅尔图片，样本率）
    """
    waveform, sample_rate = librosa.load(audio_filepath)
    # 将音频信号转换为梅尔频谱
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
    # # 将梅尔频谱转换为对数刻度
    # log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec, sample_rate


def display_spectrogram(audio_filepath):
    """
    将wav文件转化为音频谱图&打印图片
    :param audio_filepath: 音频文件路径
    :return:
    """
    spectrogram, sample_rate = wav2spectrogram(audio_filepath)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()


def get_all_spectrogram(audio_dir):
    """
    获取所有梅尔图谱
    :param audio_dir: 所有音频的根目录
    :return: 多个音频对应梅尔图谱的列表
    """
    audio_files = os.listdir(audio_dir)
    audio_files.sort()
    spectrogram_list = []
    for audio_file in audio_files:
        spectrogram, sample_rate = wav2spectrogram(audio_dir + audio_file)
        spectrogram_list.append(spectrogram)
    return spectrogram_list


def load_image(image_filepath):
    """
    加载图片
    :param image_filepath: 图片路径
    :return: 图片
    """
    _image = Image.open(image_filepath)
    return _image


def get_video_tensor(video_dir):
    """
    计算视频的tensor
    :param video_dir: 视频文件夹路径
    :return: 多张图片组成的tensor
    """
    video_files = os.listdir(video_dir)
    video_files.sort()
    concat_tensor = None
    image_type = '.jpg'
    # 遍历图片文件列表
    for image_file in video_files:
        # 加载图片
        image_path = os.path.join(video_dir, image_file)
        image = load_image(image_path)
        image_tensor = transforms.ToTensor()(image)
        if concat_tensor is None:
            concat_tensor = image_tensor
        else:
            concat_tensor = torch.cat((concat_tensor, image_tensor), dim=0)
    return concat_tensor


def get_audio_tensor(audio_dir):
    """
    获取音频tensor
    :param audio_dir: 音频文件目录
    :return: 音频tensor
    """
    audio_tensor = wav2spectrogram(audio_dir)[0]
    audio_tensor = torch.from_numpy(audio_tensor)
    return audio_tensor


def get_dataset(video_root_dir, audio_root_dir):
    """
    获取数据集
    :param video_root_dir: 视频根目录
    :param audio_root_dir: 音频根目录
    :return: tensor列表，0维视频tensor，1维音频tensor
    """
    video_dir_list = os.listdir(video_root_dir)
    audio_file_list = os.listdir(audio_root_dir)
    video_dir_list.sort()
    audio_file_list.sort()
    dataset = []
    for index in range(len(video_dir_list)):
        video_tensor = get_video_tensor(video_root_dir + video_dir_list[index])
        audio_tensor = get_audio_tensor(audio_root_dir + audio_file_list[index])
        sample = [video_tensor, audio_tensor]
        dataset.append(sample)
    return dataset


def pad_dataset(dataset):
    """
    填充数据集到相同维度
    :param dataset: 原始数据集，各个维度不同
    :return: 填充后的数据集
    """
    dataset_video = []
    dataset_audio = []
    for i in range(len(dataset)):
        dataset_video.append(dataset[i][0])
    for i in range(len(dataset)):
        dataset_audio.append(dataset[i][1])

    max_dataset_video_dim0 = max(t.shape[0] for t in dataset_video)
    max_dataset_audio_dim0 = max(t.shape[1] for t in dataset_audio)

    padded_dataset_video = []
    padded_dataset_audio = []
    for tensor in dataset_video:
        padded_tensor = torch.zeros((max_dataset_video_dim0, 256, 256), dtype=tensor.dtype)
        padded_tensor[:tensor.shape[0], :, :] = tensor
        padded_dataset_video.append(padded_tensor)

    for tensor in dataset_audio:
        padded_tensor = torch.zeros((128, max_dataset_audio_dim0), dtype=tensor.dtype)
        padded_tensor[:, :tensor.shape[1]] = tensor
        padded_dataset_audio.append(padded_tensor)

    padded_dataset = []
    for i in range(len(padded_dataset_video)):
        padded_dataset.append([padded_dataset_video[i], padded_dataset_audio[i]])
    return padded_dataset


def get_padded_dataset(video_root_dir, audio_root_dir):
    """
    获取填充后的dataset
    :param video_root_dir: 视频根目录
    :param audio_root_dir: 音频根目录
    :return: 填充后的dataset
    """
    dataset = get_dataset(video_root_dir, audio_root_dir)
    dataset = pad_dataset(dataset)
    return dataset


def get_dataloader(video_root_dir, audio_root_dir):
    dataset = get_padded_dataset(video_root_dir, audio_root_dir)
    dataset_video, dataset_audio = dataset[0], dataset[1]
    dataset = MyDataset(dataset_video, dataset_audio)
    dataloader_params = {
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 4
    }
    dataloader = DataLoader(dataset, **dataloader_params)
    return dataloader


class MyDataset(Dataset):
    """
    自定义数据集类
    """

    def __init__(self, dataset_video, dataset_audio):
        self.dataset_video = dataset_video
        self.dataset_audio = dataset_audio
        assert len(dataset_video) == len(dataset_audio)

    def __len__(self):
        return len(self.dataset_video)

    def __getitem__(self, index):
        return self.dataset_video[index], self.dataset_audio[index]


video_root_dir = 'example/video/train/'
audio_root_dir = 'example/wav/train/'
dataset = get_padded_dataset(video_root_dir, audio_root_dir)
for video, audio in dataset:
    print(video.shape, audio.shape)
