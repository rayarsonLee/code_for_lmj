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


def wav2spectrogram(audio_filepath):
    """
    将wav文件转化为音频谱图
    :param audio_filepath: 音频文件路径
    :return: （梅尔图片，样本率）
    """
    waveform, sample_rate = librosa.load(audio_filepath)
    # 将音频信号转换为梅尔频谱
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
    # 将梅尔频谱转换为对数刻度
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec, sample_rate


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


def get_dataset(video_root_dir, audio_root_dir):
    video_dir_list = os.listdir(video_root_dir)
    audio_file_list = os.listdir(audio_root_dir)
    video_dir_list.sort()
    audio_file_list.sort()
    dataset = []
    for index in range(len(video_dir_list)):
        video_tensor = get_video_tensor(video_root_dir + video_dir_list[index])
        audio_tensor = wav2spectrogram(audio_root_dir + audio_file_list[index])
        sample = {'video_tensor': video_tensor, 'audio_tensor': audio_tensor}
        dataset.append(sample)
    return dataset


video_root_dir = 'example/video/train/'
audio_root_dir = 'example/wav/train/'
dataset = get_dataset(video_root_dir, audio_root_dir)
print(dataset)
