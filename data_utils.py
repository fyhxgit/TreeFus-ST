import os
import collections
import warnings

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

ASVFile = collections.namedtuple(
    'ASVFile',
    ['speaker_id', 'file_name', 'path', 'sys_id', 'key']
)


class ASVDataset(Dataset):
    """
    Robust ASVspoof dataset loader.

    关键逻辑：
    1. 兼容 ASVspoof2019 LA 标准目录：
       ASVspoof2019_LA_train/flac
       ASVspoof2019_LA_dev/flac
       ASVspoof2019_LA_eval/flac

    2. RawBoost 只用于训练集：
       read raw waveform -> RawBoost -> pad/crop -> torch.Tensor

    3. dev/eval 不做 RawBoost，保证验证和测试干净。
    """

    def __init__(self, database_path=None, protocols_path=None, transform=None,
                 is_train=True, sample_size=None, is_logical=True, feature_name=None,
                 is_eval=False, eval_part=0, rawboost_args=None, year='2019', task='LA'):

        if database_path is None:
            raise ValueError("database_path cannot be None")
        if protocols_path is None:
            raise ValueError("protocols_path cannot be None")

        self.database_path = os.path.abspath(os.path.expanduser(database_path))
        self.protocols_path = os.path.abspath(os.path.expanduser(protocols_path))

        self.year = str(year)
        self.task = str(task).upper()
        self.is_train = bool(is_train)
        self.is_eval = bool(is_eval)
        self.eval_part = eval_part
        self.transform = transform
        self.rawboost_args = rawboost_args

        # 只在训练集启用 RawBoost
        self.is_train_mode = bool(self.is_train and (not self.is_eval) and rawboost_args is not None)

        # RawBoost 函数只导入一次，不要在每个样本里反复 import
        self.rawboost_func = None
        if self.is_train_mode:
            try:
                from RawBoost import process_Rawboost_feature
                self.rawboost_func = process_Rawboost_feature
            except Exception as e:
                raise ImportError(
                    "RawBoost is enabled, but failed to import process_Rawboost_feature "
                    "from RawBoost.py. Please check RawBoost.py is in the same directory "
                    f"or PYTHONPATH. Original error: {e}"
                )

        # ASVspoof2019 LA attack IDs: A07-A50
        self.sysid_dict = {'-': 0}
        for i, k in enumerate(range(7, 51), start=1):
            self.sysid_dict[f'A{k:02d}'] = i
        self.sysid_dict_inv = {v: k for k, v in self.sysid_dict.items()}

        self.split = self._get_split_name()
        self.protocols_fname = self._get_protocol_file()

        print("sysid_dict_inv", self.sysid_dict_inv)
        print("database_path", self.database_path)
        print("protocols_file", self.protocols_fname)
        print("dataset year", self.year)
        print("dataset task", self.task)
        print("dataset split", self.split)
        print("rawboost enabled", self.is_train_mode)

        self.files_meta = self.parse_protocols_file(self.protocols_fname)

        if sample_size is not None:
            self.files_meta = self.files_meta[:sample_size]

        if len(self.files_meta) == 0:
            raise RuntimeError(
                f"No trials loaded from protocol file: {self.protocols_fname}. "
                "Please check protocols_path, year, and task."
            )

        self.length = len(self.files_meta)

        print(f'Loaded {len(self.files_meta)} trials from protocol file.')
        print("first_audio_path_example", self.files_meta[0].path)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        meta = self.files_meta[idx]

        # 1. 读取原始 waveform，numpy.float32，shape: [T]
        audio = self.read_file(meta)

        # 2. 训练集先做 RawBoost；必须在 pad / tensor 前做
        if self.is_train_mode and self.rawboost_args is not None and self.rawboost_func is not None:
            algo = int(getattr(self.rawboost_args, "algo", 0))
            if algo != 0:
                audio = self.rawboost_func(
                    audio,
                    16000,
                    self.rawboost_args,
                    algo
                )
                audio = np.asarray(audio, dtype=np.float32)
                audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        # 3. 再做 pad/crop 和 tensor transform
        if self.transform is not None:
            audio = self.transform(audio)

        # 4. 兼容 transform 返回 numpy 或 torch.Tensor
        x = self._to_tensor(audio)

        return x, float(meta.key), meta

    def _to_tensor(self, audio):
        if isinstance(audio, torch.Tensor):
            return audio.float()

        if isinstance(audio, np.ndarray):
            return torch.from_numpy(audio.astype(np.float32)).float()

        return torch.tensor(audio, dtype=torch.float32)

    def _get_split_name(self):
        if self.is_train:
            return 'train'
        if self.is_eval:
            return 'eval'
        return 'dev'

    def _get_protocol_file(self):
        if self.year == '2019':
            if self.is_train:
                fname = 'ASVspoof2019.LA.cm.train.trn.txt'
            else:
                if self.is_eval:
                    fname = 'ASVspoof2019.LA.cm.eval.trl.txt'
                else:
                    fname = 'ASVspoof2019.LA.cm.dev.trl.txt'

            path = os.path.join(self.protocols_path, fname)
        else:
            path = os.path.join(self.protocols_path, 'trial_metadata.txt')

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Protocol file not found: {path}\n"
                f"Please check protocols_path: {self.protocols_path}"
            )

        return path

    def _infer_2019_split_from_utt_id(self, utt_id):
        if utt_id.startswith('LA_T_'):
            return 'train'
        if utt_id.startswith('LA_D_'):
            return 'dev'
        if utt_id.startswith('LA_E_'):
            return 'eval'
        return self.split

    def _candidate_audio_paths(self, utt_id, split=None):
        if split is None:
            split = self._infer_2019_split_from_utt_id(utt_id)

        root = self.database_path
        task = self.task

        candidate_dirs = []

        if self.year == '2019':
            candidate_dirs.extend([
                # 标准结构：
                # database_path/ASVspoof2019_LA_train/flac/xxx.flac
                os.path.join(root, f'ASVspoof2019_{task}_{split}', 'flac'),
                os.path.join(root, f'ASVspoof2019_LA_{split}', 'flac'),

                # 有些人会多套一层 LA
                os.path.join(root, 'LA', f'ASVspoof2019_LA_{split}', 'flac'),

                # 有些人整理成 train/dev/eval/flac
                os.path.join(root, split, 'flac'),

                # 兼容旧写法
                os.path.join(root, 'flac'),

                # 如果 database_path 本身就是 flac 或 split 目录
                root,
            ])
        else:
            candidate_dirs.extend([
                os.path.join(root, 'flac'),
                os.path.join(root, task, 'flac'),
                os.path.join(root, f'ASVspoof2021_{task}_eval', 'flac'),
                os.path.join(root, f'ASVspoof2021_{task}', 'flac'),
                root,
            ])

        # 去重，保持顺序
        seen = set()
        unique_dirs = []
        for d in candidate_dirs:
            d = os.path.abspath(d)
            if d not in seen:
                unique_dirs.append(d)
                seen.add(d)

        exts = ['.flac', '.FLAC', '.wav', '.WAV']
        paths = []
        for d in unique_dirs:
            for ext in exts:
                paths.append(os.path.join(d, utt_id + ext))

        return paths

    def _resolve_audio_path(self, utt_id, split=None):
        candidates = self._candidate_audio_paths(utt_id, split=split)

        for path in candidates:
            if os.path.isfile(path):
                return path

        # 找不到时返回首选路径，用于报错显示
        return candidates[0]

    def read_file(self, meta):
        split = self._infer_2019_split_from_utt_id(meta.file_name) if self.year == '2019' else self.split
        candidates = self._candidate_audio_paths(meta.file_name, split=split)

        audio_path = None
        for path in candidates:
            if os.path.isfile(path):
                audio_path = path
                break

        if audio_path is None:
            tried = "\n".join(candidates[:20])
            raise FileNotFoundError(
                f"Audio file not found for utt_id: {meta.file_name}\n"
                f"Expected original meta.path: {meta.path}\n"
                f"Tried candidate paths:\n{tried}\n"
                f"If none of these exist, please run:\n"
                f"find {self.database_path} -name '{meta.file_name}*'"
            )

        try:
            waveform, sr = torchaudio.load(audio_path)

            if sr != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=16000
                )
                waveform = resampler(waveform)

            # 多声道转单声道
            if waveform.dim() == 2 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)
            waveform = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)

            if waveform.shape[0] == 0:
                raise RuntimeError(f"Empty waveform loaded from: {audio_path}")

            return waveform

        except Exception as e:
            raise RuntimeError(f"torchaudio cannot read {audio_path}: {e}")

    def _parse_2021_line(self, line):
        """
        兼容 ASVspoof2021 trial_metadata。
        注意：不同整理版本列含义可能有差异，这里保持你原来代码的主要逻辑。
        """
        tokens = line.strip().split()
        if len(tokens) < 2:
            return None

        speaker_id = tokens[0]
        utt_id = tokens[1]

        attack_id = '-'
        if len(tokens) > 4:
            attack_id = tokens[4]

        key_str = 'spoof'
        if len(tokens) > 5:
            key_str = tokens[5].lower()

        key = 1 if key_str == 'bonafide' else 0
        sys_id = self.sysid_dict.get(attack_id, 0)

        path = self._resolve_audio_path(utt_id, split=self.split)

        return ASVFile(
            speaker_id=speaker_id,
            file_name=utt_id,
            path=path,
            sys_id=sys_id,
            key=key
        )

    def parse_protocols_file(self, fname):
        with open(fname, 'r') as f:
            lines = f.readlines()

        files_meta = []
        eval_count = 0
        progress_count = 0
        hidden_count = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            tokens = line.split()

            if self.year == '2019':
                # ASVspoof2019 LA protocol:
                # speaker_id utt_id - attack_id key
                if len(tokens) < 5:
                    continue

                speaker_id = tokens[0]
                utt_id = tokens[1]
                attack_id = tokens[3]
                key_str = tokens[4].lower()

                # 训练标签：
                # bonafide -> 1
                # spoof    -> 0
                key = 1 if key_str == 'bonafide' else 0
                sys_id = self.sysid_dict.get(attack_id, 0)

                split = self._infer_2019_split_from_utt_id(utt_id)
                path = self._resolve_audio_path(utt_id, split=split)

                meta = ASVFile(
                    speaker_id=speaker_id,
                    file_name=utt_id,
                    path=path,
                    sys_id=sys_id,
                    key=key
                )
                files_meta.append(meta)
                eval_count += 1

            else:
                # ASVspoof2021 metadata:
                # 你原代码使用 tokens[7] 判断 phase，这里保留兼容。
                if len(tokens) >= 8:
                    phase = tokens[7]
                    if self.is_eval and phase != 'eval':
                        if phase == 'progress':
                            progress_count += 1
                        elif phase == 'hidden_track':
                            hidden_count += 1
                        continue

                meta = self._parse_2021_line(line)
                if meta is not None:
                    files_meta.append(meta)
                    eval_count += 1

        if self.year == '2021':
            print(f"Filtered: eval={eval_count}, progress={progress_count}, hidden={hidden_count}")

        return files_meta