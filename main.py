import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

import data_utils
from model import RawGAT_ST
from RawBoost import RawBoostArgs


def set_random_seed(seed, args=None):
    """Set random seeds for reproducible training and evaluation."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    deterministic = True
    benchmark = False
    if args is not None:
        deterministic = bool(getattr(args, "cudnn_deterministic_toggle", True))
        benchmark = bool(getattr(args, "cudnn_benchmark_toggle", False))

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def load_checkpoint(model, model_path, device):
    """Load a PyTorch checkpoint and handle optional DataParallel prefixes."""
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key.replace("module.", "", 1)
        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    return model


def pad(x, max_len=64600):
    x = np.asarray(x, dtype=np.float32)
    x_len = x.shape[0]

    if x_len <= 0:
        return np.zeros(max_len, dtype=np.float32)

    if x_len >= max_len:
        return x[:max_len].astype(np.float32)

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, num_repeats)[:max_len]

    return padded_x.astype(np.float32)


def _as_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, (list, tuple)):
        return list(x)

    return [x]


def _unpack_batch_meta(batch_meta):
    """
    兼容 PyTorch DataLoader 对 namedtuple 的 collate 行为。

    Dataset 返回单个 meta 是 ASVFile namedtuple。
    batch 后通常会变成：
        ASVFile(
            speaker_id=(...),
            file_name=(...),
            path=(...),
            sys_id=tensor(...),
            key=tensor(...)
        )
    """
    if hasattr(batch_meta, "_fields"):
        speaker_ids = _as_list(batch_meta.speaker_id)
        file_names = _as_list(batch_meta.file_name)
        paths = _as_list(batch_meta.path)
        sys_ids = _as_list(batch_meta.sys_id)
        keys = _as_list(batch_meta.key)

        return speaker_ids, file_names, paths, sys_ids, keys

    speaker_ids, file_names, paths, sys_ids, keys = [], [], [], [], []

    for meta in batch_meta:
        speaker_ids.append(meta.speaker_id)
        file_names.append(meta.file_name)
        paths.append(meta.path)
        sys_ids.append(meta.sys_id)
        keys.append(meta.key)

    return speaker_ids, file_names, paths, sys_ids, keys


def evaluate_accuracy(data_loader, model, device, eval_batch_size=8):
    val_loss = 0.0
    num_total = 0.0

    model.eval()

    # label: spoof=0, bonafide=1
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    pbar = tqdm(
        data_loader,
        desc='Evaluating',
        ncols=100,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    with torch.no_grad():
        for batch_x, batch_y, batch_meta in pbar:
            batch_size = batch_x.size(0)
            num_total += batch_size

            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            batch_out = model(batch_x, Freq_aug=False)
            batch_loss = criterion(batch_out, batch_y)

            val_loss += (batch_loss.item() * batch_size)

    val_loss /= max(num_total, 1.0)

    return val_loss


def produce_evaluation_file(dataset, model, device, save_path, task='LA',
                            year='2019', batch_size=16, num_workers=0):
    """
    生成评分文件。

    2019 LA:
        输出 4 列：
        utt_id attack_id key score

    2021 LA:
        输出 2 列：
        utt_id score

    2021 DF:
        输出 2 列：
        utt_id score

    注意：
        ASVspoof2021 官方 evaluate.py 需要提交格式为：
        utterance_id score
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    model.eval()

    pbar = tqdm(
        data_loader,
        desc=f'Producing {year} {task} scores',
        ncols=120,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                   '[{elapsed}<{remaining}, {rate_fmt}, {postfix}]'
    )

    num_written = 0

    with open(save_path, 'w', encoding='utf-8') as fh:
        with torch.no_grad():
            for batch_x, batch_y, batch_meta in pbar:
                batch_x = batch_x.to(device)

                batch_out = model(batch_x, Freq_aug=False)

                # class 1 = bonafide score
                batch_score = batch_out[:, 1].detach().cpu().numpy().ravel()

                speaker_ids, file_names, paths, sys_ids, keys = _unpack_batch_meta(batch_meta)

                # 只有 2019 LA 保留 4 列格式
                if str(year) == '2019' and task == 'LA':
                    for utt_id, sys_id, key, score in zip(file_names, sys_ids, keys, batch_score):
                        key_str = 'bonafide' if int(key) == 1 else 'spoof'
                        attack_id = dataset.sysid_dict_inv.get(int(sys_id), '-')
                        fh.write('{} {} {} {}\n'.format(utt_id, attack_id, key_str, score))
                        num_written += 1
                else:
                    # 2021 LA / 2021 DF 输出官方提交格式：utt_id score
                    for utt_id, score in zip(file_names, batch_score):
                        fh.write('{} {}\n'.format(utt_id, score))
                        num_written += 1

                fh.flush()
                pbar.set_postfix(written=num_written)

    print('Result saved to {}'.format(save_path), flush=True)
    print('Total scores written: {}'.format(num_written), flush=True)


def train_epoch(data_loader, model, lr, optimizer, device):
    running_loss = 0.0
    num_total = 0.0

    model.train()

    # label: spoof=0, bonafide=1
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    pbar = tqdm(
        data_loader,
        desc='Training Epoch',
        ncols=100,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    for batch_x, batch_y, batch_meta in pbar:
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        batch_out = model(batch_x, Freq_aug=True)
        batch_loss = criterion(batch_out, batch_y)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        running_loss += (batch_loss.item() * batch_size)

        avg_loss = running_loss / max(num_total, 1.0)
        pbar.set_postfix(loss=f'{avg_loss:.5f}')

    running_loss /= max(num_total, 1.0)

    return running_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TreeFus-ST for spoofed speech detection')

    # Dataset
    parser.add_argument(
        '--database_path',
        type=str,
        default='/public/home/acal2okrm7997/wmnt/LA/LA/',
        help='Base directory containing ASVspoof2019 LA data folders'
    )

    parser.add_argument(
        '--protocols_path',
        type=str,
        default='/public/home/acal2okrm7997/wmnt/LA/LA/ASVspoof2019_LA_cm_protocols/',
        help='Directory containing ASVspoof2019 LA protocol files'
    )

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE', help='Weighted Cross Entropy Loss')

    # Model / run
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--model_path', type=str, default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None, help='Comment to describe the saved model')

    # Auxiliary arguments
    parser.add_argument(
        '--track',
        type=str,
        default='logical',
        choices=['logical', 'physical'],
        help='logical/physical'
    )

    parser.add_argument('--eval_output', type=str, default=None, help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False, help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False, help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    parser.add_argument('--features', type=str, default='Raw_GAT')

    # backend options
    parser.add_argument(
        '--cudnn-deterministic-toggle',
        action='store_false',
        default=True,
        help='use cudnn-deterministic?'
    )

    parser.add_argument(
        '--cudnn-benchmark-toggle',
        action='store_true',
        default=False,
        help='use cudnn-benchmark?'
    )

    # RawBoost data augmentation options
    parser.add_argument(
        '--rawboost_algo',
        type=int,
        default=4,
        help='RawBoost algorithm: 0=no augmentation, 1=LnL, 2=ISD, 3=SSI, 4=LnL+ISD+SSI'
    )

    parser.add_argument(
        '--enable_rawboost',
        action='store_true',
        default=False,
        help='Enable RawBoost data augmentation for training set'
    )

    # Dataset year / task
    parser.add_argument(
        '--year',
        type=str,
        default='2019',
        choices=['2019', '2021'],
        help='Dataset year'
    )

    parser.add_argument(
        '--task',
        type=str,
        default='LA',
        choices=['LA', 'DF'],
        help='Task type: LA or DF'
    )

    # Optional dataloader workers
    parser.add_argument('--num_workers', type=int, default=0)

    dir_yaml = os.path.splitext('model_config_TreeFus-ST')[0] + '.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)

    if not os.path.exists('models'):
        os.mkdir('models')

    args = parser.parse_args()

    set_random_seed(args.seed, args)

    track = args.track
    assert track in ['logical', 'physical'], 'Invalid track given'
    is_logical = (track == 'logical')

    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track,
        args.loss,
        args.num_epochs,
        args.batch_size,
        args.lr
    )

    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)

    model_save_path = os.path.join('models', model_tag)

    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    transform = transforms.Compose([
        lambda x: pad(x),
        lambda x: torch.tensor(x, dtype=torch.float32)
    ])

    # RawBoost 参数来自 RawBoost.py；评估时不要加 --enable_rawboost
    rawboost_args = None
    if args.enable_rawboost:
        rawboost_args = RawBoostArgs(algo=args.rawboost_algo)
        print(f'RawBoost enabled with algorithm: {args.rawboost_algo}', flush=True)
        print(
            'RawBoost official-style config: '
            f'nBands={rawboost_args.nBands}, '
            f'minF={rawboost_args.minF}, '
            f'maxF={rawboost_args.maxF}, '
            f'minCoeff={rawboost_args.minCoeff}, '
            f'maxCoeff={rawboost_args.maxCoeff}, '
            f'g_sd={rawboost_args.g_sd}, '
            f'SNR=[{rawboost_args.SNRmin}, {rawboost_args.SNRmax}]',
            flush=True
        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device), flush=True)

    # dev/eval set: 不做 RawBoost
    dev_set = data_utils.ASVDataset(
        database_path=args.database_path,
        protocols_path=args.protocols_path,
        is_train=False,
        is_logical=is_logical,
        transform=transform,
        feature_name=args.features,
        is_eval=args.is_eval,
        eval_part=args.eval_part,
        rawboost_args=None,
        year=args.year,
        task=args.task
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    eval_loader = DataLoader(
        dev_set,
        batch_size=4,
        shuffle=False,
        num_workers=args.num_workers
    )

    model_cfg = parser1['model']

    use_treefus_st = bool(model_cfg.get('use_treefus_st', True))

    model = RawGAT_ST(
        model_cfg,
        device,
        use_treefus_st=use_treefus_st
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    if args.model_path:
        load_checkpoint(model, args.model_path, device)
        print('Model loaded : {}'.format(args.model_path), flush=True)

    # Inference mode
    if args.eval:
        assert args.eval_output is not None, 'You must provide an output path'
        assert args.model_path is not None, 'You must provide model checkpoint'

        produce_evaluation_file(
            dev_set,
            model,
            device,
            args.eval_output,
            task=args.task,
            year=args.year,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        sys.exit(0)

    # 训练集固定为 2019 LA
    # 原因：ASVspoof2021 LA/DF 通常用于 eval，没有同格式官方训练集。
    train_set = data_utils.ASVDataset(
        database_path=args.database_path,
        protocols_path=args.protocols_path,
        is_train=True,
        is_logical=is_logical,
        transform=transform,
        feature_name=args.features,
        rawboost_args=rawboost_args,
        year='2019',
        task='LA'
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    writer = SummaryWriter('logs/{}'.format(model_tag))

    for epoch in range(args.num_epochs):
        running_loss = train_epoch(
            train_loader,
            model,
            args.lr,
            optimizer,
            device
        )

        val_loss = evaluate_accuracy(
            eval_loader,
            model,
            device
        )

        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)

        print(f'\n{epoch} - train_loss={running_loss:.10f} - val_loss={val_loss:.10f}', flush=True)

        torch.save(
            model.state_dict(),
            os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch))
        )

    writer.close()