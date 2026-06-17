import numpy as np
import os
import sys
import random
import torch
import zipfile
import urllib.request
import ssl
from PIL import Image
from utils.dataset_utils import check, separate_data, split_data, save_file, imbalance_factor
import utils.dataset_utils as du


random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
num_clients = 20
dir_path = "TinyImageNet/"


def load_tinyimagenet(root):
    """
    加载 TinyImageNet-200 数据集，自动下载。
    返回与 CIFAR 相同格式的 (train_images, train_labels, test_images, test_labels)。
    """
    data_dir = os.path.join(root, "tiny-imagenet-200")

    if not os.path.exists(data_dir):
        # 自动下载并解压
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = os.path.join(root, "tiny-imagenet-200.zip")

        if not os.path.exists(zip_path):
            print(f"[Info] 正在下载 TinyImageNet-200 ...")
            print(f"       URL: {url}")
            try:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_peer = False
                urllib.request.urlretrieve(url, zip_path)
            except Exception as e:
                print(f"[Warn] SSL 下载失败，尝试跳过证书验证: {e}")
                try:
                    ctx = ssl._create_unverified_context()
                    urllib.request.urlretrieve(url, zip_path, context=ctx)
                except Exception as e2:
                    raise RuntimeError(
                        f"TinyImageNet 自动下载失败: {e2}\n"
                        f"请手动下载并解压:\n"
                        f"  1. 下载: {url}\n"
                        f"  2. 解压到: {root}/ 目录下（确保存在 {root}/tiny-imagenet-200/）"
                    )
            print(f"[Info] 下载完成: {zip_path}")

        print(f"[Info] 正在解压 ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root)
        print(f"[Info] 解压完成: {data_dir}")

        if os.path.exists(zip_path):
            os.remove(zip_path)

    # 读取类别ID列表
    wnids_file = os.path.join(data_dir, "wnids.txt")
    with open(wnids_file, 'r') as f:
        wnids = [line.strip() for line in f.readlines()]
    wnids.sort()
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # ===== 加载训练集 =====
    train_images = []
    train_labels = []
    train_dir = os.path.join(data_dir, "train")

    print(f"[Info] 正在加载训练集 (200类 x 500张 = 100,000张图片)...")
    for idx, wnid in enumerate(wnids):
        img_dir = os.path.join(train_dir, wnid, "images")
        label = wnid_to_label[wnid]
        for img_name in os.listdir(img_dir):
            if not img_name.endswith(".JPEG"):
                continue
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)  # (64, 64, 3) uint8
            train_images.append(img_array)
            train_labels.append(label)
        if (idx + 1) % 20 == 0:
            print(f"       已加载 {idx + 1}/200 类 ({(idx+1)*500} 张)", flush=True)

    # ===== 加载验证集（作为测试集） =====
    print(f"[Info] 正在加载验证集 (10,000张)...")
    val_images = []
    val_labels = []
    val_dir = os.path.join(data_dir, "val")
    val_annotations_file = os.path.join(val_dir, "val_annotations.txt")

    # 解析 val_annotations.txt: 每行格式为 "img_name\twnid\t..."
    val_img_to_wnid = {}
    with open(val_annotations_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            val_img_to_wnid[parts[0]] = parts[1]

    val_img_dir = os.path.join(val_dir, "images")
    for img_name, wnid in val_img_to_wnid.items():
        img_path = os.path.join(val_img_dir, img_name)
        if not os.path.exists(img_path):
            continue
        label = wnid_to_label[wnid]
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        val_images.append(img_array)
        val_labels.append(label)

    train_images = np.array(train_images, dtype=np.uint8)
    train_labels = np.array(train_labels, dtype=np.int64)
    val_images = np.array(val_images, dtype=np.uint8)
    val_labels = np.array(val_labels, dtype=np.int64)

    print(f"[Info] TinyImageNet 加载完成:")
    print(f"       训练集: {train_images.shape}, 测试集: {val_images.shape}")
    print(f"       类别数: {len(wnids)}")

    return train_images, train_labels, val_images, val_labels


def generate_dataset(dir_path, num_clients, niid, balance, partition, longtail=False, longtail_type=None, imb_factor=None):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition, longtail, longtail_type):
        return

    # 加载 TinyImageNet
    rawdata_dir = os.path.join(dir_path, "rawdata")
    if not os.path.exists(rawdata_dir):
        os.makedirs(rawdata_dir)
    train_images, train_labels, test_images, test_labels = load_tinyimagenet(rawdata_dir)

    # 合并训练集和测试集（与 CIFAR 生成逻辑一致）
    dataset_image = np.concatenate([train_images, test_images], axis=0)
    dataset_label = np.concatenate([train_labels, test_labels], axis=0)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # 设置长尾分布的不平衡因子
    if imb_factor is not None:
        imb_factor_value = imb_factor
    else:
        imb_factor_value = imbalance_factor

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=20, longtail=longtail,
                                    longtail_type=longtail_type, imbalance_factor=imb_factor_value, train_path=train_path)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
        statistic, niid, balance, partition, longtail, longtail_type, imb_factor_value)

    # ===== 保存全局测试集 =====
    global_test_path = os.path.join(dir_path, "global_test.npz")
    global_test_data = {
        'x': test_images,
        'y': test_labels
    }
    np.savez_compressed(global_test_path, data=global_test_data)
    print(f"[Info] Saved global test set: {global_test_path}")
    print(f"       - Samples: {len(test_images)}")
    print(f"       - Classes: {num_classes} (balanced, ~{len(test_images)//num_classes} per class)")


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    longtail = False
    longtail_type = None
    imb_factor = None

    if len(sys.argv) > 4:
        longtail = True if sys.argv[4] == "longtail" else False

        if longtail and len(sys.argv) > 5:
            longtail_type_map = {
                "global": "global_longtail",
                "local": "local_longtail",
                "mixed": "mixed_longtail",
                "-": "global_longtail"
            }
            user_type = sys.argv[5] if sys.argv[5] in ["global", "local", "mixed", "-"] else "global"
            longtail_type = longtail_type_map[user_type]

            if len(sys.argv) > 6:
                try:
                    imb_factor = float(sys.argv[6])
                    if imb_factor < 1:
                        print("Warning: IF should be >= 1. Using default.")
                        imb_factor = None
                except ValueError:
                    print("Warning: Invalid imbalance factor, using default value.")

    alpha_tag = sys.argv[7] if len(sys.argv) > 7 else "0.5"
    if len(sys.argv) > 7:
        try:
            alpha_val = float(sys.argv[7])
            if alpha_val <= 0 or alpha_val > 1:
                print("Warning: alpha should be in (0, 1]. Using default 0.5")
                du.alpha = 0.5
            else:
                du.alpha = alpha_val
        except ValueError:
            print("Warning: Invalid alpha value; using default:", du.alpha)

    if len(sys.argv) > 8:
        try:
            num_clients_cli = int(sys.argv[8])
            if num_clients_cli > 0:
                num_clients = num_clients_cli
            else:
                print("Warning: num_clients should be > 0; using default:", num_clients)
        except ValueError:
            print("Warning: Invalid num_clients argument; using default:", num_clients)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if longtail:
        if imb_factor is not None and imb_factor >= 1:
            IF_tag = f"IF{int(imb_factor)}"
        else:
            IF_tag = "IFauto"
        type_tag = user_type if longtail and 'user_type' in locals() else 'global'
        dir_name = f"TinyImageNet-{IF_tag}-α{alpha_tag}-{type_tag}-NC{num_clients}"
    else:
        dir_name = f"TinyImageNet-NC{num_clients}"
    dir_path = os.path.join(script_dir, dir_name) + os.sep

    print(f"Generating TinyImageNet dataset with settings:")
    print(f"  - Non-IID: {niid}")
    print(f"  - Balanced: {balance}")
    print(f"  - Partition: {partition}")
    print(f"  - Long-tail: {longtail}")
    print(f"  - Long-tail Type: {user_type if longtail and 'user_type' in locals() else '-'}")
    print(f"  - Imbalance Factor: {imb_factor if imb_factor is not None else imbalance_factor}")
    print(f"  - Alpha: {du.alpha}")
    print(f"  - Output Dir: {dir_path}")

    print(f"[debug] alpha={du.alpha}, num_clients={num_clients}")

    generate_dataset(dir_path, num_clients, niid, balance, partition, longtail, longtail_type, imb_factor)
