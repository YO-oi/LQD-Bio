import os
import shutil
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset, DownloadConfig, Audio


# =========================================================
# Robust Loader (带超时与重试的加载器)
# =========================================================
def load_dataset_with_retry(
        repo_id,
        subset,
        split,
        max_retries=5,
        base_delay=10
):
    """
    带指数退避重试的 load_dataset。
    应对 HF 镜像下载大文件时容易中途 Connection Reset 的问题。
    """
    # 配置底层下载参数：增加 timeout 和 内部 max_retries
    dl_config = DownloadConfig(
        max_retries=10,  # 遇到单次断流时，底层的重试次数
    )

    for attempt in range(max_retries):
        try:
            print(f"\n---> 正在尝试下载 {split} (第 {attempt + 1}/{max_retries} 次尝试)...")

            dataset = load_dataset(
                path=repo_id,
                name=subset,
                split=split,
                trust_remote_code=True,
                download_config=dl_config
            )

            # 【修复地雷】彻底关闭音频自动解码，否则循环时依然会拖垮 CPU
            dataset = dataset.cast_column("audio", Audio(decode=False))

            print(f"---> {split} 数据集加载并解析成功！")
            return dataset

        except Exception as e:
            print(f"\n[Error] 下载或加载失败: {e}")
            if attempt < max_retries - 1:
                # 指数退避：等待时间逐渐变长 10s, 20s, 40s...
                sleep_time = base_delay * (2 ** attempt)
                print(f"---> 等待 {sleep_time} 秒后启动重试机制...")
                time.sleep(sleep_time)
            else:
                raise RuntimeError(f"❌ {split} 下载彻底失败，已达到最大重试次数！")


def save_split(dataset, split_name, root_dir):
    split_dir = root_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = root_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []

    print("\n" + "=" * 60)
    print(f"开始处理 split: {split_name}")
    print("=" * 60)
    print(f"样本数量: {len(dataset)}")

    for idx, sample in enumerate(dataset):
        try:
            audio_info = sample["audio"]
            src_path = audio_info["path"]

            if src_path is None:
                continue

            src_path = Path(src_path)

            saved_file_name = f"{idx}_{src_path.name}"
            dst_path = split_dir / saved_file_name

            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)

            row = {}
            for key, value in sample.items():
                if key == "audio":
                    row["audio_path"] = str(dst_path)
                    row["original_audio_path"] = str(src_path)
                    continue
                row[key] = value

            metadata_rows.append(row)

            if idx % 500 == 0:  # 稍微调低打印频率，避免刷屏
                print(f"[{idx}/{len(dataset)}] {saved_file_name}")

        except Exception as e:
            print(f"\n错误样本 index: {idx}")
            print(e)

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = metadata_dir / f"{split_name}_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print(f"\n✅ metadata 已保存: {metadata_path}")
    print(f"✅ {split_name} 完成")


def download_per_dataset():
    #https://huggingface.co
    #https://hf-mirror.com
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    os.environ["HF_HOME"] = "./hf_cache"
    # [新增] 全局控制底层 huggingface_hub 的下载超时时间 (单位：秒)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1200"

    repo_id = "DBD-research-group/BirdSet"
    subset = "PER"

    # 保持你目前的绝对路径
    save_root = Path("C:/PycharmProjects/LQD/PER_audio")
    save_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("下载 BirdSet PER")
    print("包含：增加 Timeout / 指数退避重试 / 关闭音频解码")
    print("=" * 60)

    # ---------------------------------------------------
    # train
    # ---------------------------------------------------
    train_dataset = load_dataset_with_retry(
        repo_id=repo_id,
        subset=subset,
        split="train"
    )
    save_split(train_dataset, "train", save_root)

    # ---------------------------------------------------
    # test
    # ---------------------------------------------------
    test_dataset = load_dataset_with_retry(
        repo_id=repo_id,
        subset=subset,
        split="test"
    )
    save_split(test_dataset, "test", save_root)

    print("\n🎉 全部下载与处理完成!")


if __name__ == "__main__":
    download_per_dataset()