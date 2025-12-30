import os, glob, shutil, subprocess

ROOT = "data/raw"
BAD_DIR = "data/bad"
os.makedirs(BAD_DIR, exist_ok=True)

# 你要扫哪些后缀
EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

# 每个文件检测超时（秒）
TIMEOUT = 5

def is_ok_audio(path: str) -> bool:
    """
    用 ffprobe 检查音频是否可解析：
    - 返回码为 0 且 encourages 有音频流 -> OK
    - 超时/非0 -> BAD
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type,codec_name,sample_rate,channels",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        if r.returncode != 0:
            return False
        out = (r.stdout or "").strip()
        # 有输出说明识别到音频流
        return len(out) > 0
    except subprocess.TimeoutExpired:
        return False
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found. Please install ffmpeg and ensure ffprobe is in PATH.")

def main():
    files = []
    for dp, _, fns in os.walk(ROOT):
        for fn in fns:
            if fn.lower().endswith(EXTS):
                files.append(os.path.join(dp, fn))

    print(f"Found {len(files)} audio files under {ROOT}")

    bad = []
    for i, fp in enumerate(files, 1):
        ok = is_ok_audio(fp)
        if not ok:
            bad.append(fp)
            print(f"[BAD] {fp}")
        if i % 200 == 0:
            print(f"checked {i}/{len(files)} ... bad={len(bad)}")

    print(f"Done. total={len(files)} bad={len(bad)}")

    # 移动坏文件
    for fp in bad:
        rel = os.path.relpath(fp, ROOT).replace("\\", "_").replace("/", "_")
        dst = os.path.join(BAD_DIR, rel)
        try:
            shutil.move(fp, dst)
        except Exception as e:
            print("move failed:", fp, "->", dst, "|", e)

    print("Bad files moved to:", BAD_DIR)

if __name__ == "__main__":
    main()
