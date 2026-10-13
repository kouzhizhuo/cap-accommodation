import sys, time, os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"; os.environ["HF_HUB_DISABLE_XET"]="1"; os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]="60"
from huggingface_hub import snapshot_download
repo=sys.argv[1]; d="<HOME>/arr_oct/data/P3_prefdata/"+repo.split("/")[1]
for i in range(30):
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=d, max_workers=8); print("DONE", repo, flush=True); break
    except Exception as e:
        print("retry", i, repr(e)[:200], flush=True); time.sleep(10)
