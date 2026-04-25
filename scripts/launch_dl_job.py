"""
Launch DL downscaling pods on RunPod.

Usage:
    RUNPOD_API_KEY=<key> python downscaling/launch_dl_job.py --task prithvi-inference --years 2015-2021
    RUNPOD_API_KEY=<key> python downscaling/launch_dl_job.py --task unet-train
    RUNPOD_API_KEY=<key> python downscaling/launch_dl_job.py --task prithvi-finetune
    RUNPOD_API_KEY=<key> python downscaling/launch_dl_job.py --list
    RUNPOD_API_KEY=<key> python downscaling/launch_dl_job.py --list-gpus
    RUNPOD_API_KEY=<key> python downscaling/launch_dl_job.py --stop <POD_ID>
    RUNPOD_API_KEY=<key> python downscaling/launch_dl_job.py --status <POD_ID>

TASK choices:
    unet-train          U-Net FiLM training on Drôme-Ardèche 2000-2021 (A100, ~2h)
    prithvi-finetune    Fine-tune Prithvi WxC DEMConditionedAdapter (A100, ~3h)
    prithvi-inference   Prithvi WxC inference campaign 2015-2021 (L4, ~45min)

Requirements:
    pip install runpod

Environment variables:
    RUNPOD_API_KEY    RunPod API key (Settings → API Keys on runpod.io)
    MLFLOW_TRACKING_URI  MLflow server (default: from .env or http://localhost:5500)
    SCW_ACCESS_KEY    Scaleway S3 key (for output upload)
    SCW_SECRET_KEY    Scaleway S3 secret
"""

import argparse
import os
import sys

IMAGE = "registry.fr-par.scw.cloud/karpos/downscaling:latest"
VOLUME_NAME = "downscaling-workspace"
VOLUME_MOUNT = "/workspace"
CONTAINER_DISK_GB = 30

TASKS = {
    "unet-train": {
        "gpu": "A100",
        "vram_gb": 20,
        "cmd": (
            "uv run python downscaling/downscaling/scripts/run_dl_train.py"
            " --config downscaling/config/drome_ardeche.yml --epochs 150"
        ),
        "desc": "U-Net FiLM training — Drôme-Ardèche 2000-2021 (~2h, A100)",
    },
    "prithvi-finetune": {
        "gpu": "A100",
        "vram_gb": 40,
        "cmd": (
            "uv run python downscaling/downscaling/prtihvi_wxc/finetune.py"
            " --config downscaling/config/drome_ardeche.yml"
        ),
        "desc": "Prithvi WxC DEMConditionedAdapter fine-tuning (~3h, A100 80GB)",
    },
    "prithvi-inference": {
        "gpu": "L4",
        "vram_gb": 20,
        "cmd": (
            "uv run python downscaling/downscaling/scripts/run_dl_inference.py"
            " --config downscaling/config/drome_ardeche.yml"
        ),
        "desc": "Prithvi WxC inference campaign (~45min, L4 24GB)",
    },
}

VALID_TASKS = list(TASKS.keys())


def get_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        sys.exit("ERROR: set RUNPOD_API_KEY  (runpod.io → Settings → API Keys)")
    return key


def get_pub_key(path: str) -> str:
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        sys.exit(f"ERROR: SSH public key not found at {expanded}")
    with open(expanded) as f:
        return f.read().strip()


def get_volume_id(api_key: str) -> str:
    from runpod.api.graphql import run_graphql_query
    import runpod
    runpod.api_key = api_key
    result = run_graphql_query(
        "{ myself { networkVolumes { id name size dataCenter { id } } } }"
    )
    volumes = result.get("data", {}).get("myself", {}).get("networkVolumes", [])
    for v in volumes:
        if v["name"] == VOLUME_NAME:
            return v["id"]
    available = [v["name"] for v in volumes]
    sys.exit(
        f"ERROR: network volume '{VOLUME_NAME}' not found.\n"
        f"Available: {available}\n"
        f"Create it:  terraform -chdir=infra apply  (see infra/runpod_volumes.tf)"
    )


def get_gpu_candidates(rp, display_name: str) -> list[str]:
    gpus = rp.get_gpus()
    matches = [g for g in gpus if display_name.lower() in g["id"].lower()]
    if not matches:
        print("Available GPU types:")
        for g in gpus:
            print(f"  {g['id']}  ({g.get('memoryInGb', '?')} GB)")
        sys.exit(f"ERROR: no GPU matching '{display_name}'")
    return [g["id"] for g in sorted(matches, key=lambda g: g.get("memoryInGb", 0), reverse=True)]


def list_gpus() -> None:
    import runpod as rp
    rp.api_key = get_api_key()
    gpus = rp.get_gpus()
    gpus_sorted = sorted(gpus, key=lambda g: (
        -g.get("lowestPrice", {}).get("stockStatus", 0),
        g.get("memoryInGb", 0),
    ))
    print(f"\n{'GPU ID':<45} {'VRAM':>6}  {'$/hr':>6}  Stock")
    print("-" * 75)
    for g in gpus_sorted:
        mem = g.get("memoryInGb", "?")
        price = g.get("lowestPrice", {})
        cost = price.get("minimumBidPrice") or price.get("uninterruptablePrice") or "?"
        stock = price.get("stockStatus", "?")
        cost_str = f"${cost:.3f}" if isinstance(cost, (int, float)) else str(cost)
        print(f"  {g['id']:<43} {str(mem):>5} GB  {cost_str:>6}  {stock}")
    print()
    print("Recommended: --gpu L4 (prithvi-inference)  --gpu A100 (training)")


def list_pods() -> None:
    import runpod as rp
    rp.api_key = get_api_key()
    pods = rp.get_pods()
    if not pods:
        print("No running pods.")
        return
    for p in pods:
        status = p.get("desiredStatus", "?")
        name = p.get("name", "?")
        print(f"  {p['id']}  {name:<35}  {status}")


def status_pod(pod_id: str) -> None:
    import runpod as rp
    rp.api_key = get_api_key()
    pod = rp.get_pod(pod_id)
    if not pod:
        sys.exit(f"Pod {pod_id} not found.")
    print(f"Pod ID  : {pod_id}")
    print(f"Name    : {pod.get('name', '?')}")
    print(f"Status  : {pod.get('desiredStatus', '?')}")
    print(f"GPU     : {pod.get('machine', {}).get('gpuDisplayName', '?')}")
    ports = (pod.get("runtime") or {}).get("ports") or []
    for port in ports:
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            h = port.get("ip", "ssh.runpod.io")
            print(f"SSH     : ssh root@{h} -p {port['publicPort']} -i ~/.ssh/id_ed25519")


def stop_pod(pod_id: str) -> None:
    import runpod as rp
    rp.api_key = get_api_key()
    rp.terminate_pod(pod_id)
    print(f"Pod {pod_id} terminated.")


def create_pod(task: str, years: str | None, ssh_key_path: str,
               gpu_override: str | None, dry_run: bool) -> None:
    import runpod as rp
    import runpod.error as rp_error

    spec = TASKS[task]
    gpu_name = gpu_override or spec["gpu"]

    cmd = spec["cmd"]
    if years and task == "prithvi-inference":
        cmd += f" --years {years}"

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5500")
    scw_access = os.environ.get("SCW_ACCESS_KEY", "")
    scw_secret = os.environ.get("SCW_SECRET_KEY", "")

    print(f"\nTask     : {task}")
    print(f"Desc     : {spec['desc']}")
    print(f"GPU      : {gpu_name}  (≥{spec['vram_gb']} GB VRAM required)")
    print(f"Image    : {IMAGE}")
    print(f"Volume   : {VOLUME_NAME} → {VOLUME_MOUNT}")
    print(f"Command  : {cmd}")
    if dry_run:
        print("\n[dry-run] — no pod created")
        return

    api_key = get_api_key()
    rp.api_key = api_key
    pub_key = get_pub_key(ssh_key_path)
    volume_id = get_volume_id(api_key)
    gpu_candidates = get_gpu_candidates(rp, gpu_name)

    pod = None
    for gpu_type_id in gpu_candidates:
        print(f"\nTrying gpu={gpu_type_id}  volume={volume_id}")
        try:
            pod = rp.create_pod(
                name=f"dl-{task}",
                image_name=IMAGE,
                gpu_type_id=gpu_type_id,
                gpu_count=1,
                container_disk_in_gb=CONTAINER_DISK_GB,
                network_volume_id=volume_id,
                ports="22/tcp",
                env={
                    "TASK": task,
                    "RUNPOD_PUBLIC_KEY": pub_key,
                    "MLFLOW_TRACKING_URI": mlflow_uri,
                    "SCW_ACCESS_KEY": scw_access,
                    "SCW_SECRET_KEY": scw_secret,
                    "HF_HOME": f"{VOLUME_MOUNT}/.hf_cache",
                },
                docker_args=cmd,
            )
            break
        except rp_error.QueryError as e:
            if "no longer any instances available" in str(e):
                print(f"  → {gpu_type_id} full, trying next...")
                continue
            raise

    if pod is None:
        sys.exit(
            f"ERROR: all {gpu_name} GPU types currently full.\n"
            f"Retry later or pass --gpu H100 to use a larger GPU."
        )

    pod_id = pod["id"]
    print(f"\nPod created  id={pod_id}")
    print(f"Watch : python downscaling/launch_dl_job.py --status {pod_id}")
    print(f"Stop  : python downscaling/launch_dl_job.py --stop {pod_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch DL downscaling pods on RunPod")
    parser.add_argument("--task", choices=VALID_TASKS, help="DL task to run")
    parser.add_argument("--years", metavar="RANGE",
                        help="Year range for inference, e.g. 2015-2021 (prithvi-inference only)")
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519.pub", metavar="PATH")
    parser.add_argument("--gpu", metavar="NAME",
                        help="Override GPU family (default: per-task recommendation)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config without creating a pod")
    parser.add_argument("--stop", metavar="POD_ID", help="Terminate a pod")
    parser.add_argument("--status", metavar="POD_ID", help="Show pod status and SSH info")
    parser.add_argument("--list", action="store_true", help="List running pods")
    parser.add_argument("--list-gpus", action="store_true", help="Show available GPU types and cost")
    args = parser.parse_args()

    try:
        import runpod  # noqa: F401
    except ImportError:
        sys.exit("ERROR: runpod not installed. Run: pip install runpod")

    if args.stop:
        stop_pod(args.stop)
    elif args.status:
        status_pod(args.status)
    elif args.list:
        list_pods()
    elif args.list_gpus:
        list_gpus()
    elif args.task:
        create_pod(
            task=args.task,
            years=args.years,
            ssh_key_path=args.ssh_key,
            gpu_override=args.gpu,
            dry_run=args.dry_run,
        )
    else:
        parser.print_help()
        print("\nAvailable tasks:")
        for name, spec in TASKS.items():
            print(f"  {name:<22} {spec['desc']}")


if __name__ == "__main__":
    main()
