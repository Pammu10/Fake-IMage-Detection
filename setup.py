import argparse
import os
import zipfile


REQUIREMENTS = [
    "torch",
    "torchvision",
    "transformers",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "pandas",
    "numpy",
    "pillow",
    "python-pptx",
    "gradio",
    "kaggle",
]


def create_project_folders() -> None:
    folders = [
        os.path.join("data", "cifake"),
        os.path.join("outputs", "graphs"),
        os.path.join("outputs", "models"),
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")


def write_requirements() -> None:
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(REQUIREMENTS) + "\n")
    print("Created requirements.txt")


def download_cifake_from_kaggle(dataset_slug: str = "birdy654/cifake-real-and-ai-generated-synthetic-images") -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("kaggle package is not installed yet. Install it first using requirements.txt")
        return

    print("Trying to download CIFAKE from Kaggle...")
    print("Kaggle credential file should exist at one of these paths:")
    print("- Linux/WSL/macOS: ~/.kaggle/kaggle.json")
    print("- Windows: %USERPROFILE%\\.kaggle\\kaggle.json")

    api = KaggleApi()
    api.authenticate()

    zip_path = os.path.join("data", "cifake.zip")
    target_dir = os.path.join("data", "cifake")

    api.dataset_download_files(dataset_slug, path="data", unzip=False)

    # Try to find the newest zip if Kaggle changed naming.
    if not os.path.exists(zip_path):
        zip_candidates = [
            os.path.join("data", x)
            for x in os.listdir("data")
            if x.lower().endswith(".zip")
        ]
        if zip_candidates:
            zip_path = max(zip_candidates, key=os.path.getmtime)

    if not os.path.exists(zip_path):
        print("Could not locate downloaded zip file automatically.")
        print("Please unzip the downloaded CIFAKE zip manually into data/cifake")
        return

    print(f"Unzipping: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    print(f"Dataset extracted to: {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="Project setup for AI vs Real Image Detection")
    parser.add_argument(
        "--download-cifake",
        action="store_true",
        help="Download CIFAKE from Kaggle API during setup",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default="birdy654/cifake-real-and-ai-generated-synthetic-images",
        help="Kaggle dataset slug, e.g. owner/dataset-name",
    )
    args = parser.parse_args()

    print("=== Project Setup: AI vs Real Image Detection ===")
    create_project_folders()
    write_requirements()

    if args.download_cifake:
        download_cifake_from_kaggle(args.dataset_slug)

    print("\nNext steps:")
    print("1) Install dependencies: pip install -r requirements.txt")
    print("2) (Optional) Download CIFAKE with Kaggle API: python setup.py --download-cifake")
    print("3) Ensure dataset is available at data/cifake with REAL/FAKE class folders")
    print("4) Run training: python train.py")
    print("5) Generate slides: python generate_ppt.py")
    print("6) Launch demo: python demo.py")


if __name__ == "__main__":
    main()
