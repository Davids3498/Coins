import gdown
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "AIzaSyA3HigIKTDeIwigXGQMCIzt3thXO5OQODk"
MAIN_FOLDER_ID = "1W798SkFlpLCJdFJaIynMxXV8M_G9qw43"
OUTPUT_DIR = Path("~/datasets/side_a_images").expanduser()
TARGET_SUBFOLDER = "side_a"
MAX_WORKERS = 5

def list_children(folder_id: str, mime_type: str = None) -> list[dict]:
    q = f"'{folder_id}' in parents and trashed=false"
    if mime_type:
        q += f" and mimeType='{mime_type}'"
    
    url = "https://www.googleapis.com/drive/v3/files"
    all_files = []
    page_token = None

    while True:
        params = {
            "q": q,
            "key": API_KEY,
            "fields": "nextPageToken, files(id,name)",
            "pageSize": 1000,
            "supportsAllDrives": True,
            "includeItemsFromAllDrives": True,
        }
        if page_token:
            params["pageToken"] = page_token

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        all_files.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return all_files

def download_file(file: dict, dest_dir: Path) -> None:
    output = dest_dir / file["name"]
    if output.exists():
        return  # skip already downloaded
    
    url = f"https://www.googleapis.com/drive/v3/files/{file['id']}?alt=media&key={API_KEY}"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def download_class(cls: dict) -> str:
    subfolders = list_children(cls["id"], mime_type="application/vnd.google-apps.folder")
    side_a = next((f for f in subfolders if f["name"] == TARGET_SUBFOLDER), None)
    
    if not side_a:
        return f"SKIP: no '{TARGET_SUBFOLDER}' in {cls['name']}"

    dest = OUTPUT_DIR / cls["name"]
    dest.mkdir(exist_ok=True)

    files = list_children(side_a["id"])
    for file in files:
        download_file(file, dest)

    return f"DONE: {cls['name']} ({len(files)} files)"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    class_folders = list_children(MAIN_FOLDER_ID, mime_type="application/vnd.google-apps.folder")
    print(f"Found {len(class_folders)} class folders, downloading with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_class, cls): cls for cls in class_folders}
        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    main()