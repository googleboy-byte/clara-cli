def download_tess_sector_threaded(sector_number, catalogues_folder,
                                  out_dir_tess,
                                  start_index=0,
                                  max_downloads=None,
                                  log_file="../logs/download_tess_lc.log",
                                  num_threads=8):
    import requests
    import os
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    from clara_benchmark.utils.logging import log

    sh_filename = f"tesscurl_sector_{sector_number}_lc.sh"
    sh_path = os.path.join(catalogues_folder, sh_filename)
    
    out_dir = os.path.join(out_dir_tess, str(sector_number))
    os.makedirs(out_dir, exist_ok=True)

    # Parse .sh file for URLs
    urls = []
    with open(sh_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("curl") or line.startswith("wget"):
                url = line.split()[-1].strip('"')
                urls.append(url)

    if not urls:
        log("No download URLs found in script.", log_file)
        # print("⚠️ No download URLs found in script.")
        return

    log(f"Found {len(urls)} FITS files in sector {sector_number}", log_file, call_log_main=False)
    # print(f"🌐 Found {len(urls)} FITS files in sector {sector_number}")
    log(f"Downloading {max_downloads if max_downloads is not None else len(urls)} of {len(urls)} FITS files...", log_file, call_log_main=False)

    urls = urls[start_index:]
    if max_downloads is not None:
        urls = urls[:max_downloads]

    def download_single_file(url):
        filename = os.path.join(out_dir, os.path.basename(url))
        if os.path.exists(filename):
            # log(f"✔️ Already exists: {filename}", log_file)
            # print(f"✔️ Already exists: {filename}")
            return None
        try:
            log(f"Downloading {os.path.basename(url)}", log_file, call_log_main=False)
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return os.path.basename(url)
        except Exception as e:
            log(f"Failed to download {url}: {e}", log_file)
            return None

    last_downloaded = None
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(download_single_file, urls),
                            total=len(urls),
                            desc=f"Downloading sector {sector_number}",
                            unit="file"))
        for res in results:
            if res:
                last_downloaded = res

    if last_downloaded:
        log(f"Last successfully downloaded: {last_downloaded}", log_file)
        # print(f"✅ Last successfully downloaded: {last_downloaded}")