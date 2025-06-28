import sys
import os
import configparser
import logging
from datetime import datetime
from multiprocessing import Pool
import re
from casacore.tables import table

sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"logs/pipeline_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("pipeline")

from modules.uvsub_mstransform import run_uvsub_mstransform_with_casa
from modules.deep_wsclean import run_deep_wsclean
from modules.timeseries_wsclean import run_time_wsclean
from modules.pybdsf_runner import run_pybdsf
from modules.lightcurve_generator import generate_lightcurves_and_detect_transients
from modules.scan_splitter import split_scans_with_mstransform
from casacore.tables import table
import numpy as np


def estimate_disk_usage(config):
    section = 'wsclean_timeseries'
    try:
        ms = config.get(section, 'ms', fallback='').strip()
        if not ms or not os.path.exists(ms):
            return 0.0

        nx, ny = map(int, config.get(section, 'size', fallback='8000,8000').split(','))
        chans = int(config.get(section, 'channels-out', fallback='1'))
        pol = config.get(section, 'pol', fallback='I').upper()
        time_interval = float(config.get(section, 'time_interval', fallback='0'))

        tb = table(ms, ack=False)
        times = tb.getcol("TIME")
        scan_ids = tb.getcol("SCAN_NUMBER")
        tb.close()

        scan_str = config.get(section, 'scan', fallback='')
        scan_numbers = list(map(int, scan_str.strip().split())) if scan_str else sorted(set(scan_ids))

        total_duration = sum(
            np.max(times[scan_ids == scan]) - np.min(times[scan_ids == scan])
            for scan in scan_numbers if len(times[scan_ids == scan]) > 0
        )

        n_intervals = int(np.ceil(total_duration / time_interval)) if time_interval > 0 else 1
        n_pol = {'I': 1, 'IQUV': 4, 'RR,LL': 2, 'XX,YY': 2}.get(pol, 1)

        total_images = n_intervals * chans * n_pol
        bytes_per_img = nx * ny * 4  # WSClean outputs FITS files containing images with 32-bit floating-point, 1 byte= 8 bits, 32 bit = 4 bytes
        total_bytes = total_images * bytes_per_img
        total_gb = total_bytes / (1024 ** 3)  # 1kb = 1024 bytes, 1 GB = 1024^3 bytes

        return total_gb

    except Exception as e:
        logger.warning(f"⚠️ Could not estimate WSClean disk usage: {e}")
        return 0.0


def estimate_total_disk_usage(config):
    logger.info("Estimating total disk usage...")

    total_gb = 0.0

    if config.getboolean('modules', 'uvsub_mstransform', fallback=False):
        input_ms = config.get('uvsub', 'input_ms', fallback='').strip()
        if os.path.exists(input_ms):
            size_gb = os.path.getsize(input_ms) / (1024 ** 3)
            est = size_gb * 1
            total_gb += est
            logger.info(f"UVSUB mstransform: ~{est:.2f} GB")

    if config.getboolean('modules', 'deep_wsclean', fallback=False):
        nx, ny = map(int, config.get('deep_wsclean', 'size', fallback='8000,8000').split(','))
        chans = int(config.get('deep_wsclean', 'channels-out', fallback='4'))
        pol = config.get('deep_wsclean', 'pol', fallback='I').upper()
        n_pol = {'I': 1, 'IQUV': 4, 'RR,LL': 2, 'XX,YY': 2}.get(pol, 1)
        bytes_per_img = nx * ny * 4
        est = (bytes_per_img * chans * n_pol) / (1024 ** 3)
        total_gb += est
        logger.info(f"Deep WSClean: ~{est:.2f} GB")

    if config.getboolean('general', 'split_scans', fallback=False):
        input_ms = config.get('uvsub', 'input_ms', fallback='').strip()
        if os.path.exists(input_ms):
            size_gb = os.path.getsize(input_ms) / (1024 ** 3)


            tb = table(input_ms, ack=False)
            scan_numbers = sorted(set(tb.getcol("SCAN_NUMBER")))
            tb.close()

            approx_scans = len(scan_numbers)
            est = size_gb * approx_scans
            total_gb += est
            logger.info(f"Split scans: ~{est:.2f} GB")

    if config.getboolean('modules', 'timeseries_wsclean', fallback=False):
        est = estimate_disk_usage(config)
        total_gb += est
        logger.info(f"Time-series WSClean: ~{est:.2f} GB")

    logger.warning(f"TOTAL ESTIMATED DISK USAGE: {total_gb:.2f} GB\n")


def process_single_scan(ms_path, config_path="config.ini"):
    config = configparser.ConfigParser()
    config.read(config_path)

    base_name = os.path.basename(ms_path).replace('.ms', '')
    ms_dir = os.path.dirname(ms_path)
    scan_name = os.path.basename(ms_dir)

    config.set('wsclean_timeseries', 'ms', ms_path)
    config.set('wsclean_timeseries', 'name', base_name + "_wsc")

    output_dir = config.get('lightcurve_generator', 'output_dir', fallback='lightcurve_plots')
    transient_dir = config.get('lightcurve_generator', 'transient_plot_dir', fallback='transient_detection_plots')

    logger.info(f"[{scan_name}] Starting scan pipeline for: {ms_path}")
    logger.info(f"[{scan_name}] Light curve plots will be saved in: {output_dir}")
    logger.info(f"[{scan_name}] Transient plots will be saved in: {transient_dir}")

    if not config.has_option('lightcurve_generator', 'catalog_file'):
        logger.info(f"[{scan_name}] Catalog file not specified in config. Searching for *.srl.fits...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [f for f in os.listdir(script_dir) if f.endswith('.srl.fits')]
        if not candidates:
            logger.error(f"[{scan_name}] ❌ No catalog file (*.srl.fits) found in: {script_dir}")
            return
        catalog_path = os.path.abspath(os.path.join(script_dir, candidates[0]))
        logger.info(f"[{scan_name}] ✅ Found catalog file: {catalog_path}")
        config.set('lightcurve_generator', 'catalog_file', catalog_path)
        os.environ["CATALOG_FILE"] = catalog_path
    else:
        logger.info(f"[{scan_name}] ✅ Using catalog file from config: {config.get('lightcurve_generator', 'catalog_file')}")

    cwd = os.getcwd()
    try:
        os.chdir(ms_dir)
        logger.info(f"[{scan_name}] Changed directory to scan folder: {os.getcwd()}")

        if config.getboolean('modules', 'timeseries_wsclean', fallback=False):
            estimate_disk_usage(config)
            run_time_wsclean(config)

        if config.getboolean('modules', 'lightcurve_generator', fallback=False):
            generate_lightcurves_and_detect_transients(config)

        logger.info(f"[{scan_name}] ✅ Completed scan pipeline for: {ms_path}")
    except Exception as e:
        logger.error(f"[{scan_name}] ❌ Error while processing {ms_path}: {e}")
    finally:
        os.chdir(cwd)


def main():
    config = configparser.ConfigParser()
    config.read("config.ini")

    logger.info("📡 Starting radio transient pipeline")

    estimate_total_disk_usage(config)

    if config.getboolean('modules', 'uvsub_mstransform', fallback=False):
        logger.info("📡 Step 1: CASA uvsub + mstransform")
        run_uvsub_mstransform_with_casa(config)

    if config.getboolean('modules', 'deep_wsclean', fallback=False):
        logger.info("📡 Step 2: Running deep WSClean")
        run_deep_wsclean(config)

    if config.getboolean('modules', 'pybdsf', fallback=False):
        logger.info("📡 Step 3: Running PyBDSF source detection")
        run_pybdsf(config)

    if config.getboolean('general', 'split_scans', fallback=False):
        logger.info("📡 Step 4: Splitting scans using mstransform...")
        split_scans_with_mstransform(config)

        logger.info("📡 Step 5: Preparing scan-based processing...")

        scan_str = config.get('wsclean_timeseries', 'scan', fallback='').strip()
        allowed_scans = list(map(int, scan_str.split())) if scan_str else None

        split_dir = "split_ms"
        ms_files = []

        for scan_folder in sorted(os.listdir(split_dir)):
            match = re.match(r"scan(\d+)", scan_folder)
            if match:
                scan_num = int(match.group(1))
                if (allowed_scans is None) or (scan_num in allowed_scans):
                    full_dir = os.path.join(split_dir, scan_folder)
                    for fname in os.listdir(full_dir):
                        if fname.endswith(".ms"):
                            ms_path = os.path.join(full_dir, fname)
                            ms_files.append(ms_path)
                            logger.info(f"✅ Selected scan {scan_num}: {ms_path}")

        ms_files.sort()
        logger.info(f" Found {len(ms_files)} scan .ms files to process")

        if config.has_option('general', 'max_parallel_scans'):
            val = config.get('general', 'max_parallel_scans').strip()
            if val:
                max_processes = int(val)
                logger.info(f"Using max {max_processes} parallel processes from config")
            else:
                max_processes = len(ms_files)
                logger.info(f"'max_parallel_scans' is empty. Using all {max_processes} scans in parallel")
        else:
            max_processes = len(ms_files)
            logger.info(f"No 'max_parallel_scans' set. Using all {max_processes} scans in parallel")

        with Pool(processes=max_processes) as pool:
            pool.starmap(process_single_scan, [(os.path.abspath(ms_path),) for ms_path in ms_files])

        logger.info("✅ Pipeline completed!")
        return

    logger.info("⚠️ No scan-based processing requested. Pipeline ended.")
    logger.info("✅ Pipeline completed!")


if __name__ == "__main__":
    main()

