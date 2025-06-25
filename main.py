import sys
import os
import configparser
import logging
from datetime import datetime
from multiprocessing import Pool

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

    pass

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

        import fnmatch
        split_dir = "split_ms"
        ms_files = []
        for root, dirs, files in os.walk(split_dir):
            for d in dirs:
                if fnmatch.fnmatch(d, "*.ms"):
                    ms_path = os.path.join(root, d)
                    ms_files.append(ms_path)
                    logger.info(f"Found scan MS: {ms_path}")

        ms_files.sort()
        logger.info(f" Found {len(ms_files)} scan .ms directories to process")

        with Pool() as pool:
            pool.starmap(process_single_scan, [(os.path.abspath(ms_path),) for ms_path in ms_files])

        logger.info("✅ pipeline completed!")
        return

    logger.info("⚠️ No scan-based processing requested. Pipeline ended.")
    logger.info("✅ Pipeline completed!")


if __name__ == "__main__":
    main()

