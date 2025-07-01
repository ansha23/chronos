import subprocess
import os
import glob
from modules.logger import logger

def run_pybdsf(config):
    logger.info("📡 Starting PyBDSF step...")

    pybdsf_conf = config['pybdsf']
    fits_file = pybdsf_conf.get('fits_image', '').strip()

    if not fits_file or not os.path.exists(fits_file):
        logger.warning("⚠️ FITS image not provided or not found. Searching for fallback...")

        candidates = glob.glob("*_wsc-MFS-image.fits")
        if candidates:
            fits_file = candidates[0]
            logger.info(f"🔄 Using fallback FITS image: {fits_file}")
        else:
            logger.error("❌ No suitable *_wsc-MFS-image.fits file found.")
            raise FileNotFoundError("FITS image not provided and fallback search failed.")

    pybdsf_script = f"""
inp process_image
filename='{fits_file}'
go
write_catalog
inp export_image
outfile=''
img_type='island_mask'
go
exit
"""

    logger.info(f"🕒 Running PyBDSF on {fits_file}...")

    process = subprocess.Popen(
        ['pybdsf'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(pybdsf_script)

    if process.returncode != 0:
        logger.error("❌ PyBDSF failed with the following error:")
        logger.error(stderr)
    else:
        logger.info("✅ PyBDSF completed successfully.")
        logger.debug(stdout) 
