import os
import glob
import subprocess
from casacore.tables import table
from modules.logger import logger

def split_scans_with_mstransform(config):
    output_ms = config.get('mstransform', 'output_ms', fallback='').strip()

    if not output_ms:
        candidates = [f for f in glob.glob("*_uvsub.ms") if os.path.isdir(f)]  #will search for directory with that extension
        if len(candidates) == 1:
            output_ms = os.path.abspath(candidates[0])
            logger.warning(f"⚠️ [mstransform][output_ms] not set. Using detected file: {output_ms}")
        elif len(candidates) > 1:
            logger.error("❌ Multiple *_uvsub.ms files found. Please set [mstransform][output_ms] in config.ini explicitly.")
            return
        else:
            logger.error("❌ No *_uvsub.ms file found and [mstransform][output_ms] not set. Cannot continue.")
            return

    if not os.path.exists(output_ms):
        logger.error(f"❌ Cannot split scans: File not found: {output_ms}")
        return

    logger.info(f"Splitting scans from: {output_ms}")

    output_base_dir = "split_ms"
    os.makedirs(output_base_dir, exist_ok=True)

    try:
        tb = table(output_ms, ack=False)
        scan_numbers = sorted(set(tb.getcol("SCAN_NUMBER")))
        tb.close()
    except Exception as e:
        logger.error(f"❌ Failed to read scan numbers from {output_ms}: {e}")
        return

    casa_path = config.get('general', 'casa_dir', fallback='casa')

    for scan in scan_numbers:
        scan_dir = os.path.join(output_base_dir, f"scan{scan}")
        os.makedirs(scan_dir, exist_ok=True)

        ms_name = f"{os.path.basename(output_ms).replace('.ms', '')}_scan{scan}.ms"
        output_scan_ms = os.path.join(scan_dir, ms_name)

        if os.path.exists(output_scan_ms):
            logger.info(f"✅ Already exists: {output_scan_ms}")
            continue

        logger.info(f" Splitting scan {scan} → {output_scan_ms}")

        casa_script = f"""
from casatasks import mstransform
mstransform(vis="{output_ms}", outputvis="{output_scan_ms}", datacolumn="data", scan="{scan}")
"""

        with open("split_scan_temp.py", "w") as f:
            f.write(casa_script)

        try:
            subprocess.run([casa_path, "--nogui", "-c", "split_scan_temp.py"], check=True)
            logger.info(f"✅ Created {output_scan_ms}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ CASA mstransform failed for scan {scan}: {e}")
        finally:
            if os.path.exists("split_scan_temp.py"):
                os.remove("split_scan_temp.py")
                logger.info("Deleted temporary CASA script.")

