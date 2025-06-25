import os
from casacore.tables import table
import subprocess
from modules.logger import logger

def split_scans_with_mstransform(config):
    input_ms = config.get('uvsub', 'input_ms').strip()
    output_base_dir = "split_ms"
    os.makedirs(output_base_dir, exist_ok=True)

    tb = table(input_ms, ack=False)
    scan_numbers = sorted(set(tb.getcol("SCAN_NUMBER")))
    tb.close()

    casa_path = config.get('general', 'casa_dir', fallback='casa')

    for scan in scan_numbers:
        scan_dir = os.path.join(output_base_dir, f"scan{scan}")
        os.makedirs(scan_dir, exist_ok=True)

        ms_name = f"{os.path.basename(input_ms).replace('.ms', '')}_scan{scan}.ms"
        output_ms = os.path.join(scan_dir, ms_name)

        if os.path.exists(output_ms):
            logger.info(f"✅ Already exists: {output_ms}")
            continue

        logger.info(f" Splitting scan {scan} → {output_ms}")
        casa_script = f"""
from casatasks import mstransform
mstransform(vis="{input_ms}", outputvis="{output_ms}", datacolumn="corrected", scan="{scan}")
"""
        with open("split_scan_temp.py", "w") as f:
            f.write(casa_script)

        subprocess.run([casa_path, "--nogui", "-c", "split_scan_temp.py"], check=True)
        logger.info(f"✅ Created {output_ms}")

    if os.path.exists("split_scan_temp.py"):
      os.remove("split_scan_temp.py")
      logger.info("Temporary CASA script deleted.")


