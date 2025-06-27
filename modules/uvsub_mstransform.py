import configparser
import subprocess
from modules.logger import logger

def run_uvsub_mstransform_with_casa(config):
    logger.info("📡 Starting CASA uvsub + mstransform step...")

    script_file = 'uvsub_mstransform.py'
    casa_dir = config.get('general', 'casa_dir', fallback='casa')
    preamble = config.get('general', 'preamble', fallback='')

    casa_script = f"""
{preamble}
from casatasks import uvsub, mstransform
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

input_ms = config.get('uvsub', 'input_ms')
scan_param = config.get('mstransform', 'scan', fallback=None)
output_ms = config.get('mstransform', 'output_ms', fallback='').strip()
if not output_ms:
      output_ms = input_ms.replace('.ms','_uvsub.ms')

print(f"🕒 Running uvsub on: {{input_ms}}")
uvsub(vis=input_ms)
print("✅ UVSUB done.")

print(f"🕒 Running mstransform to generate: {{output_ms}}")
mstransform(vis=input_ms,
            outputvis=output_ms,
            datacolumn='corrected',
            scan=scan_param)
print(f"✅ MSTRANSFORM created: {{output_ms}}")
"""

    try:
        with open(script_file, "w") as f:
            f.write(casa_script)
        logger.info(f"CASA script written to: {script_file}")

        casa_cmd = f"{casa_dir} --nogui -c {script_file}"
        logger.info(f" Executing CASA command:\n{casa_cmd}")

        result = subprocess.run(
            casa_cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.stdout:
            logger.info("[CASA STDOUT]")
            for line in result.stdout.strip().splitlines():
                logger.info(line)
        if result.stderr:
            logger.warning("[CASA STDERR]")
            for line in result.stderr.strip().splitlines():
                logger.warning(line)

        logger.info("✅ CASA uvsub + mstransform completed successfully.\n")

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ CASA command failed with exit status {e.returncode}")
        logger.error(e.stderr)
        raise
    except Exception as e:
        logger.exception("❌ Unexpected error in CASA uvsub + mstransform")
        raise

