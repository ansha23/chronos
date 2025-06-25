import os
import subprocess
from modules.logger import logger

def run_deep_wsclean(config):
    logger.info("📡 Starting Deep WSClean step...")

    section = 'wsclean_deep'

    ms = config.get(section, 'ms', fallback='').strip()
    if not ms:
        ms = config.get('mstransform', 'output_ms', fallback='')
        if not ms:
            input_ms = config.get('uvsub', 'input_ms', fallback='')
            if input_ms:
                ms = input_ms.replace('.ms', '_uvsub.ms')

    if not os.path.exists(ms):
        logger.error(f"❌ Measurement set not found: {ms}")
        return

    output_prefix = config.get(section, 'output_prefix', fallback='').strip()
    if not output_prefix:
        ms_basename = os.path.basename(ms)
        if ms_basename.endswith('.ms'):
            ms_basename = ms_basename[:-3]
        output_prefix = ms_basename + '_uvsubwsc'

    scale = config.get(section, 'scale')
    size = config.get(section, 'size')
    niter = config.get(section, 'niter')
    mgain = config.get(section, 'mgain')
    wsclean_path = config.get('general', 'wsclean_path', fallback='wsclean')

    cmd = [
        wsclean_path,
        '-name', output_prefix,
        '-size', *size.split(','),
        '-scale', scale,
        '-niter', niter,
        '-mgain', mgain,
    ]

    channels_out = config.get(section, 'channels-out', fallback='').strip()
    if channels_out:
        cmd += ['-channels-out', channels_out]

    pol = config.get(section, 'pol', fallback='').strip()
    if pol:
        cmd += ['-pol', pol]

    weight = config.get(section, 'weight', fallback='').strip()
    if weight:
        cmd += ['-weight'] + weight.split() 

    super_weight = config.get(section, 'super-weight', fallback='').strip()
    if super_weight:
        cmd += ['-super-weight', super_weight]

    rank_filter = config.get(section, 'weighting-rank-filter-size', fallback='').strip()
    if rank_filter:
        cmd += ['-weighting-rank-filter-size', rank_filter]

    cmd.append(ms)

    logger.info(f" Running WSClean with command:\n{' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        for line in process.stdout:
            logger.info(line.strip())

        process.stdout.close()
        returncode = process.wait()

        if returncode != 0:
            logger.error(f"❌ WSClean failed with exit code {returncode}")
            raise subprocess.CalledProcessError(returncode, cmd)
        else:
            logger.info("✅ WSClean completed successfully.")

    except FileNotFoundError:
        logger.error(f"❌ WSClean executable not found: '{wsclean_path}'. Check your config or PATH.")
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred during WSClean execution: {e}")

