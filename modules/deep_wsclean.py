import os
import subprocess
from modules.logger import logger
import glob

def run_deep_wsclean(config):
    logger.info("📡 Starting Deep WSClean step...")

    section = 'wsclean_deep'

    ms = config.get(section, 'ms', fallback='').strip()

    if not ms:
        ms = config.get('mstransform', 'output_ms', fallback='').strip()

    if not ms:
        uvsub_candidates = glob.glob("*_uvsub.ms")
        if len(uvsub_candidates) == 1:
            ms = os.path.abspath(uvsub_candidates[0])
            logger.warning(f"⚠️ No MS specified in config. Using detected file: {ms}")
        elif len(uvsub_candidates) > 1:
            logger.error("❌ Multiple *_uvsub.ms files found. Please specify 'ms' explicitly in the config.")
            return
        else:
            logger.error("❌ No *_uvsub.ms file found in the directory. Please check your config or file location.")
            return

    if not os.path.exists(ms):
        logger.error(f"❌ Measurement set not found: {ms}")
        return

    output_prefix = config.get(section, 'output_prefix', fallback='').strip()
    if not output_prefix:
        ms_basename = os.path.basename(ms)
        if ms_basename.endswith('.ms'):
            ms_basename = ms_basename[:-3]
        output_prefix = ms_basename + '_wsc'

    wsclean_path = config.get('general', 'wsclean_path', fallback='wsclean')

    size = config.get(section, 'size')
    scale = config.get(section, 'scale')
    channels_out = config.get(section, 'channels-out')
    kernel_size = config.get(section, 'wstack-kernel-size')
    oversampling = config.get(section, 'wstack-oversampling')
    pol = config.get(section, 'pol')
    data_column = config.get(section, 'data-column')
    niter = config.get(section, 'niter')
    auto_mask = config.get(section, 'auto-mask')
    auto_threshold = config.get(section, 'auto-threshold')
    gain = config.get(section, 'gain')
    mgain = config.get(section, 'mgain')
    scale_bias = config.get(section, 'multiscale-scale-bias')
    spectral_pol = config.get(section, 'fit-spectral-pol')
    padding = config.get(section, 'padding')
    parallel_deconv = config.get(section, 'parallel-deconvolution')

    cmd = [
        wsclean_path,
        '-name', output_prefix,
        '-weight', 'briggs', '0.0',
        '-super-weight', '1.0',
        '-weighting-rank-filter-size', '16',
        '-taper-gaussian', '0',
        '-size', *size.split(','),
        '-scale', scale,
        '-channels-out', channels_out,
        '-wstack-grid-mode', 'kb',
        '-wstack-kernel-size', kernel_size,
        '-wstack-oversampling', oversampling,
        '-pol', pol,
        '-data-column', data_column,
        '-niter', niter,
        '-auto-mask', auto_mask,
        '-auto-threshold', auto_threshold,
        '-gain', gain,
        '-mgain', mgain,
        '-join-channels',
        '-no-negative',
        '-multiscale-scale-bias', scale_bias,
        '-fit-spectral-pol', spectral_pol,
        '-fit-beam',
        '-elliptical-beam',
        '-padding', padding,
        '-parallel-deconvolution', parallel_deconv,
        ms
    ]

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

