import subprocess
import os
import glob
from chronos.modules.logger import logger
from casacore.tables import table
import numpy as np

def get_total_scan_duration(ms_path, scan_str=None):
    try:
        tb = table(ms_path, ack=False)
        times = tb.getcol("TIME")
        scan_ids = tb.getcol("SCAN_NUMBER")
        tb.close()

        scan_numbers = list(map(int, scan_str.strip().split())) if scan_str else sorted(set(scan_ids))
        total_duration = sum(np.max(times[scan_ids == scan]) - np.min(times[scan_ids == scan])
                             for scan in scan_numbers if len(times[scan_ids == scan]) > 0)
        return total_duration
    except Exception as e:
        logger.error(f"❌ Failed to compute total scan duration: {e}")
        return None

def run_time_wsclean(config):
    section = 'wsclean_timeseries'
    ms = config.get(section, 'ms', fallback='').strip()

    if not ms:
        ms = config.get('mstransform', 'output_ms', fallback='').strip()

    if not ms:
        candidates = glob.glob("*_uvsub.ms")
        if len(candidates) == 1:
            ms = os.path.abspath(candidates[0])
            logger.warning(f"⚠️ No MS specified. Using detected file: {ms}")
        elif len(candidates) > 1:
            logger.error("❌ Multiple *_uvsub.ms files found. Please specify the correct one in config.")
            raise RuntimeError("Ambiguous MS file. Please set it explicitly.")
        else:
            logger.error("❌ No *_uvsub.ms file found. Please provide an input MS in the config.")
            raise FileNotFoundError("No MS found to run WSClean.")

    if not os.path.exists(ms):
        logger.error(f"❌ Measurement set not found: {ms}")
        raise FileNotFoundError(f"Measurement set not found: {ms}")

    name = config.get(section, 'name', fallback='').strip()
    if not name:
        ms_dir = os.path.dirname(ms)
        ms_basename = os.path.basename(ms).replace('.ms', '')
        name = os.path.join(ms_dir, f"{ms_basename}_wsc")
        logger.info(f"Auto-generated WSClean name: {name}")

    wsclean_path = config.get('general', 'wsclean_path', fallback='wsclean')

    cmd = [
        wsclean_path,
        '-name', name,
        '-weight', 'briggs', '0.0',
        '-super-weight', '1.0',
        '-weighting-rank-filter-size', '16',
        '-taper-gaussian', '0',
        '-join-channels',
        '-no-negative',
        '-fit-beam',
        '-elliptical-beam',
        '-wstack-grid-mode', 'kb',
    ]

    config_keys = {
        'size': ',',
        'scale': ' ',
        'channels-out': ' ',
        'wstack-kernel-size':' ',
        'wstack-oversampling': ' ',
        'pol': ' ',
        'data-column': ' ',
        'niter': ' ',
        'auto-mask': ' ',
        'auto-threshold': ' ',
        'gain': ' ',
        'mgain': ' ',
        'multiscale-scale-bias': ' ',
        'fit-spectral-pol': ' ',
        'padding': ' ',
        'parallel-deconvolution': ' '
    }

    for key, splitter in config_keys.items():
        value = config.get(section, key, fallback='').strip()
        if value:
            cmd.append(f'-{key}')
            cmd.extend(value.split(splitter))

    time_interval = config.get(section, 'time_interval', fallback='').strip()
    scan_param = config.get(section, 'scan', fallback='').strip()
    if time_interval:
        try:
            time_interval = float(time_interval)
            total_duration = get_total_scan_duration(ms, scan_param)
            if total_duration and total_duration > 0:
                intervals_out = int(np.ceil(total_duration / time_interval))
                logger.info(f"🕒 Total scan duration: {total_duration:.2f}s")
                logger.info(f"⏱️ Time interval: {time_interval}s → Intervals-out: {intervals_out}")
                cmd += ['-intervals-out', str(intervals_out)]
            else:
                logger.warning("⚠️ Could not compute intervals-out: invalid scan duration.")
        except ValueError:
            logger.warning(f"⚠️ Invalid time_interval value: {time_interval}")

    cmd.append(ms)

    logger.info("Running WSClean (time series) with command:")
    logger.info(' '.join(cmd))

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=os.path.dirname(ms)
        )
        for line in process.stdout:
            logger.info(line.strip())
        process.stdout.close()
        returncode = process.wait()

        if returncode != 0:
            logger.error(f"❌ WSClean failed with exit code {returncode}")
            raise subprocess.CalledProcessError(returncode, cmd)
        else:
            logger.info("✅ Timeseries-WSClean completed successfully.")

    except FileNotFoundError:
        logger.error(f"❌ WSClean executable not found: '{wsclean_path}'. Check your config or PATH.")
    except Exception as e:
        logger.error(f"❌ Unexpected error during WSClean execution: {e}")

