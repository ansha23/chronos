import subprocess
import os
from modules.logger import logger
from casacore.tables import table
import numpy as np

def get_total_scan_duration(ms_path, scan_str=None):
    """Compute total duration by summing durations of each selected scan."""
    try:
        tb = table(ms_path, ack=False)
        times = tb.getcol("TIME")
        scan_ids = tb.getcol("SCAN_NUMBER")
        tb.close()

        if scan_str:
            scan_numbers = list(map(int, scan_str.strip().split()))
        else:
            scan_numbers = sorted(set(scan_ids))

        total_duration = 0
        for scan in scan_numbers:
            scan_times = times[scan_ids == scan]
            if len(scan_times) > 0:
                duration = np.max(scan_times) - np.min(scan_times)
                total_duration += duration

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
            input_ms = config.get('uvsub', 'input_ms', fallback='').strip()
            if input_ms:
                ms = input_ms.replace('.ms', '_uvsub.ms')

    if not ms or not os.path.exists(ms):
        logger.error(f"❌ Measurement set not found or not provided: {ms}")
        raise FileNotFoundError(f"Measurement set not found or not provided: {ms}")

    name = config.get(section, 'name', fallback='').strip()
    if not name:
        ms_dir = os.path.dirname(ms)
        ms_basename = os.path.basename(ms).replace('.ms', '')
        name = os.path.join(ms_dir, f"{ms_basename}wsc")
        logger.info(f"Auto-generated WSClean name: {name}")

    wsclean_path = config.get('general', 'wsclean_path', fallback='wsclean')
    cmd = [wsclean_path, '-name', name]

    time_interval = None
    scan_param = None

    for key, value in config.items(section):
        if key in ['ms', 'name']:
            continue

        if key == 'time_interval':
            time_interval = value.strip()
            continue

        if key == 'scan':
            scan_param = value.strip()
            continue

        key_clean = key.strip().replace('_', '-')

        if value.lower() in ['true', 'false']:
            if value.lower() == 'true':
                cmd.append(f"--{key_clean}")
        elif value.strip() == '':
            cmd.append(f"--{key_clean}")
        else:
            parts = value.split()
            cmd.append(f"--{key_clean}")
            cmd.extend(parts)

    if time_interval:
        try:
            time_interval = float(time_interval)
            total_duration = get_total_scan_duration(ms, scan_param)
            if total_duration is not None and total_duration > 0:
                intervals_out = int(np.ceil(total_duration / time_interval))
                logger.info(f"🕒 Total scan duration: {total_duration:.2f}s")
                logger.info(f"⏱️ Time interval: {time_interval}s → Intervals-out: {intervals_out}")
                cmd.append('--intervals-out')
                cmd.append(str(intervals_out))
            else:
                logger.warning("⚠️ Could not compute intervals-out: invalid scan duration.")
        except ValueError:
            logger.warning(f"⚠️ Invalid time_interval value: {time_interval}")

    cmd.append(ms)

    logger.info(" Running WSClean (time series) with command:")
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
        logger.error(f"❌ An unexpected error occurred during WSClean execution: {e}")

