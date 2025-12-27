import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from chronos.modules.logger import logger

def concatenate_and_analyze_lightcurves(config):
    concat_conf = config['concatenate_catalogs']

    scan_dirs   = [d.strip() for d in concat_conf.get('scan_dirs', '').split(',')]
    output_dir  = concat_conf.get('output_dir', 'concatenated_lightcurves')
    window_size = int(concat_conf.get('window_size', 31))
    k_sigma     = float(concat_conf.get('k_sigma', 3.0))
    min_points  = int(concat_conf.get('min_points', 1))

    os.makedirs(output_dir, exist_ok=True)
    lc_plot_dir = os.path.join(output_dir, "concatenated_lightcurves")
    trans_plot_dir = os.path.join(output_dir, "transient_detection_plots")
    perio_dir = os.path.join(output_dir, "lombscargle_periodograms")
    for d in (lc_plot_dir, trans_plot_dir, perio_dir):
        os.makedirs(d, exist_ok=True)

    first_cat = None
    for d in scan_dirs:
        f = glob.glob(os.path.join(d, "*ref_catalog.csv"))
        if f:
            logger.info(f"✅ Found catalog: {f[0]}")
            first_cat = pd.read_csv(f[0])
            break
    if first_cat is None:
        logger.error("❌ No <scan>_ref_catalog.csv found in any scan_dirs.")
        return

    sources = first_cat[["source_id", "RA_deg", "DEC_deg", "RA_hms", "DEC_dms"]].copy()
    sources.set_index("source_id", inplace=True)

    global_idx = 0

    for d in sorted(scan_dirs):
        pat = os.path.join(d, "*ref_catalog.csv")
        files = glob.glob(pat)
        if not files:
            logger.warning(f"⚠️ No catalog in {d}")
            continue
        df = pd.read_csv(files[0]).set_index("source_id")

        time_mjd_cols = sorted([c for c in df.columns if c.startswith("Time_t") and c.endswith("_MJD")])
        time_iso_cols = sorted([c for c in df.columns if c.startswith("Time_t") and c.endswith("_ISO")])
        flux_cols     = sorted([c for c in df.columns if c.startswith("Flux_t")])
        err_cols      = sorted([c for c in df.columns if c.startswith("Error_t")])
        rms_cols      = sorted([c for c in df.columns if c.startswith("RMS_t")])

        n = len(time_mjd_cols)
        for j in range(n):
            i = global_idx + j
            sources[f"Time_t{i:04d}_MJD"]  = df[time_mjd_cols[j]]
            sources[f"Time_t{i:04d}_ISO"]  = df[time_iso_cols[j]]
            sources[f"Flux_t{i:04d}"]      = df[flux_cols[j]]
            sources[f"Error_t{i:04d}"]     = df[err_cols[j]]
            sources[f"RMS_t{i:04d}"]       = df[rms_cols[j]]

        global_idx += n

    out_csv = os.path.join(output_dir, "concatenated_ref_catalog.csv")
    sources.reset_index().to_csv(out_csv, index=False)
    logger.info(f"✅ Concatenated catalog saved to {out_csv}")

    times_iso = sources.filter(like="_ISO").values
    times_mjd = sources.filter(like="_MJD").values
    fluxes    = sources.filter(like="Flux_t").values
    errors    = sources.filter(like="Error_t").values

    n_src, n_time = fluxes.shape

    for i_src in range(n_src):
        valid = ~np.isnan(fluxes[i_src])
        if not valid.any():
            continue
        t_iso = times_iso[i_src, valid]
        f     = fluxes[i_src, valid]
        e     = errors[i_src, valid]

        plt.figure(figsize=(8, 5))
        plt.errorbar(t_iso, f, yerr=e, fmt='o-')
        plt.xticks(rotation=30)
        plt.xlabel("Time (ISO)")
        plt.ylabel("Flux (Jy)")
        plt.title(f"Source {sources.index[i_src]} | {sources.iloc[i_src].RA_hms}, {sources.iloc[i_src].DEC_dms}")
        plt.tight_layout()
        plt.savefig(os.path.join(lc_plot_dir, f"source_{sources.index[i_src]:03d}_lc.png"))
        plt.close()

    candidates = []
    half = window_size // 2
    for i_src in range(n_src):
        f = fluxes[i_src]
        t = times_mjd[i_src]
        err = errors[i_src]
        med = np.full(n_time, np.nan)
        mad = np.full(n_time, np.nan)
        mask = np.zeros(n_time, bool)

        for k in range(n_time):
            lo = max(0, k - half)
            hi = min(n_time, k + half + 1)
            w = f[lo:hi][~np.isnan(f[lo:hi])]
            if len(w) < min_points:
                continue
            m = np.median(w)
            M = np.median(np.abs(w - m)) * 1.4826
            med[k] = m
            mad[k] = M
            if not np.isnan(f[k]) and (f[k] > m + k_sigma * M or f[k] < m - k_sigma * M):
                mask[k] = True

        if not mask.any():
            continue

        idxs = np.where(mask)[0]
        candidates.append({
            "source_id": sources.index[i_src],
            "RA_hms": sources.iloc[i_src].RA_hms,
            "DEC_dms": sources.iloc[i_src].DEC_dms,
            "n_transients": int(mask.sum()),
            "times_ISO": list(times_iso[i_src, idxs]),
            "times_MJD": list(times_mjd[i_src, idxs]),
            "fluxes": list(f[mask])
        })

        plt.figure(figsize=(8, 5))
        plt.plot(times_iso[i_src], f, 'k-o', label="Flux")
        plt.plot(times_iso[i_src], med, 'b--', label="Running Median")
        plt.fill_between(times_iso[i_src], med - k_sigma * mad, med + k_sigma * mad,
                         color='cyan', alpha=0.3, label=f"±{k_sigma}×MAD")
        plt.scatter(times_iso[i_src][mask], f[mask], color='red', s=50, label="Transient")
        plt.xticks(rotation=30)
        plt.xlabel("Time (ISO)")
        plt.ylabel("Flux (Jy)")
        plt.legend()
        plt.tight_layout()
        outp = os.path.join(trans_plot_dir, f"source_{sources.index[i_src]:03d}_transient.png")
        plt.savefig(outp)
        plt.close()

    if candidates:
        df_t = pd.DataFrame(candidates)
        df_t.to_csv(os.path.join(output_dir, "transient_candidates_only.csv"), index=False)
        logger.info("✅ Transient candidates saved to transient_candidates_only.csv")
    else:
        logger.info("ℹ️ No transient candidates found")

    periods = []
    for c in candidates:
        src = c["source_id"]
        times = np.array(c["times_MJD"])
        flux = np.array(c["fluxes"])
        try:
            ls = LombScargle(times, flux)
            freq, power = ls.autopower()
            best = 1. / freq[np.argmax(power)]
        except Exception as e:
            logger.warning(f"⚠️ LS failed for {src}: {e}")
            best = np.nan

        plt.figure(figsize=(8, 4))
        plt.plot(1 / freq, power)
        plt.xlabel("Period (days)")
        plt.ylabel("Power")
        plt.title(f"LS Periodogram | Source {src}")
        plt.tight_layout()
        plt.savefig(os.path.join(perio_dir, f"source_{src:03d}_periodogram.png"))
        plt.close()

        periods.append({
            "source_id": src,
            "Best_Period_days": best
        })

    if periods:
        pd.DataFrame(periods).to_csv(os.path.join(output_dir, "transient_candidates_lombscargle.csv"), index=False)
        logger.info("✅ Lomb–Scargle summary saved.")

