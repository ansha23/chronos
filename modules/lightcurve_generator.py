import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from astropy.wcs.utils import proj_plane_pixel_area
from modules.logger import logger
import glob
import re

def generate_lightcurves_and_detect_transients(config):
    scan_dir = os.getcwd()
    scan_name = os.path.basename(os.path.normpath(scan_dir))

    lc_conf = config['lightcurve_generator']
    search_dir = "."

    catalog_file = lc_conf.get('catalog_file')
    image_template = lc_conf.get('image_template')

    if not catalog_file or not os.path.isfile(catalog_file):
        catalog_matches = glob.glob(os.path.join(search_dir, '*.pybdsf.srl.fits'))
        if catalog_matches:
            catalog_file = catalog_matches[0]
            logger.warning(f"⚠️ Catalog not provided. Using detected catalog: {catalog_file}")
        else:
            raise FileNotFoundError("❌ No catalog file (*.pybdsf.srl.fits) found in current directory.")

    if not image_template:
        image_files = sorted(glob.glob(os.path.join(search_dir, '*-image.fits')))
        if image_files:
            logger.warning(f"⚠️ Image template not provided. Found {len(image_files)} image files ending with '-image.fits'.")
        else:
            raise FileNotFoundError("❌ No image files (*-image.fits) found in current directory.")
    else:

        image_files = [image_template.format(i) for i in range(10000)]
        image_files = [f for f in image_files if os.path.exists(f)]

    n_images = len(image_files)

    if n_images == 0:
        raise RuntimeError("❌ No matching image files found.")

    with fits.open(image_files[0]) as hdul:
        header0 = hdul[0].header
        center_ra = np.mod(header0['CRVAL1'], 360.0)
        center_dec = header0['CRVAL2']
        sky_center = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg)
        logger.info(f"Image center from header: RA = {center_ra:.4f}, Dec = {center_dec:.4f}")

    rms_region = SkyCoord(
        ra=float(lc_conf['rms_ra_deg']) * u.deg,
        dec=float(lc_conf['rms_dec_deg']) * u.deg
    )
    rms_radius_pix = int(lc_conf['rms_radius_pix'])

    catalog = Table.read(catalog_file)
    ras = np.mod(catalog['RA'], 360.0)
    decs = catalog['DEC']
    majs = catalog['Maj']
    mins = catalog['Min']
    pas = catalog['PA']

    catalog_coords = SkyCoord(ra=ras, dec=decs)
    separations = catalog_coords.separation(sky_center)

    n_sources = len(catalog)
    fluxes = np.full((n_sources, n_images), np.nan)
    flux_errors = np.full((n_sources, n_images), np.nan)
    rms_values = np.full(n_images, np.nan)
    obs_times = []

    for i_img, image_file in enumerate(image_files):
        logger.info(f"[{scan_name}] 🕒 Processing image {i_img+1}/{n_images}: {image_file}")

        with fits.open(image_file) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            wcs = WCS(header)
            pixel_area_deg2 = proj_plane_pixel_area(wcs)
            bmaj = header.get('BMAJ')
            bmin = header.get('BMIN')
            beam_area_sr = 1.1331 * bmaj * bmin
            pixel_area_sr = pixel_area_deg2 * (np.pi / 180.0) ** 2
            beam_area_pix = beam_area_sr / pixel_area_sr

            obs_time = header.get('DATE-OBS') or header.get('DATE')
            if not obs_time:
                raise ValueError(f"❌ Observation time not found in {image_file} header.")
            obs_times.append(obs_time)

        data_img = data[0, 0, :, :] if data.ndim == 4 else data

        x_rms, y_rms, _, _ = wcs.world_to_pixel_values(rms_region.ra.deg, rms_region.dec.deg, 0, 0)
        yy, xx = np.ogrid[:data_img.shape[0], :data_img.shape[1]]
        mask_rms = (xx - x_rms) ** 2 + (yy - y_rms) ** 2 <= rms_radius_pix ** 2
        rms = np.nanstd(data_img[mask_rms])
        rms_values[i_img] = rms

        freq_val = 0
        pol_val = 0
        freq_arr = np.full_like(ras, freq_val)
        pol_arr = np.full_like(ras, pol_val)
        x_pix, y_pix, _, _ = wcs.world_to_pixel_values(ras, decs, freq_arr, pol_arr)

        for i_src, (x, y) in enumerate(zip(x_pix, y_pix)):
            a_deg = majs[i_src]
            b_deg = mins[i_src]
            pa_rad = -np.radians(90 + pas[i_src])
            pixel_scale_deg = np.sqrt(pixel_area_deg2)
            a_pix = a_deg / pixel_scale_deg
            b_pix = b_deg / pixel_scale_deg

            x_min = max(int(x - a_pix) - 100, 0)
            x_max = min(int(x + a_pix) + 100, data_img.shape[1] - 1)
            y_min = max(int(y - b_pix) - 100, 0)
            y_max = min(int(y + b_pix) + 100, data_img.shape[0] - 1)

            cutout = data_img[y_min:y_max+1, x_min:x_max+1]
            yy, xx = np.mgrid[y_min:y_max+1, x_min:x_max+1]
            x_shift = xx - x
            y_shift = yy - y
            x_rot = np.cos(pa_rad) * x_shift + np.sin(pa_rad) * y_shift
            y_rot = -np.sin(pa_rad) * x_shift + np.cos(pa_rad) * y_shift
            ellipse_mask = (x_rot / a_pix) ** 2 + (y_rot / b_pix) ** 2 <= 1

            aperture_area = np.sum(ellipse_mask)
            flux_sum = np.nansum(cutout[ellipse_mask])
            flux_Jy = flux_sum * (aperture_area / beam_area_pix)

            N_beams = aperture_area / beam_area_pix
            error = np.sqrt((0.1 * flux_Jy) ** 2 + (N_beams * rms) ** 2)

            fluxes[i_src, i_img] = flux_Jy
            flux_errors[i_src, i_img] = error

    times = Time(obs_times, format='isot')

    flux_colnames = [f'Flux_t{i:04d}' for i in range(n_images)]
    error_colnames = [f'Error_t{i:04d}' for i in range(n_images)]

    coords = SkyCoord(ra=ras, dec=decs)
    ra_hms = coords.ra.to_string(unit=u.hour, sep=':', precision=2, pad=True)
    dec_dms = coords.dec.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True, pad=True)

    data_dict = {
        'source_id': np.arange(n_sources),
        'RA_deg': ras,
        'DEC_deg': decs,
        'RA_hms': ra_hms,
        'DEC_dms': dec_dms,
        'Separation_deg': separations.degree
    }

    for i in range(n_images):
        data_dict[flux_colnames[i]] = fluxes[:, i]
        data_dict[error_colnames[i]] = flux_errors[:, i]
        data_dict[f'RMS_t{i:04d}'] = np.full(n_sources, rms_values[i])

    scan_dir = os.getcwd()
    output_dir = lc_conf.get('output_dir', 'lightcurve_plots')

    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(data_dict)
    scan_name = os.path.basename(os.path.normpath(scan_dir))
    output_csv_path = os.path.join(output_dir, f'{scan_name}_ref_catalog.csv')

    df.to_csv(output_csv_path, index=False)
    logger.info(f"✅ CSV file saved as '{output_csv_path}'")

    for i_src in range(n_sources):
        coord = SkyCoord(ra=ras[i_src]*u.deg, dec=decs[i_src]*u.deg)
        ra_seg = coord.ra.to_string(unit=u.hour, sep='_', precision=2, pad=True)
        dec_seg = coord.dec.to_string(unit=u.deg, sep='_', precision=2, alwayssign=True, pad=True)

        plt.figure(figsize=(8, 5))
        plt.errorbar(times.datetime, fluxes[i_src], yerr=flux_errors[i_src], fmt='o-', capsize=3, label='Flux ± Error')
        plt.xlabel('Observation Time')
        plt.ylabel('Flux (Jy)')
        plt.title(f'Source {i_src} | RA: {ra_seg}, Dec: {dec_seg}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/source_{i_src:03d}_RA{ra_seg}_DEC{dec_seg}.png')
        plt.close()

    logger.info(f'✅ All light curve plots saved in {output_dir}.')
    
    window_size = int(lc_conf.get('window_size', 31))
    k_sigma = float(lc_conf.get('k_sigma', 3))
    min_points = int(lc_conf.get('min_points', 1))
    transient_plot_dir = lc_conf.get('transient_plot_dir', 'transient_detection_plots')
    if not os.path.isabs(transient_plot_dir):
        transient_plot_dir = os.path.join(scan_dir, transient_plot_dir)
    os.makedirs(transient_plot_dir, exist_ok=True)

    transients = detect_transients_mad(
        fluxes=fluxes,
        times=times,
        ra_hms=ra_hms,
        dec_dms=dec_dms,
        separations=separations,
        window_size=window_size,
        k_sigma=k_sigma,
        min_points=min_points,
        plot_dir=transient_plot_dir
    )

    if transients:
        df_trans = pd.DataFrame(transients)
        transient_csv_path = os.path.join(transient_plot_dir, f'{scan_name}_transient_candidates.csv')
        df_trans.to_csv(transient_csv_path, index=False)
        logger.info(f"✅ {len(transients)} transient candidates saved to '{transient_csv_path}'")

    else:
        logger.warning("⚠️ No transient candidates found. Try adjusting parameters.")


def detect_transients_mad(fluxes, times, ra_hms, dec_dms, separations, window_size, k_sigma, min_points, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    half_window = window_size // 2
    n_sources = fluxes.shape[0]
    transient_candidates = []

 
    times_mjd = times.mjd
    times_datetime = times.datetime 

    for i_src in range(n_sources):
        flux_series = fluxes[i_src]
        time_series = times_mjd

        median_line = np.full(flux_series.shape, np.nan, dtype=float)
        mad_line = np.full(flux_series.shape, np.nan, dtype=float)
        transient_mask = np.full(flux_series.shape, False, dtype=bool)

        for i in range(len(flux_series)):
            start = max(0, i - half_window)
            end = min(len(flux_series), i + half_window + 1)

            window_data = flux_series[start:end]
            window_data = window_data[~np.isnan(window_data)]

            if len(window_data) < min_points:
                continue

            running_median = np.median(window_data)
            mad = np.median(np.abs(window_data - running_median)) * 1.4826

            median_line[i] = running_median
            mad_line[i] = mad

            current_flux = flux_series[i]
            if not np.isnan(current_flux):
                lower_thresh = running_median - k_sigma * mad
                upper_thresh = running_median + k_sigma * mad
                if current_flux < lower_thresh or current_flux > upper_thresh:
                    transient_mask[i] = True

        if np.any(transient_mask):
            transient_times = times_datetime[transient_mask]
            transient_fluxes = flux_series[transient_mask]

            transient_candidates.append({
                'source_id': i_src,
                'RA_hms': ra_hms[i_src],
                'DEC_dms': dec_dms[i_src],
                'Separation_deg': separations[i_src].value,
                'n_transient_points': np.sum(transient_mask),
                'transient_times': [t.isoformat() for t in transient_times],
                'transient_fluxes': list(transient_fluxes)
            })

            plt.figure(figsize=(10, 5))
            plt.plot(times_datetime, flux_series, 'k-o', label='Flux')
            plt.plot(times_datetime, median_line, 'b--', label='Running Median')

            valid = ~np.isnan(median_line) & ~np.isnan(mad_line)
            plt.fill_between(np.array(times_datetime)[valid],
                             median_line[valid] - k_sigma * mad_line[valid],
                             median_line[valid] + k_sigma * mad_line[valid],
                             color='cyan', alpha=0.3, label=f'±{k_sigma} MAD')

            plt.scatter(transient_times, transient_fluxes, color='red', s=50, label='Transient Points', zorder=5)
            plt.xlabel('Time')
            plt.ylabel('Flux (Jy)')
            plt.title(f'Source {i_src} - Transient Detection (Median ± {k_sigma}×MAD)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{plot_dir}/source_{i_src:03d}_transient.png')
            plt.close()

    return transient_candidates
    
