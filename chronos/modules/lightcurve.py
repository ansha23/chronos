#!/usr/bin/env python3
"""
Light curve extraction and transient detection pipeline.
Version: 3.1 - Added input directory support
"""

# Import necessary libraries or packages
import os
import sys
import glob
import configparser
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from scipy.ndimage import convolve1d
import warnings
warnings.filterwarnings('ignore')

# Configure logging
def setup_logging():
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

def read_config(config_file='config.ini'):
    """
    Read configuration from config.ini file.
    
    We will take the following from config.ini:
    input_dir, catalog_file, image_template, rms_ra_deg, rms_dec_deg, 
    rms_radius_pix, output_dir, transient_plot_dir, filter_bank, SNR_threshold.
    """
    if not os.path.exists(config_file):
        logger.error(f"❌ Config file '{config_file}' not found!")
        raise FileNotFoundError(f"Config file '{config_file}' not found!")
    
    config = configparser.ConfigParser()
    config.read(config_file)
    
    if 'lightcurve' not in config:
        raise ValueError("❌ [lightcurve] section not found in config.ini")
    
    lc_conf = config['lightcurve']
    
    # Get RMS values with safe conversion
    rms_ra_deg_str = lc_conf.get('rms_ra_deg', '')
    rms_dec_deg_str = lc_conf.get('rms_dec_deg', '')
    
    return {
        'input_dir': lc_conf.get('input_dir', ''),  # New: Input directory for images
        'catalog_file': lc_conf.get('catalog_file', ''),
        'image_template': lc_conf.get('image_template', ''),
        'rms_ra_deg': rms_ra_deg_str,  # Keep as string for now
        'rms_dec_deg': rms_dec_deg_str,  # Keep as string for now
        'rms_radius_pix': int(lc_conf.get('rms_radius_pix', '50')),
        'output_dir': lc_conf.get('output_dir', 'lightcurve_plots'),
        'transient_plot_dir': lc_conf.get('transient_plot_dir', 'transient_detection_plots'),
        'filter_bank': lc_conf.get('filter_bank', 'auto'),
        'SNR_threshold': float(lc_conf.get('SNR_threshold', '5.0'))
    }

def search_for_catalog(catalog_file):
    """
    Search for a catalog file if not provided or not found.
    """
    if catalog_file and os.path.isfile(catalog_file):
        logger.info(f"✅ Using provided catalog: {catalog_file}")
        return os.path.abspath(catalog_file)
    
    # Search for catalog file
    candidates = glob.glob('./*.pybdsf.srl.fits')
    
    if not candidates:
        candidates = glob.glob('../*.pybdsf.srl.fits')
    
    if not candidates:
        candidates = glob.glob('../../*.pybdsf.srl.fits')
    
    if candidates:
        catalog_file = os.path.abspath(candidates[0])
        logger.warning(f"⚠️ Catalog not provided. Using detected catalog: {catalog_file}")
        return catalog_file
    else:
        raise FileNotFoundError("❌ No catalog file (*.pybdsf.srl.fits) found in scan, parent, or root directory.")

def load_image_files(input_dir, image_template, search_dir='.'):
    """
    DATA LOADER: Load image files from input directory based on template or auto-search.
    """
    # Use input_dir if specified, otherwise use search_dir
    base_dir = input_dir if input_dir else search_dir
    
    # Ensure the directory exists
    if input_dir and not os.path.exists(input_dir):
        logger.warning(f"⚠️ Input directory '{input_dir}' not found. Searching in current directory.")
        base_dir = search_dir
    
    logger.info(f"🔍 Searching for images in: {os.path.abspath(base_dir)}")
    
    if not image_template:
        # Auto-search for image files
        image_files = sorted(glob.glob(os.path.join(base_dir, '*-image.fits')))
        if not image_files:
            # Try other common patterns
            image_files = sorted(glob.glob(os.path.join(base_dir, '*-t*-image.fits')))
        if not image_files:
            image_files = sorted(glob.glob(os.path.join(base_dir, '*-image_*.fits')))
        if not image_files:
            image_files = sorted(glob.glob(os.path.join(base_dir, '*.fits')))
        
        if image_files:
            logger.warning(f"⚠️ Image template not provided. Found {len(image_files)} FITS files.")
        else:
            raise FileNotFoundError(f"❌ No FITS image files found in {base_dir}")
    else:
        # Use template
        if input_dir:
            # Template with input directory
            image_files = [os.path.join(input_dir, image_template.format(i)) for i in range(10000)]
        else:
            # Template without input directory
            image_files = [image_template.format(i) for i in range(10000)]
        
        image_files = [f for f in image_files if os.path.exists(f)]
        if image_files:
            logger.info(f"✅ Using image template: {image_template}")
    
    n_images = len(image_files)
    
    if n_images == 0:
        raise RuntimeError("❌ No matching image files found.")
    
    logger.info(f"📊 Found {n_images} image files")
    return image_files, n_images

def arrange_images_by_time(image_files):
    """
    Arrange images according to time in ascending order,
    refer fits header for original timestamps of images.
    """
    times = []
    valid_files = []
    
    for img in image_files:
        try:
            with fits.open(img) as hdul:
                header = hdul[0].header
                obs_time = header.get('DATE-OBS') or header.get('DATE')
                
                if obs_time:
                    try:
                        t = Time(obs_time, format='isot')
                        times.append(t)
                        valid_files.append(img)
                    except:
                        logger.warning(f"⚠️ Could not parse time for {img}, using file order")
                        times.append(Time(datetime.now(), format='datetime'))
                        valid_files.append(img)
                else:
                    logger.warning(f"⚠️ No observation time in header for {img}")
                    times.append(Time(datetime.now(), format='datetime'))
                    valid_files.append(img)
        except Exception as e:
            logger.warning(f"⚠️ Could not read {img}: {e}")
            continue
    
    if not valid_files:
        return image_files, [Time(datetime.now(), format='datetime')] * len(image_files)
    
    # Sort by time
    sorted_indices = np.argsort([t.mjd for t in times])
    sorted_images = [valid_files[i] for i in sorted_indices]
    sorted_times = [times[i] for i in sorted_indices]
    
    logger.info(f"📊 Arranged {len(sorted_images)} images in time-ascending order")
    return sorted_images, sorted_times

def generate_filter_bank(filter_bank_input):
    """
    Take the filter bank from config.ini file, this is a list of filters,
    defined in minutes, it is user defined in the config file.
    Or if this config file says auto, generate the filter bank inside the code,
    it will contain templates of various widths separated logarithmically.
    Display the filter bank on terminal screen.
    """
    if filter_bank_input.lower().strip() == 'auto':
        # Generate logarithmic spaced filter widths
        min_width =  0.1 # 0.5 minutes (30 seconds)
        max_width = 20  # 60 minutes
        num_filters = 30
        filter_bank = np.logspace(np.log10(min_width), np.log10(max_width), num_filters)
        logger.info("🔄 Auto-generated logarithmic filter bank")
    else:
        # Parse comma-separated list from config (can include floats)
        try:
            filter_bank = [float(x.strip()) for x in filter_bank_input.split(',')]
            logger.info("📋 Using user-defined filter bank")
        except Exception as e:
            logger.warning(f"⚠️ Could not parse filter bank: {e}, using defaults")
            filter_bank = [0.1, 1, 5, 15, 30, 60, 120, 240, 480, 960]
    
    # Remove duplicates and sort
    filter_bank = np.unique(filter_bank)
    filter_bank = sorted(filter_bank)
    
    # Display filter bank on terminal screen
    print("\n" + "="*60)
    print("FILTER BANK CONFIGURATION")
    print("="*60)
    print(f"Number of filters: {len(filter_bank)}")
    print(f"Filter widths (minutes):")
    for i, width in enumerate(filter_bank):
        if width < 1:
            print(f"  Filter {i+1:2d}: {width:.3f} min ({width*60:.1f} sec)")
        elif width < 60:
            print(f"  Filter {i+1:2d}: {width:.1f} min")
        else:
            hours = width / 60
            print(f"  Filter {i+1:2d}: {width:.1f} min ({hours:.1f} hours)")
    print("="*60 + "\n")
    
    logger.info(f"📊 Filter bank: {len(filter_bank)} filters from {filter_bank[0]:.3f} to {filter_bank[-1]:.1f} minutes")
    return filter_bank

def apply_filters_to_lightcurve(fluxes, times_mjd, filter_bank_minutes, SNR_threshold):
    """
    Apply each filter from the filter bank along the time axis 
    of the lightcurve of each source.
    """
    if len(fluxes) < 3:
        return [], None, 0.0, None, []
    
    # Remove NaN values
    valid_mask = ~np.isnan(fluxes)
    if np.sum(valid_mask) < 3:
        return [], None, 0.0, None, []
    
    flux_clean = fluxes[valid_mask]
    time_clean = times_mjd[valid_mask]
    
    # Convert times to minutes from first observation
    times_minutes = (time_clean - time_clean[0]) * 24 * 60
    
    best_snr = 0.0
    best_filter = None
    best_filtered_flux = None
    results = []
    detected_filters_data = []  # Store data for all filters that detected the transient
    
    for filter_width in filter_bank_minutes:
        # Skip if filter is too wide for data
        if filter_width > (times_minutes[-1] - times_minutes[0]) * 0.5:
            continue
        
        # Convert filter width from minutes to data points
        if len(times_minutes) > 1:
            avg_time_step = np.median(np.diff(times_minutes))
            if avg_time_step <= 0:
                avg_time_step = 1.0
        else:
            avg_time_step = 1.0
        
        width_points = max(1, int(filter_width / avg_time_step))
        width_points = min(width_points, len(flux_clean) // 3)  # Don't use too wide filters
        
        if width_points < 1:
            continue
        
        # Create boxcar filter
        filter_kernel = np.ones(width_points) / width_points
        
        try:
            # Apply filter
            filtered = convolve1d(flux_clean, filter_kernel, mode='reflect')
            
            # Calculate SNR using median absolute deviation (robust to outliers)
            residuals = flux_clean - filtered
            
            # Use median absolute deviation (MAD) as robust noise estimator
            # MAD = median(|x - median(x)|)
            # For normally distributed data, σ ≈ 1.4826 * MAD
            if len(residuals) > 1:
                mad = np.median(np.abs(residuals - np.median(residuals)))
                noise = 1.4826 * mad if mad > 0 else 1.0
            else:
                noise = 1.0
            
            if noise > 0:
                # Use maximum absolute deviation from median as signal
                median_flux = np.median(filtered)
                signal = np.max(np.abs(filtered - median_flux))
                snr = signal / noise
            else:
                snr = 0.0
            
            detected = snr >= SNR_threshold
            
            # Map filtered data back to original time grid
            filtered_full = np.full_like(fluxes, np.nan)
            filtered_full[valid_mask] = filtered
            
            results.append({
                'filter_width_minutes': filter_width,
                'width_points': width_points,
                'max_snr': float(snr),
                'detected': bool(detected),
                'noise_level': float(noise) if noise > 0 else 0.0,
                'median_flux': float(np.median(filtered))  # Changed from mean to median
            })
            
            # Store data for detected filters
            if detected:
                detected_filters_data.append({
                    'filter_width': filter_width,
                    'snr': snr,
                    'filtered_data': filtered_full
                })
            
            if snr > best_snr:
                best_snr = snr
                best_filter = filter_width
                best_filtered_flux = filtered_full
                
        except Exception as e:
            logger.debug(f"Filter {filter_width} min failed: {e}")
            continue
    
    return results, best_filter, best_snr, best_filtered_flux, detected_filters_data

def plot_transient_lightcurve_with_all_filters(source_id, ra_str, dec_str, times, fluxes, errors, 
                                             best_filter, best_snr, best_filtered_flux, 
                                             detected_filters_data, output_dir, SNR_threshold):
    """
    Plot the transient with ALL filters that detected it.
    """
    plt.figure(figsize=(14, 8))
    
    # Color palette for filters
    colors = plt.cm.rainbow(np.linspace(0, 1, len(detected_filters_data))) if detected_filters_data else ['red']
    
    # Plot original data with errors
    plt.errorbar(times.datetime, fluxes, yerr=errors, fmt='o-', 
                 capsize=4, label='Original Flux ± Error', alpha=0.8, 
                 color='black', markersize=6, linewidth=1.5, elinewidth=1, zorder=10)
    
    # Plot each detected filter
    for idx, filter_data in enumerate(detected_filters_data):
        filter_width = filter_data['filter_width']
        filter_snr = filter_data['snr']
        filtered_flux = filter_data['filtered_data']
        
        valid_mask = ~np.isnan(filtered_flux)
        if np.any(valid_mask):
            color = colors[idx] if len(detected_filters_data) > 1 else 'red'
            linewidth = 3.0 if filter_width == best_filter else 2.0
            linestyle = '-' if filter_width == best_filter else '--'
            alpha = 0.9 if filter_width == best_filter else 0.7
            
            label = (f'{filter_width:.1f} min (SNR: {filter_snr:.2f})' + 
                    (' [BEST]' if filter_width == best_filter else ''))
            
            plt.plot(np.array(times.datetime)[valid_mask], filtered_flux[valid_mask], 
                    color=color, linewidth=linewidth, linestyle=linestyle,
                    label=label, alpha=alpha, zorder=5-idx)
    
    plt.xlabel('Observation Time', fontsize=14)
    plt.ylabel('Flux (Jy)', fontsize=14)
    
    # Create title with source info
    title = f'Source {source_id} | RA: {ra_str}, Dec: {dec_str}'
    if best_filter is not None:
        title += f'\nBest Filter: {best_filter:.1f} min | Max SNR: {best_snr:.2f}'
    title += f'\nDetected by {len(detected_filters_data)} filter(s) at SNR ≥ {SNR_threshold}'
    
    plt.title(title, fontsize=15, pad=15)
    
    plt.grid(True, alpha=0.3, linestyle='--', zorder=0)
    
    # Adjust legend based on number of filters
    if len(detected_filters_data) <= 6:
        plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
    else:
        # For many filters, use two columns or outside plot
        plt.legend(loc='upper left', fontsize=10, framealpha=0.9, ncol=2)
    
    plt.xticks(rotation=30, fontsize=11)
    plt.yticks(fontsize=11)
    
    # Add SNR threshold annotation
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5, zorder=0)
    
    # Add text box with filter summary
    filter_summary = f"Total detected filters: {len(detected_filters_data)}\n"
    
    plt.figtext(0.02, 0.02, filter_summary, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for text box
    
    # Create filename
    ra_clean = ra_str.replace(':', '_').replace('.', 'p')
    dec_clean = dec_str.replace(':', '_').replace('.', 'p').replace('+', 'p').replace('-', 'm')
    filename = f'source_{source_id:04d}_RA{ra_clean}_DEC{dec_clean}_all_filters.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug(f"📊 Saved multi-filter transient plot: {filename}")

def run_lightcurve(config_dict=None, config_file='config.ini'):
    """
    Run lightcurve pipeline with optional config dictionary.
    Can be called from other scripts.
    """
    import tempfile
    
    # If config_dict is provided, write it to a temporary config file
    if config_dict is not None:
        import configparser
        
        # Create a ConfigParser object
        temp_config = configparser.ConfigParser()
        
        # Add sections and values from config_dict
        for section, values in config_dict.items():
            if not temp_config.has_section(section):
                temp_config.add_section(section)
            for key, value in values.items():
                temp_config.set(section, key, str(value))
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            temp_config.write(f)
            temp_config_path = f.name
        
        # Use the temporary config file
        original_config = config_file
        config_file = temp_config_path
    else:
        temp_config_path = None
    
    try:
        # Store original config file for main() to use
        import lightcurve as lc_module
        lc_module.CONFIG_FILE = config_file
        
        # Run main
        main()
    finally:
        # Clean up temporary config file if created
        if temp_config_path and os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def main():
    """Main pipeline function."""
    logger.info("="*70)
    logger.info("🚀 STARTING LIGHT CURVE EXTRACTION AND TRANSIENT DETECTION PIPELINE")
    logger.info("="*70)
    
    start_time = datetime.now()
    
    try:
        # Read configuration
        logger.info("📖 Reading configuration from config.ini...")
        config = read_config()
        
        # Search for catalog file
        logger.info("🔍 Searching for catalog file...")
        catalog_file = search_for_catalog(config['catalog_file'])
        
        # Load image files from input directory
        logger.info("🖼️  Loading image files...")
        image_files, n_images = load_image_files(
            config['input_dir'], 
            config['image_template']
        )
        logger.info(f"📊 Found {n_images} image files")
        
        # Arrange images according to time in ascending order
        logger.info("⏰ Arranging images by observation time...")
        image_files, obs_times_list = arrange_images_by_time(image_files)
        
        # Get image center from first image
        with fits.open(image_files[0]) as hdul:
            header0 = hdul[0].header
            center_ra = np.mod(header0['CRVAL1'], 360.0)
            center_dec = header0['CRVAL2']
            sky_center = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg)
            logger.info(f"📍 Image center from header: RA = {center_ra:.4f}, Dec = {center_dec:.4f}")
        
        # Setup RMS region - check if values are provided in config, otherwise use image center
        rms_ra_str = config['rms_ra_deg']
        rms_dec_str = config['rms_dec_deg']
        
        if rms_ra_str.strip() and rms_dec_str.strip():
            # Use values from config
            try:
                rms_ra = float(rms_ra_str)
                rms_dec = float(rms_dec_str)
                logger.info(f"📍 Using RMS region coordinates from config: RA = {rms_ra:.4f}, Dec = {rms_dec:.4f}")
            except ValueError:
                logger.warning("⚠️ Could not parse RMS coordinates from config, using image center")
                rms_ra = center_ra
                rms_dec = center_dec
        else:
            # Use image center from header
            rms_ra = center_ra
            rms_dec = center_dec
            logger.info(f"📍 RMS region not specified in config, using image center: RA = {rms_ra:.4f}, Dec = {rms_dec:.4f}")
        
        rms_region = SkyCoord(
            ra=rms_ra * u.deg,
            dec=rms_dec * u.deg
        )
        rms_radius_pix = config['rms_radius_pix']
        
        # Read catalog
        logger.info(f"📚 Reading catalog: {os.path.basename(catalog_file)}")
        catalog = Table.read(catalog_file)
        
        # Extract source parameters
        ras = np.mod(catalog['RA'], 360.0)
        decs = catalog['DEC']
        majs = catalog['Maj']
        mins = catalog['Min']
        pas = catalog['PA']
        
        catalog_coords = SkyCoord(ra=ras, dec=decs)
        separations = catalog_coords.separation(sky_center)
        
        # Display sources info
        n_sources = len(catalog)
        logger.info(f"📊 Found {n_sources} sources in catalog")
        
        # Generate filter bank
        logger.info("🔄 Generating filter bank...")
        filter_bank = generate_filter_bank(config['filter_bank'])
        SNR_threshold = config['SNR_threshold']
        logger.info(f"📊 SNR threshold for detection: {SNR_threshold}")
        
        # Initialize arrays
        logger.info("💾 Initializing data arrays...")
        fluxes = np.full((n_sources, n_images), np.nan)
        flux_errors = np.full((n_sources, n_images), np.nan)
        rms_values = np.full(n_images, np.nan)
        obs_times = []
        
        # Process each image
        logger.info("⚙️  Processing images and extracting fluxes...")
        for i_img, image_file in enumerate(image_files):
            #logger.info(f"  [{i_img+1}/{n_images}] Processing: {os.path.basename(image_file)}")
            
            try:
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
                        logger.warning(f"⚠️ Observation time not found in {image_file} header, using placeholder")
                        obs_time = f"unknown_{i_img}"
                    obs_times.append(obs_time)
                
                data_img = data[0, 0, :, :] if data.ndim == 4 else data
                
                # Calculate RMS in specified region
                x_rms, y_rms, _, _ = wcs.world_to_pixel_values(rms_region.ra.deg, rms_region.dec.deg, 0, 0)
                yy, xx = np.ogrid[:data_img.shape[0], :data_img.shape[1]]
                mask_rms = (xx - x_rms) ** 2 + (yy - y_rms) ** 2 <= rms_radius_pix ** 2
                rms = np.nanstd(data_img[mask_rms])
                rms_values[i_img] = rms
                
                # Convert source positions to pixels
                freq_val = 0
                pol_val = 0
                freq_arr = np.full_like(ras, freq_val)
                pol_arr = np.full_like(ras, pol_val)
                x_pix, y_pix, _, _ = wcs.world_to_pixel_values(ras, decs, freq_arr, pol_arr)
                
                # Extract fluxes for each source
                for i_src, (x, y) in enumerate(zip(x_pix, y_pix)):
                    a_deg = majs[i_src]
                    b_deg = mins[i_src]
                    pa_rad = -np.radians(90 + pas[i_src])
                    pixel_scale_deg = np.sqrt(pixel_area_deg2)
                    a_pix = a_deg / pixel_scale_deg
                    b_pix = b_deg / pixel_scale_deg
                    
                    # Define extraction region
                    x_min = max(int(x - a_pix) - 100, 0)
                    x_max = min(int(x + a_pix) + 100, data_img.shape[1] - 1)
                    y_min = max(int(y - b_pix) - 100, 0)
                    y_max = min(int(y + b_pix) + 100, data_img.shape[0] - 1)
                    
                    cutout = data_img[y_min:y_max+1, x_min:x_max+1]
                    yy_local, xx_local = np.mgrid[y_min:y_max+1, x_min:x_max+1]
                    x_shift = xx_local - x
                    y_shift = yy_local - y
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
                    
            except Exception as e:
                logger.error(f"❌ Error processing {image_file}: {e}")
                continue
        
        logger.info(f"✅ Completed flux extraction for all {n_images} images")
        
        # Convert observation times
        try:
            times = Time(obs_times, format='isot')
        except:
            # If time format parsing fails, create artificial times
            logger.warning("⚠️ Could not parse observation times, using sequential times")
            times = Time(np.arange(n_images), format='mjd')
        
        times_iso = times.isot
        times_mjd = times.mjd
        
        # Create output directories
        output_dir = config['output_dir']
        transient_plot_dir = config['transient_plot_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(transient_plot_dir, exist_ok=True)
        
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"📁 Transient plot directory: {transient_plot_dir}")
        
        # Prepare coordinate strings
        coords = SkyCoord(ra=ras, dec=decs)
        ra_hms_all = coords.ra.to_string(unit=u.hour, sep=':', precision=2, pad=True)
        dec_dms_all = coords.dec.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True, pad=True)
        
        # Apply each filter from the filter bank along the time axis of each source
        logger.info("🔍 Applying filters to detect transients...")
        all_filter_results = []
        transient_detections = []
        
        for i_src in range(n_sources):
            #if i_src % max(1, n_sources // 20) == 0:  # Progress every 5%
               # logger.info(f"  Processing source {i_src+1}/{n_sources}")
            
            source_fluxes = fluxes[i_src, :]
            source_errors = flux_errors[i_src, :]
            
            # Skip sources with insufficient data
            valid_points = np.sum(~np.isnan(source_fluxes))
            if valid_points < 3:
                continue
            
            # Apply filters to this source's lightcurve
            filter_results, best_filter, best_snr, best_filtered_flux, detected_filters_data = apply_filters_to_lightcurve(
                source_fluxes, times_mjd, filter_bank, SNR_threshold
            )
            
            # Store all filter results for CSV output
            for res in filter_results:
                all_filter_results.append({
                    'source_id': i_src,
                    'RA_deg': float(ras[i_src]),
                    'DEC_deg': float(decs[i_src]),
                    'RA_hms': ra_hms_all[i_src],
                    'DEC_dms': dec_dms_all[i_src],
                    'separation_deg': float(separations[i_src].degree),
                    'filter_width_minutes': res['filter_width_minutes'],
                    'snr': res['max_snr'],
                    'detected': res['detected'],
                    'n_valid_points': int(valid_points)
                })
            
            # Check if transient detected with any filter
            if detected_filters_data:  # If we have any detected filters
                # Plot the transient lightcurve with ALL detected filters
                plot_transient_lightcurve_with_all_filters(
                    i_src, ra_hms_all[i_src], dec_dms_all[i_src],
                    times, source_fluxes, source_errors,
                    best_filter, best_snr, best_filtered_flux, 
                    detected_filters_data, transient_plot_dir, SNR_threshold
                )
                
                # Store detection info
                transient_detections.append({
                    'source_id': i_src,
                    'RA_deg': float(ras[i_src]),
                    'DEC_deg': float(decs[i_src]),
                    'RA_hms': ra_hms_all[i_src],
                    'DEC_dms': dec_dms_all[i_src],
                    'separation_deg': float(separations[i_src].degree),
                    'best_filter_minutes': float(best_filter) if best_filter else -1,
                    'max_snr': float(best_snr),
                    'detected_filter_count': len(detected_filters_data),
                    'detected_filters': ','.join([str(f['filter_width']) for f in detected_filters_data]),
                    'detected_snrs': ','.join([str(f['snr']) for f in detected_filters_data]),
                    'snr_threshold': float(SNR_threshold),
                    'detected': True,
                    'n_valid_points': int(valid_points),
                    'median_flux': float(np.nanmedian(source_fluxes)),  # Changed from mean to median
                    'mad_flux': float(np.nanmedian(np.abs(source_fluxes - np.nanmedian(source_fluxes))))  # Changed std to MAD
                })
        
        # Generate a .csv file and store the source info, filter info,
        # the filter that successfully detected the transient source and with how much SNR.
        if all_filter_results:
            df_all_results = pd.DataFrame(all_filter_results)
            filter_csv_path = os.path.join(output_dir, 'all_filter_results.csv')
            df_all_results.to_csv(filter_csv_path, index=False)
            logger.info(f"✅ Saved all filter results to {filter_csv_path}")
        
        if transient_detections:
            df_transients = pd.DataFrame(transient_detections)
            transient_csv_path = os.path.join(transient_plot_dir, 'transient_detections.csv')
            df_transients.to_csv(transient_csv_path, index=False)
            
            # Display summary
            print("\n" + "="*70)
            print("TRANSIENT DETECTION RESULTS")
            print("="*70)
            print(f"Found {len(transient_detections)} transient sources (SNR ≥ {SNR_threshold}):")
            print("-"*70)
            for i, td in enumerate(transient_detections[:10]):  # Show first 10
                print(f"{i+1:3d}. Source {td['source_id']:4d} | "
                      f"RA: {td['RA_hms']:12s} | Dec: {td['DEC_dms']:12s} | "
                      f"SNR: {td['max_snr']:6.2f} | Filter: {td['best_filter_minutes']:7.1f} min | "
                      f"Detected by: {td['detected_filter_count']:2d} filters")
            if len(transient_detections) > 10:
                print(f"... and {len(transient_detections) - 10} more transients")
            print("="*70 + "\n")
            
            logger.info(f"✅ Found {len(transient_detections)} transient candidates!")
            logger.info(f"✅ Saved transient detections to {transient_csv_path}")
            logger.info(f"✅ Saved {len(transient_detections)} transient plots to {transient_plot_dir}")
        else:
            logger.warning(f"⚠️ No transient sources detected above SNR threshold ({SNR_threshold})")
        
        # Save full light curve data
        logger.info("💾 Saving full light curve data...")
        flux_colnames = [f'Flux_t{i:04d}' for i in range(n_images)]
        error_colnames = [f'Error_t{i:04d}' for i in range(n_images)]
        
        data_dict = {
            'source_id': np.arange(n_sources),
            'RA_deg': ras,
            'DEC_deg': decs,
            'RA_hms': ra_hms_all,
            'DEC_dms': dec_dms_all,
            'Separation_deg': separations.degree
        }
        
        for i in range(n_images):
            data_dict[flux_colnames[i]] = fluxes[:, i]
            data_dict[error_colnames[i]] = flux_errors[:, i]
            data_dict[f'RMS_t{i:04d}'] = np.full(n_sources, rms_values[i])
            data_dict[f'Time_t{i:04d}_MJD'] = np.full(n_sources, times_mjd[i])
            data_dict[f'Time_t{i:04d}_ISO'] = np.full(n_sources, times_iso[i])
        
        # Save to CSV
        scan_dir = os.getcwd()
        df = pd.DataFrame(data_dict)
        scan_name = os.path.basename(os.path.normpath(scan_dir))
        output_csv_path = os.path.join(output_dir, f'{scan_name}_lightcurves.csv')
        df.to_csv(output_csv_path, index=False)
        logger.info(f"✅ CSV file saved as '{output_csv_path}'")
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = end_time - start_time
        
        logger.info("="*70)
        logger.info(f"✅ PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"📊 Processed {n_sources} sources across {n_images} images")
        logger.info(f"⏱️  Total processing time: {processing_time}")
        if transient_detections:
            logger.info(f"🔭 Found {len(transient_detections)} transient candidates")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()