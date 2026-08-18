CHRONOS

Configurable Handling of Radio Observations for traNsient Object Search (CHRONOS) is a modular pipeline for detecting radio transients in archival GMRT data.
CHRONOS is a Python-based pipeline developed to automate the end-to-end search for long-period radio transients (LPTs) in radio
interferometric datasets. The pipeline combines standard radio astronomy
software such as CASA, WSClean, and PyBDSF with custom-developed Python modules.

Scientific Objective

The primary objective of CHRONOS is to provide an automated framework
for searching for long-period radio transients in radio
observations. Long-period transient searches require the analysis of
source behaviour over time and therefore involve processing large numbers of time-resolved images and measuring the flux density of variable sources.

CHRONOS is designed to reduce repetitive manual processing while
maintaining control over the scientific parameters used at each stage.
The modular structure allows individual processing steps to be enabled,
disabled, or rerun according to the requirements of a particular
dataset.

Methodology

The input data used is the pre-calibrated measurement sets, typically produced using standard calibration pipelines such as CAPTURE or CASA-based routines. Calibration corrects for instrumental and atmospheric effects, producing visibility datasets.

The first processing stage is continuum subtraction. The CASA task uvsub
is used to remove modelled continuum sources from the visibilities. This removes contributions from static sources that have already been modelled during calibration and imaging, leaving residual visibility data that can be used to search for transient emission. The CASA task mstransform is used to copy the residual visibilities into a measurement set, preparing it for further imaging and analysis.

For efficient processing of large observations, CHRONOS can split the
Measurement Set into individual scans. The scan_splitter.py module
automates this operation using CASA mstransform and supports parallel
processing through Python's multiprocessing library. Processing scans
independently can reduce imaging time and allow different scans to be
handled separately.

Deep Imaging and Source Catalogue Creation

The continuum-subtracted data are first processed with WSClean to
produce a deep interferometric image. The deep image is generated using
configurable imaging parameters such as image size, pixel scale,
weighting, W-stacking parameters, number of CLEAN iterations, masking
thresholds, and deconvolution settings.

The resulting FITS image is processed using PyBDSF to identify radio
sources and construct a reference source catalogue. The catalogue contains information such as right ascension, declination, major and minor axes, position angle, and flux density.
This source catalogue provides the list of sources that are subsequently
examined for temporal variability.

The deep imaging stage is automated by the deep_wsclean.py module, while source extraction is handled by the pybdsf_runner.py module.

Time-Resolved Imaging

After the deep image and reference source catalogue have been produced, the residual visibility data are used to generate time-resolved images.
WSClean is configured using the intervals-out parameter to divide the
observation into a specified number of temporal intervals.

The required number of intervals is determined from the duration of the
observation or individual scan, and the desired temporal resolution. For example, if a particular scan is to be analysed at a specified time
interval, the duration of that scan is used to determine the
corresponding number of output intervals.

The timeseries_wsclean.py module automates this process and generates the sequence of time-resolved images required for light-curve
construction. The time resolution can be changed through the configuration file according to the scientific requirements of the analysis.

Light-Curve Generation

The time-resolved images are used to measure the flux density of the
sources identified in the deep-image catalogue. The lightcurve_generator.py module processes the time-sliced images and
extracts flux measurements for each catalogued source.

For each image, the module reads the FITS data and extracts the World
Coordinate System information required to convert the cataloged right
ascension and declination of each source into image pixel coordinates.
The source morphology information from the PyBDSF catalogue is used to
define elliptical apertures around the sources.

The flux within each source aperture is measured from the image pixels.
The measured values are converted to flux density using the relevant
beam and pixel-area information. The uncertainty associated with each
measurement is estimated using the local RMS noise and the aperture
area.

The measurements obtained from all time slices are stored in CSV format.
The resulting data contain flux density, uncertainty, and RMS
information for each source and each time interval.

Transient Detection

The primary transient-detection stage uses a running-median and Median Absolute Deviation (MAD) approach to identify significant deviations in
source light curves.

For each source, a sliding temporal window is used to calculate the
local median and MAD of the measured flux densities. Measurements that
deviate significantly from the local temporal behaviour are selected as
potential transient events. The detection threshold is controlled
through a configurable parameter expressed in MAD units.

This provides a robust method for identifying changes in the source brightness while reducing sensitivity to isolated noise fluctuations and gradual variations in the light curve. The minimum number of valid
measurements required for detection can also be specified through the
configuration.

Detected transient candidates are stored in a separate output catalogue
containing relevant source and temporal information. 

Light-Curve Visualisation

For each source, CHRONOS can generate a light-curve plot showing flux
density as a function of observation time, together with the associated
measurement uncertainties. These plots provide a visual representation
of source variability and allow candidate events identified by the
statistical detection procedure to be inspected.

Diagnostic plots can highlight measurements identified as transient
candidates, allowing the automated results to be compared with the
observed the temporal behaviour of individual sources.
After light curves from individual scans have been generated, the
concat_lc_lombscargle.py module can combine the measurements for
individual sources across scans.

The scan_splitter.py module separates Measurement Sets into individual
scans and supports parallel processing using Python's multiprocessing
functionality.

Configuration

CHRONOS is controlled through a user-editable config.ini configuration file. The configuration file allows the user to select processing stages and specify parameters for data preparation, imaging, source extraction, light-curve generation, transient detection, scan processing, and file management.
The [modules] section determines which major stages are executed. uvsub_mstransform controls CASA continuum subtraction and Measurement Set transformation, deep_wsclean enables deep imaging, pybdsf enables source detection, timeseries_wsclean enables time-resolved imaging, and lightcurve_generator enables light-curve generation and transient detection. These parameters accept true or false values.
The [general] section contains execution-environment parameters. casa_dir specifies the path to CASA, wsclean_path specifies the WSClean executable, split_scans determines whether the Measurement Set is divided into scans, max_parallel_scans controls the maximum number of simultaneously processed scans, and the optional preamble parameter can specify shell commands to run before CASA.
The [uvsub] section contains continuum-subtraction parameters, including the input_ms parameter specifying the input Measurement Set.
The [mstransform] section controls Measurement Set transformation. 'scan' can specify a particular scan; leaving it empty allows all scans to be considered. output_ms specifies the output Measurement Set name and can be left empty when the default naming convention is preferred.
The [wsclean_deep] section controls deep imaging. Parameters include the input Measurement Set, output prefix, image size, pixel scale, number of output channels, W-stacking kernel size, W-stacking oversampling, polarisation product, data column, CLEAN iterations, auto-mask threshold, auto-threshold, CLEAN gain, multi-scale gain, multi-scale scale bias, spectral polynomial order, padding, and parallel deconvolution.
The [wsclean_timeSeries] section controls time-resolved imaging. time_interval specifies the temporal interval between images in seconds, while scan can restrict processing to a particular scan. Leaving scan empty means all available scans will be processed.
The [lightcurve_generator] section controls light-curve extraction and transient detection. Parameters include the PyBDSF catalogue, FITS image template, RMS estimation coordinates and radius, output directories, running-median window size, MAD threshold, and minimum number of points required for detection.
The [concatenate_catalogs] section controls the combination of per-scan catalogues. concatenate determines whether catalogues are combined, output_dir specifies the output directory, and window_size, k_sigma, which is the threshold, and min_points control the variability and transient-detection analysis.
The [file_cleanup] section controls the removal of intermediate files. The delete_files parameter specifies file patterns that can be deleted after processing to reduce disk usage.

Configuration Example

A representative configuration can be adapted as follows:
[modules]
uvsub_mstransform = true
deep_wsclean = true
pybdsf = true
timeseries_wsclean = true
lightcurve_generator = true

[general]
casa_dir = /path/to/casa
split_scans = true
max_parallel_scans =
wsclean_path = wsclean
preamble =

[uvsub]
input_ms = input.ms

[mstransform]
scan =
output_ms =

[wsclean_deep]
ms =
output_prefix =
size = 8000,8000
scale = 1.5asec
channels-out = 2
wstack-kernel-size = 7
wstack-oversampling = 63
pol = I
data-column = DATA
niter = 10
auto-mask = 5
auto-threshold = 3
gain = 0.1
mgain = 0.7
multiscale-scale-bias = 0.6
fit-spectral-pol = 3
padding = 1.3
parallel-deconvolution = 8192

[wsclean_timeSeries]
time_interval = 10
scan =

[lightcurve_generator]
catalog_file =
image_template =
rms_ra_deg = 
rms_dec_deg = 
rms_radius_pix = 400
output_dir = lightcurve_plots
transient_plot_dir = transient_detection_plots
window_size = 31
k_sigma = 3
min_points = 3

[concatenate_catalogs]
concatenate = true
output_dir = concatenated_lightcurves
window_size = 31
k_sigma = 3.0
min_points = 3

[file_cleanup]
delete_files =
The example values are provided for demonstration and should be adjusted.

Outputs

Depending on the enabled processing modules, CHRONOS can produce
continuum-subtracted Measurement Sets, scan-level Measurement Sets, deep FITS images, PyBDSF source catalogues, time-resolved FITS images, source light curves, RMS and uncertainty measurements, CSV catalogues,
transient-candidate catalogues, diagnostic plots, combined light curves,
and Lomb–Scargle periodograms.

These outputs provide the intermediate and final data products required
to identify, inspect, and characterise candidate long-period radio transients.

Project Structure

The repository contains the main CHRONOS Python package together with configuration and project-management files. The core modules include uvsub_mstransform.py, deep_wsclean.py, timeseries_wsclean.py,
pybdsf_runner.py, lightcurve_generator.py, scan_splitter.py,
concat_lc_lombscargle.py, and file_cleanup.py. The repository also
contains config.ini, setup.py, pyproject.toml, and README.md.

The exact repository structure may change as the pipeline continues to
develop.

Installation

CHRONOS is distributed as a Python package and can be installed using pip. Before installing CHRONOS, the required external radio astronomy software are CASA, WSClean, and PyBDSF.
Python packages used by the pipeline include Astropy, NumPy, Pandas, and Matplotlib.
The repository can be downloaded from GitHub using:
git clone https://github.com/ansha23/chronos.git
cd chronos
CHRONOS can then be installed using:
pip install .

Running CHRONOS

After installation and configuration, CHRONOS is executed from the terminal by providing the configuration file as an argument:
chronos config.ini

The command reads the specified configuration file and executes the enabled processing modules according to the supplied parameters.
For a new dataset, the user should prepare a suitable config.ini, provide the required input Measurement Set and software paths, and select the desired processing modules.
