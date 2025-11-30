import numpy as np
import matplotlib.pyplot as plt
# fitting scipy
from scipy.optimize import curve_fit
import pandas as pd
import os
import sys
import glob
import csv
from io import StringIO
# parasite host plots
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib.ticker import FuncFormatter, MaxNLocator
import mpl_toolkits.axisartist as AA


def gaussian(x, a, x0, sigma):
    """Gaussian function for curve fitting."""
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def profile_plotter(profile_data, bin_size=1.0, unit='', xlabel='', ylabel='', ymax=None, title='', savepath=None, fit=True):
    """
    Create a line plot for profile data.

    Parameters:
    - profile_data: array-like, y-axis data (profile values)
    - xlabel: str, label for the x-axis
    - ylabel: str, label for the y-axis
    - title: str, title of the plot
    - savepath: str or None, if provided, the path to save the plot image

    Returns:
    - fig: matplotlib figure object
    - ax: matplotlib axes object
    """
    profile_data = np.array(profile_data)
    if ymax is not None:
        profile_data = np.clip(profile_data, None, ymax)

    fig, ax = plt.subplots()

    highest_value = np.max(profile_data)
    # compute mean around maximum only
    if fit:
        mask = profile_data >= (0.7 * highest_value)
        # gaussian fit only on significant values
        try:
            popt, _ = curve_fit(gaussian, np.arange(len(profile_data))[
                            mask], profile_data[mask], p0=[highest_value, np.argmax(profile_data), 1])
        except Exception as e:
            print(f"Gaussian fit failed: {e}")
            popt = None
        if popt is not None:
            mean_values = popt[1] * bin_size
            std = abs(popt[2]) * bin_size
        else:
            mean_values = np.mean(profile_data[mask]) * bin_size
            std = np.std(profile_data[mask]) * bin_size

    ax.plot(profile_data, marker='+', color='blue',
            linestyle='-', label='Profile Data')
    if fit and popt is not None:
        x_fit = np.linspace(0, len(profile_data)-1, 200)
        y_fit = gaussian(x_fit, *popt)
        ax.plot(x_fit, y_fit, color='b', linestyle='--',
                label='Gaussian Fit', alpha=0.5)
        text_for_box = '\n'.join([f"Mean: {mean_values:.2f} {unit}", f"Std Dev: {std:.2f} {unit}",
                              f"Bin size: {bin_size} {unit}"])
        ax.text(0.05, 0.95, text_for_box, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    ticks = np.arange(0, len(profile_data), max(1, len(profile_data)//10))
    tick_labels = [f"{i * bin_size:.2f}" for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.set_xlabel(xlabel+f" ({unit})" if unit else xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_title(title)

    ax.legend()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)
        print(f"Profile plot saved to {savepath}")
    plt.close()
    return fig, ax


def spectrums_loader(file_path):
    """
    Load spectrum data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    returns: pd.DataFrame: DataFrame containing the spectrum data.
    list containing counts per bin
    """
    # Parse the file manually to get the trailing block of integer counts
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]

    # Find the last contiguous block of lines that are integers
    counts = []
    # Traverse from bottom collecting integer lines until a non-integer line is found
    for ln in reversed(lines):
        if ln == "":
            # allow a trailing blank, but stop if we've already started collecting
            if counts:
                break
            continue
        try:
            val = int(ln)
            counts.append(val)
        except ValueError:
            # Stop if we have already collected some integers
            if counts:
                break

    counts = list(reversed(counts))

    # If instead the last line has a single row of comma/space-separated integers
    # handle that case as well
    if not counts:
        for ln in reversed(lines):
            # try splitting by comma or whitespace and parsing all as int
            parts = [p for p in ln.replace(",", " ").split() if p]
            if parts:
                try:
                    counts = [int(p) for p in parts]
                    break
                except ValueError:
                    continue

    if not counts:
        raise ValueError("No integer counts found at the end of the file.")

    # Build a simple DataFrame: channel index and counts
    data = pd.DataFrame({
        "Channel": np.arange(len(counts)),
        "Counts": counts
    })

    return data


def peak_finder(profile_data,
                x_data =None,
                x_label= " ",
                bin_size=1.0,
                bin_unit=' ',                
                maximum_peak = None,
                range=[0.000001, 1e+20], 
                num_peaks=10, 
                save=False, 
                sigma_filter=2, 
                threshold=1e-10, 
                prominence=1e-10, 
                filename=None):
    '''   
    Find and plot peaks in the provided data.
    Parameters: 
    - x_data: array-like, x-axis data
    - y_data: array-like, y-axis data
    - x_label: str, label for the x-axis
    - range: list, [min, max] range for x_data to consider
    - num_peaks: int, number of peaks to identify
    - save: bool, whether to save the plot
    - sigma_filter: float, standard deviation for Gaussian filter
    - threshold: float, threshold for peak detection
    - prominence: float, prominence for peak detection
    - filename: str or bool, filename to save the plot  
    Returns:
    - peaks: array-like, indices of the detected peaks
    '''

    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    from matplotlib.lines import Line2D

    
    if x_data is None:
        x_data = np.arange(len(profile_data))
    
    
    y_data = np.array(profile_data)    
    mask = np.where(y_data <= maximum_peak) if maximum_peak is not None else np.ones_like(y_data, dtype=bool)
    
    x_data = x_data[mask]
    y_data = y_data[mask]
    
    
    good_indices = np.where((x_data >= range[0]) & (x_data <= range[1]))

    x_data = x_data[good_indices]
    y_data = y_data[good_indices]

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    

    y_data_gaussian = gaussian_filter1d(y_data, sigma=sigma_filter)
    

    peaks, _ = find_peaks(y_data_gaussian, threshold=threshold, prominence=prominence)
    
    
    if len(peaks) < num_peaks:
        print(f"Found {len(peaks)} peaks.")

    proxy = Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='red', markersize=10, label='Detected Peaks')

    custom_line = Line2D([0], [0], marker='o', color='black', lw=2)

    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'b-', linewidth=1, label="Data")
    handles, labels2 = plt.gca().get_legend_handles_labels()

    n = 1
    for peak in peaks[:num_peaks]:
        plt.axvline(x=x_data[peak], color='r', linestyle='--')
        print(f"Peak found at: {x_data[peak]* (1.0*bin_size):1.6f} {bin_unit} with amplitude {y_data[peak]:1.6f}")
        labels2.append(f"E0[{n}] = {x_data[peak]* (1.0*bin_size):1.6f} {bin_unit}, A[{n}] = {y_data[peak]:1.6f}")
        handles.append(proxy)
        n = n+1

    plt.legend(handles=handles, loc="best", fontsize="large", labels=labels2)

    plt.xlabel(f"{x_label}", fontsize=14)
    plt.ylabel(f"Energy ({bin_unit})", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if save and filename is not None:
        plt.savefig(filename, format='png')
    plt.close()
    return peaks


if __name__ == "__main__": 
    # example of use
    file_path = "/Users/santiagomartinez/Documents/PARIS SACLAY/SEMESTER 3/TP_THOMX/TP-DATA/20251121/spectrum/"
    filename = "23B1535b_withSn_foil_beam_on_Sn_foil"
    format = ".mca"
    data = spectrums_loader(file_path + filename + format)
    profile_plotter(data['Counts'][1:],
                    bin_size=1.0,
                    unit=' ',
                    xlabel='Energy (bin)',
                    ylabel='Counts',
                    ymax=400,
                    title='X-ray Spectrum',
                    savepath=file_path + f"{filename}_spectrum_plot.png",
                    fit = False
                    )
    peaks = peak_finder(profile_data = data['Counts'][1:].values,
                x_data = None,
                x_label = 'Energy (bin)',
                bin_size = 1.0,
                bin_unit = ' ',
                maximum_peak = 400,
                range = [500, 2863],
                num_peaks = 10,
                save = True,
                sigma_filter = 2,
                threshold = 1e-1,
                prominence = 11,
                filename = file_path + f"{filename}_spectrum_peaks.png"
                )
    xdata = np.arange(len(data['Counts'][1:].values))
    ydata = data['Counts'][1:].values
    
    ydata_mask = ydata <= 400
    xdata_cut = xdata[ydata_mask]
    ydata_cut = ydata[ydata_mask]
    
    xdata_peaks = xdata[peaks+500]
    
    Sn_Peaks = [2477] #Kalpha1, Kbeta1
    Sn_energies = [25.271] #keV
    
    unit_conversion = Sn_energies[0]/ Sn_Peaks[0]  # keV per bin
    bin_size_keV = unit_conversion
    
    
    plt.plot(xdata_cut*bin_size_keV, ydata_cut, label='Spectrum Data', color='blue', alpha=0.5, linestyle='--')
    plt.plot(xdata_peaks*bin_size_keV, data['Counts'][1:].values[peaks+500], 'ro', label='Detected Peaks')
    plt.plot(np.array(Sn_Peaks)*bin_size_keV, [data['Counts'][1:].values[int(p)] for p in Sn_Peaks], 'bx', label='Detected Sn Peaks')
   
    for i, txt in enumerate([f"{e} keV" for e in Sn_energies]):
        plt.annotate(txt, (Sn_Peaks[i]*bin_size_keV, data['Counts'][1:].values[int(Sn_Peaks[i])]), textcoords="offset points", xytext=(-2,2), ha='right', color='green')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts')
    plt.title('Detected Peaks vs Expected Sn Peaks')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(file_path + f"{filename}_spectrum_peaks_comparison.png")

    plt.close()
   
    print(f"Bin size in keV: {bin_size_keV}")
    unit = 'keV'
    profile_plotter(data['Counts'][1:],
                    bin_size=bin_size_keV,
                    unit=unit,
                    xlabel='Energy',
                    ylabel='Counts',
                    ymax=400,
                    title='X-ray Spectrum in keV',
                    savepath=file_path + f"{filename}_spectrum_keV_plot.png",
                    fit = False
                    )
    
    filename_paraffin = "23B1535b_withSn_foil_beam_on_paraffin"
    format = ".mca"
    data_paraffin = spectrums_loader(file_path + filename_paraffin + format)
    profile_plotter(data_paraffin['Counts'][1:],
                    bin_size=bin_size_keV,
                    unit=unit,
                    xlabel='Energy',
                    ylabel='Counts',
                    ymax = 30000,
                    title='X-ray Spectrum in keV (Paraffin)',
                    savepath=file_path + f"{filename_paraffin}_spectrum_keV_plot.png",
                    fit = False
                    )
    
    
    
    
    
    
    
    
    

    