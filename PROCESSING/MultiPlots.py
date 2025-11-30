import numpy as np
import matplotlib.pyplot as plt
#fitting scipy 
from scipy.optimize import curve_fit
import pandas as pd
import os, sys, glob
import csv
from io import StringIO
#parasite host plots
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib.ticker import FuncFormatter, MaxNLocator
import mpl_toolkits.axisartist as AA
def gaussian(x, A, mu, sigma):
    """Gaussian function for fitting."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def sigma_quad_scan(k, a, b, c):
    """Quadratic function for quad scan fitting: sigma^2 = a*k^2 + b*k + c"""
    return a * k**2 + b * k + c
def scientific_notation(x):
    """Formatter function for scientific notation with 2 decimal places."""
    exponent = int(np.floor(np.log10(abs(x)))) if x != 0 else 0
    residue = x / (10**exponent)
    return f"{residue:.2e} x10^{exponent}"

def sigmas_from_quad_scan_params(a, b, c, d, l, energy):
    """Compute sigma values from quad scan parameters."""
    energy_MeV = energy #energy in MeV
    gamma_lorentz = energy_MeV / 0.511 
    beta = np.sqrt(1 - 1/(gamma_lorentz**2))
    sigma11 = a/(d**2*l**2)
    sigma12 = (b - 2*d*l*sigma11)/(2*d**2*l)
    sigma22 = ( c - sigma11 - 2*d*sigma12 )/(d**2)
    emmitance = np.sqrt(sigma11*sigma22 - sigma12**2) 
    normalized_emmitance = emmitance * gamma_lorentz * beta
    return sigma11, sigma12, sigma22, emmitance, normalized_emmitance


def Measurement_scatter(xdata, ydata, xerr=None, yerr=None, xlabel='', ylabel='', title='', savepath=None, fit=False , fit_func=None, p0=None):
    """
    Create a scatter plot for Measurement data with error bars.

    Parameters:
    - xdata: array-like, x-axis data
    - ydata: array-like, y-axis data
    - xerr: array-like or scalar, error in x-axis data
    - yerr: array-like or scalar, error in y-axis data
    - xlabel: str, label for the x-axis
    - ylabel: str, label for the y-axis
    - title: str, title of the plot
    - savepath: str or None, if provided, the path to save the plot image

    Returns:
    - fig: matplotlib figure object
    - ax: matplotlib axes object
    """
    fig, ax = plt.subplots()
    
    ax.scatter(xdata, ydata, marker='+', color='red', label='Data')
    if xerr is not None or yerr is not None:
        ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt='none', ecolor='gray', capsize=5)   
     

    if fit and fit_func is not None:
        try:
            popt, pcov = curve_fit(fit_func, xdata, ydata, p0=p0)
            x_fit = np.linspace(min(xdata), max(xdata), 100)
            y_fit = fit_func(x_fit, *popt)
            ax.plot(x_fit, y_fit, 'b--', label='Fit')
            
            fit_params_text = '\n'.join([f'p{i} = {scientific_notation(param)}' for i, param in enumerate(popt)])
            ax.text(0.05, 0.95, fit_params_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        except Exception as e:
            print(f"Fit failed: {e}")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_title(title)
    ax.legend()


    #ticks with at most 2 decimal places in scientific notation
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
        print(f"Plot saved to {savepath}")
    plt.close()
    return fig, ax

def profile_plotter(profile_data, bin_size = 1.0, unit = '' , xlabel='', ylabel='', title='', savepath=None):
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
    fig, ax = plt.subplots()

    highest_value = np.max(profile_data)
    #compute mean around maximum only
    mask = profile_data >= (0.7 * highest_value)
    #gaussian fit only on significant values
    try:
        popt, _ = curve_fit(gaussian, np.arange(len(profile_data))[mask], profile_data[mask], p0=[highest_value, np.argmax(profile_data), 1])
    except Exception as e:
        print(f"Gaussian fit failed: {e}")
        popt = None
    if popt is not None:
        mean_values = popt[1] * bin_size
        std = abs(popt[2]) * bin_size
    else:
        mean_values = np.mean(profile_data[mask]) * bin_size
        std = np.std(profile_data[mask]) * bin_size

    
    
    ax.plot(profile_data, marker='+', color='blue', linestyle='-', label='Profile Data')
    if popt is not None:
        x_fit = np.linspace(0, len(profile_data)-1, 200)
        y_fit = gaussian(x_fit, *popt)
        ax.plot(x_fit, y_fit, color='b', linestyle='--', label='Gaussian Fit', alpha=0.5)
    ticks = np.arange(0, len(profile_data), max(1, len(profile_data)//10))
    tick_labels = [f"{i * bin_size:.2f}" for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.set_xlabel(xlabel+f" ({unit})" if unit else xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_title(title)

    text_for_box = '\n'.join([f"Mean: {mean_values:.2f} {unit}", f"Std Dev: {std:.2f} {unit}",
                              f"Bin size: {bin_size} {unit}"])
    ax.text(0.05, 0.95, text_for_box, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    ax.legend()
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath)
        print(f"Profile plot saved to {savepath}")
    plt.close()
    return fig, ax

def phase_energy_charge_plotter(phase_data, energy_data, charge_data, 
                                energy_label='Energy (MeV)', 
                                charge_label='Charge (pC)',
                                phase_label='Phase (degrees)',
                                title='Phase, Energy, and Charge Plot',
                                savepath=None):
    host = host_subplot(111)
    par =  host.twinx()

    host.set_xlabel(phase_label)
    host.set_ylabel(charge_label)
    par.set_ylabel(energy_label)


    p1 = host.scatter(phase_data, charge_data, marker = "+",color='black', label=charge_label)
    host.plot(phase_data, charge_data, linestyle='--', color='gray', alpha=0.5)
    p2 = par.scatter(phase_data, energy_data, marker = "+", color='red', label=energy_label)
    par.plot(phase_data, energy_data, linestyle='--', color='pink', alpha=0.5)
    host.legend(handles=[p1, p2])

    host.set_title(title)
    host.grid(True)
    par.grid(False)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)
        print(f"Phase-Energy-Charge plot saved to {savepath}")
    plt.close()
    return host, par

def family_measurements_plot(kind_of_measurement, mother_path, plot=False, split_blocks=False, block_pattern=None, delimiter=','):
    """
    Read family measurement data from CSV files matching `kind_of_measurement`.

    Features:
    - By default tries to read each CSV with pandas.read_csv.
    - If parsing fails or `split_blocks=True`, the file is split into consecutive
      blocks where each block has the same number of fields per line. Each block
      is returned as a DataFrame.
    - If `block_pattern` is provided (e.g. [2,3,2]) the function will try to
      locate the first contiguous sequence of blocks matching that column-count
      pattern and return only those blocks for that file.

    Parameters:
    - kind_of_measurement: str, type of measurement (e.g., 'EMeasurement')
    - mother_path: str, path to the directory containing the CSV files
    - plot: bool, if True generate simple scatter plots for any readable parts
    - split_blocks: bool, if True always attempt block-splitting parsing
    - block_pattern: list of ints or None, column-count pattern to pick (e.g. [2,3,2])
    - delimiter: CSV delimiter (default ',')

    Returns:
    - data_frames: list where each element corresponds to one file and is itself
      a list of pandas.DataFrame objects (one per block/part). Files that parse
      as a single table will be returned as a single-element list.
    """
    csv_files = glob.glob(os.path.join(mother_path, f'*{kind_of_measurement}*.csv'))
    if not csv_files:
        print(f"No CSV files found for kind_of_measurement: {kind_of_measurement}")
        return []

    data_frames = []
    for file in csv_files:
        parts = None
        # First try a simple pandas read
        if not split_blocks:
            try:
                df = pd.read_csv(file)
                parts = [df]
            except Exception as e:
                # fall through to block-splitting
                print(f"pandas.read_csv failed for {os.path.basename(file)}: {e}; trying block-splitting")

        if parts is None:
            # Use block-splitting reader
            blocks = read_csv_into_blocks(file, delimiter=delimiter)
            dfs_blocks = blocks_to_dataframes(blocks, delimiter=delimiter)
            if block_pattern:
                matched = select_blocks_by_pattern(dfs_blocks, block_pattern)
                if matched:
                    parts = [m['df'] for m in matched]
                else:
                    print(f"Pattern {block_pattern} not found in {os.path.basename(file)}; returning all blocks")
                    parts = [d['df'] for d in dfs_blocks]
            else:
                parts = [d['df'] for d in dfs_blocks]

        data_frames.append(parts)

        if plot:
            print(f"Plotting data from file: {file}")
            for idx, part in enumerate(parts):
                if isinstance(part, pd.DataFrame) and part.shape[1] >= 2:
                    xdata = part.iloc[:, 0]
                    ydata = part.iloc[:, 1]
                    Measurement_scatter(xdata, ydata, xlabel=str(part.columns[0]), ylabel=str(part.columns[1]),
                                        title=f"Data from {os.path.basename(file)} (part {idx})",
                                        savepath = mother_path + f"{os.path.basename(file).replace('.csv', f'_part{idx}_plot.png')}"
                                        )
                else:
                    print(f"Part {idx} of {os.path.basename(file)} has less than 2 columns; skipping plot")

    print(f"Loaded {len(data_frames)} files for kind_of_measurement: {kind_of_measurement}")
    return data_frames


def read_csv_into_blocks(filepath, delimiter=','):
        """Return list of blocks where each block is a dict with keys:
        - ncols: number of columns in that block
        - lines: list of raw text lines belonging to the block
        - start_line, end_line: 1-based indices in the original file
        """
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            raw_lines = f.readlines()

        blocks = []
        current_n = None
        current_lines = []
        start = None

        for i, raw in enumerate(raw_lines):
            # keep the original raw line for later re-parsing
            # determine number of fields using csv.reader on the single line
            try:
                row = next(csv.reader([raw], delimiter=delimiter))
            except Exception:
                # Fallback: naive split
                row = raw.strip().split(delimiter)

            # treat blank/empty-only lines as separators
            if len(row) == 0 or all((field.strip() == '') for field in row):
                if current_lines:
                    blocks.append({'ncols': current_n, 'lines': current_lines, 'start_line': start, 'end_line': i})
                    current_lines = []
                    current_n = None
                    start = None
                continue

            n = len(row)
            if current_n is None:
                current_n = n
                start = i + 1
                current_lines = [raw]
            elif n == current_n:
                current_lines.append(raw)
            else:
                # close previous block and start new
                blocks.append({'ncols': current_n, 'lines': current_lines, 'start_line': start, 'end_line': i})
                current_n = n
                start = i + 1
                current_lines = [raw]

        if current_lines:
            blocks.append({'ncols': current_n, 'lines': current_lines, 'start_line': start, 'end_line': len(raw_lines)})

        return blocks

def blocks_to_dataframes(blocks, delimiter=','):
    dfs = []
    for b in blocks:
        text = ''.join(b['lines'])
        # parse block text with pandas; no header assumed
        try:
            df_block = pd.read_csv(StringIO(text), header=None, sep=delimiter)
        except Exception as e:
            print(f"Warning: pandas failed to parse a block starting at line {b['start_line']}: {e}")
            # fallback: parse manually
            rows = []
            for raw in b['lines']:
                try:
                    row = next(csv.reader([raw], delimiter=delimiter))
                except Exception:
                    row = raw.strip().split(delimiter)
                rows.append(row)
            df_block = pd.DataFrame(rows)
        dfs.append({'ncols': b['ncols'], 'df': df_block, 'start_line': b['start_line'], 'end_line': b['end_line']})
    return dfs

def select_blocks_by_pattern(dfs, pattern):
        """Search dfs (list of dicts with 'ncols') for first contiguous subsequence matching pattern.
        Returns list of df dicts if found, else empty list.
        """
        nlist = [d['ncols'] for d in dfs]
        plen = len(pattern)
        for i in range(len(nlist) - plen + 1):
            if nlist[i:i+plen] == pattern:
                return dfs[i:i+plen]
        return []









if False:
    #csv retrieval
    
    mother_path = './TP-DATA/TP_GI/'  # Update this path as needed
    #EMeasurement = 'EMeasurement'
    #csv_files = family_measurements_plot(EMeasurement, mother_path, plot=True, split_blocks=False, block_pattern=None)
    Quad_Scan = 'Quad_Scan'
    csv_files_Quad = family_measurements_plot(Quad_Scan, mother_path, plot=False, split_blocks=True, block_pattern=[2,3,2])
    #print only part 2 of quad scan files
    print("Quad Scan files, part 2:")
    for file_idx, parts in enumerate(csv_files_Quad):
        axis_names = parts[1].iloc[0].tolist()
        k_data = parts[1].iloc[1:,0]; k_data = np.array(k_data, dtype=float)
        X_sigma2 = parts[1].iloc[1:,1]; X_sigma2 = np.array(X_sigma2, dtype=float)
        Y_sigma2 = parts[1].iloc[1:,2]; Y_sigma2 = np.array(Y_sigma2, dtype=float)

        Measurement_scatter(xdata= k_data,
                            ydata= X_sigma2,
                            xlabel=axis_names[0], 
                            ylabel=axis_names[1],
                            title=f"Quad Scan File {file_idx} {axis_names[0]} vs {axis_names[1]}",
                            savepath = f"./TP-DATA/TP_GI/QuadScan_File{file_idx}_{axis_names[1]}_plot.png",
                            fit = True,
                            fit_func = sigma_quad_scan,
                            p0 = [1e-7, 1e-7, 1e-7]
        )
        Measurement_scatter(xdata= k_data, 
                            ydata= Y_sigma2, 
                            xlabel=axis_names[0], 
                            ylabel=axis_names[2],
                            title=f"Quad Scan File {file_idx} {axis_names[0]} vs {axis_names[2]}",
                            savepath = f"./TP-DATA/TP_GI/QuadScan_File{file_idx}_{axis_names[2]}_plot.png",
                            fit = True,
                            fit_func = sigma_quad_scan,
                            p0 = [1e-7, 1e-7, 1e-7]
        )   

if __name__ == "__main__":
    mother_path = './TP-DATA/TP_GI/'  # Update this path as needed
    Scan_GUN = "Scan_GUN"
    csv_files_GUN = family_measurements_plot(Scan_GUN, mother_path, plot=False, split_blocks=False, block_pattern=None)
    for idx, df in enumerate(csv_files_GUN):
        phase_energy_charge_plotter(phase_data=df[0].iloc[:,0], 
                                   energy_data=df[0].iloc[:,1], 
                                   charge_data=df[0].iloc[:,2],
                                   energy_label=str(df[0].columns[1]),
                                   charge_label=str(df[0].columns[2]),
                                   phase_label=str(df[0].columns[0]),
                                   title=f"Phase, Energy, and Charge from Scan_GUN File {idx}",
                                   savepath = mother_path + f"Scan_GUN_File{idx}_Phase_Energy_Charge_plot.png"
                                   )

    mother_TP_GI3 = './TP-DATA/TP_GI3/'  # Update this path as needed
    csv_files_GI3 = family_measurements_plot(Scan_GUN, mother_TP_GI3, plot=False, split_blocks=False, block_pattern=None)
    for idx, df in enumerate(csv_files_GI3):
        profile_plotter(df[0].iloc[:,0], 
                             bin_size=28.0, 
                             unit=r'$\mu m$', 
                             xlabel="x position", 
                             ylabel="Intensity [a.u.]",
                             title=f"Profile Plot from Scan_GUN GI3",
                             savepath = mother_TP_GI3 + f"Scan_GUN_GI3_File{idx}_Profile_X_plot.png"
                             )
        profile_plotter(df[0].iloc[:,1], 
                             bin_size=28.0, 
                             unit=r'$\mu m$', 
                             xlabel="y position", 
                             ylabel="Intensity [a.u.]",
                             title=f"Profile Plot from Scan_GUN GI3",
                             savepath = mother_TP_GI3 + f"Scan_GUN_GI3_File{idx}_Profile_Y_plot.png"
                             )
        
    mother_TP_GI4 = './TP-DATA/TP_GI4/'  # Update this path as needed
    csv_files_GI4 = family_measurements_plot(Scan_GUN, mother_TP_GI4, plot=False, split_blocks=False, block_pattern=None)
    for idx, df in enumerate(csv_files_GI4):
        profile_plotter(df[0].iloc[:,0], 
                             bin_size=1.0, 
                             unit=r'$\mu m$', 
                             xlabel="x position", 
                             ylabel="Intensity [a.u.]",
                             title=f"Profile Plot from Scan_GUN GI4",
                             savepath = mother_TP_GI4 + f"Scan_GUN_GI4_File{idx}_Profile_X_plot.png"
                             )
        profile_plotter(df[0].iloc[:,1], 
                             bin_size=1.0, 
                             unit=r'$\mu m$', 
                             xlabel="y position", 
                             ylabel="Intensity [a.u.]",
                             title=f"Profile Plot from Scan_GUN GI4",
                             savepath = mother_TP_GI4 + f"Scan_GUN_GI4_File{idx}_Profile_Y_plot.png"
                             )

























    