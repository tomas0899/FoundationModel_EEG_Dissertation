# Change - Tears for fears 
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import datetime, timedelta
from typing import Tuple, Optional

def process_eeg_mat_files(folder_path: str) -> Tuple[pd.DataFrame, list]:
    """
    Scans a folder for .mat EEG files, extracts temporal metadata (T0, TF), 
    calculates durations, and identifies recording gaps.

    Args:
        folder_path (str): Path to the directory containing .mat files.

    Returns:
        Tuple[pd.DataFrame, list]: A sorted DataFrame of results and a list of errors.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The path '{folder_path}' does not exist.")

    # 1) Filter and Sort Files
    all_files = os.listdir(folder_path)
    mat_files = sorted([
        f for f in all_files 
        if f.lower().endswith(".mat") and not f.startswith(".")
    ])

    results = []
    errors = []

    # 2) Extract Metadata
    for i, filename in enumerate(mat_files):
        file_path = os.path.join(folder_path, filename)

        try:
            data_mat = loadmat(file_path)
            hdr = data_mat["hdr"]

            # Extract T0 (Start Time)
            t0_raw = hdr["orig"][0, 0]["T0"][0, 0][0]
            t0_dt = datetime(
                int(t0_raw[0]), int(t0_raw[1]), int(t0_raw[2]),
                int(t0_raw[3]), int(t0_raw[4]), int(t0_raw[5])
            )

            # Extract Sampling Frequency and Data
            fs = float(hdr["Fs"][0, 0].item())
            signal = np.asarray(data_mat["data"])
            
            # Channel validation
            channels_raw = hdr["label"][0, 0]
            n_channels = channels_raw.shape[0]

            if signal.ndim == 2:
                # Transpose if shape is (n_channels, n_samples)
                if signal.shape[1] != n_channels and signal.shape[0] == n_channels:
                    signal = signal.T
            else:
                raise ValueError(f"Unexpected signal dimensions: {signal.shape}")

            n_samples = signal.shape[0]
            duration_seconds = n_samples / fs
            tf_dt = t0_dt + timedelta(seconds=duration_seconds)

            results.append({
                "list_idx": i,
                "file": filename,
                "T0": t0_dt,
                "TF": tf_dt,
                "duration_s": duration_seconds
            })

        except Exception as e:
            errors.append((filename, str(e)))

    # 3) Data Organization & Gap Calculation
    if not results:
        print("No valid data processed.")
        return pd.DataFrame(), errors

    df = pd.DataFrame(results)
    
    # Sort by actual Start Time (T0)
    df = df.sort_values("T0").reset_index(drop=True)
    
    # Calculate Gaps between files: T0 of current - TF of previous
    df["gap_s"] = (df["T0"] - df["TF"].shift(1)).dt.total_seconds().fillna(0)

    # Logging summary
    print(f"--- Processing Summary ---")
    print(f"Successfully processed: {len(df)}")
    print(f"Errors encountered: {len(errors)}")
    print(f"Total significant gaps (>1s): {(df['gap_s'] > 1).sum()}")
    
    return df, errors

# --- Example Usage ---
# path = "/your/folder/path/here/"
# df_results, error_list = process_eeg_mat_files(path)
# print(df_results.head())

import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional

def plot_eeg_availability_with_onsetsV2(
    df_files: pd.DataFrame, 
    df_onsets: pd.DataFrame, 
    output_path: Optional[str] = None,
    show_plot: bool = True
) -> pd.DataFrame:
    """
    Plots daily EEG recording availability, overlays seizure onsets,
    shows total daily hours and lists matched onset times in the plot.
    """
    # 0) Preparación de datos
    df_files = df_files.copy()
    df_onsets = df_onsets.copy()
    
    df_files["T0"] = pd.to_datetime(df_files["T0"])
    df_files["TF"] = pd.to_datetime(df_files["TF"])
    df_onsets["onset"] = pd.to_datetime(df_onsets["onset"])

    # --- LÓGICA DE MATCHING ---
    df_files = df_files.sort_values("T0")
    df_onsets = df_onsets.sort_values("onset")

    # Unimos para saber qué onset cae en qué archivo
    matched_df = pd.merge_asof(
        df_onsets, 
        df_files, 
        left_on="onset", 
        right_on="T0", 
        direction="backward"
    )

    # El onset debe estar dentro del rango [T0, TF] del archivo
    matched_df["captured"] = (matched_df["onset"] >= matched_df["T0"]) & \
                             (matched_df["onset"] <= matched_df["TF"])
    
    df_captured_onsets = matched_df[matched_df["captured"] == True].copy()

    # 1) Calcular Estado Binario (para el escalón del gráfico)
    events = []
    for _, row in df_files.iterrows():
        events.append((row["T0"], +1))
        events.append((row["TF"], -1))

    events_df = pd.DataFrame(events, columns=["Time", "Delta"]).sort_values("Time")
    events_df = events_df.groupby("Time", as_index=False)["Delta"].sum().sort_values("Time")
    events_df["State"] = events_df["Delta"].cumsum()
    events_df["Presence"] = (events_df["State"] > 0).astype(int)
    events_df["DayStart"] = events_df["Time"].dt.floor("D")
    
    unique_days = sorted(events_df["DayStart"].unique())

    # 2) Configuración del Plot
    fig, axes = plt.subplots(
        len(unique_days), 1, 
        figsize=(14, 3 * len(unique_days)), 
        sharey=True, 
        constrained_layout=True
    )
    if len(unique_days) == 1: axes = [axes]

    # 3) Plot por cada día
    for ax, start_day in zip(axes, unique_days):
        start_day = pd.Timestamp(start_day)
        end_day = start_day + pd.Timedelta(days=1)

        # Filtrar datos del día para la línea de presencia
        day_data = events_df[(events_df["Time"] >= start_day) & (events_df["Time"] < end_day)].copy()
        
        # Lógica de bordes para que no haya huecos al inicio/fin del día
        prev_state = events_df.loc[events_df["Time"] < start_day, "State"]
        presence_at_start = int(prev_state.iloc[-1] > 0) if not prev_state.empty else 0
        boundary_points = pd.DataFrame({"Time": [start_day, end_day], "Presence": [presence_at_start, None]})
        day_data = pd.concat([day_data[["Time", "Presence"]], boundary_points], ignore_index=True).sort_values("Time")
        day_data["Presence"] = day_data["Presence"].ffill().astype(int)

        # --- CÁLCULO DE DURACIÓN TOTAL ---
        # Calculamos la diferencia entre puntos de cambio de estado
        day_data["Duration"] = day_data["Time"].diff().shift(-1)
        total_duration_td = day_data.loc[day_data["Presence"] == 1, "Duration"].sum()
        total_hours = total_duration_td.total_seconds() / 3600

        # --- IDENTIFICAR ONSETS DEL DÍA ---
        day_onsets = matched_df[(matched_df["onset"] >= start_day) & (matched_df["onset"] < end_day)]
        captured_list = []

        for _, s_row in day_onsets.iterrows():
            color = "red" if s_row["captured"] else "gray"
            ax.axvline(s_row["onset"], color=color, linestyle="--", linewidth=1.5, alpha=0.8)
            
            if s_row["captured"]:
                # Guardamos la hora formateada para la leyenda interna
                captured_list.append(s_row["onset"].strftime("%H:%M:%S"))

        # --- VISUALS ---
        ax.step(day_data["Time"], day_data["Presence"], where="post", color="steelblue", linewidth=2)
        ax.fill_between(day_data["Time"], day_data["Presence"], step="post", alpha=0.2, color="steelblue")
        
        # Título con horas acumuladas
        ax.set_title(f"Date: {start_day.date()} | Total Recording: {total_hours:.2f} hrs", 
                     loc='left', fontweight='bold', fontsize=12)
        
        # Cuadro de texto con los onsets detectados
        if captured_list:
            onset_text = "Captured Onsets:\n" + "\n".join(captured_list)
            ax.text(1.01, 0.5, onset_text, transform=ax.transAxes, fontsize=9, 
                    verticalalignment='center', color="red",
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='red'))

        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(start_day, end_day)
        ax.set_ylabel("Presence")

    axes[-1].set_xlabel("Time (HH:MM)")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show_plot: 
        plt.show()
    else: 
        plt.close()

    return df_captured_onsets

def plot_eeg_availability_with_onsets(
    df_files: pd.DataFrame, 
    df_onsets: pd.DataFrame, 
    output_path: Optional[str] = None,
    show_plot: bool = True
) -> pd.DataFrame:
    """
    Plots daily EEG recording availability and overlays seizure onsets.
    Returns a DataFrame of onsets matched with their corresponding recording files.
    """
    # 0) Preparación de datos
    df_files = df_files.copy()
    df_onsets = df_onsets.copy()
    
    df_files["T0"] = pd.to_datetime(df_files["T0"])
    df_files["TF"] = pd.to_datetime(df_files["TF"])
    df_onsets["onset"] = pd.to_datetime(df_onsets["onset"])

    # --- NUEVA LÓGICA DE MATCHING ---
    # Ordenamos para usar merge_asof
    df_files = df_files.sort_values("T0")
    df_onsets = df_onsets.sort_values("onset")

    # Buscamos el archivo cuyo T0 sea el más cercano anterior al onset
    matched_df = pd.merge_asof(
        df_onsets, 
        df_files, 
        left_on="onset", 
        right_on="T0", 
        direction="backward"
    )

    # Filtramos: el onset debe ser menor o igual al TF del archivo encontrado
    # Si no cumple, significa que el onset cayó en un hueco sin grabación
    matched_df["captured"] = (matched_df["onset"] >= matched_df["T0"]) & \
                             (matched_df["onset"] <= matched_df["TF"])
    
    # Este es el DF que te servirá para graficar después
    df_captured_onsets = matched_df[matched_df["captured"] == True].copy()
    # --------------------------------

    # 1) Calcular Estado Binario (Lógica original para el plot)
    events = []
    for _, row in df_files.iterrows():
        events.append((row["T0"], +1))
        events.append((row["TF"], -1))

    events_df = pd.DataFrame(events, columns=["Time", "Delta"]).sort_values("Time")
    events_df = events_df.groupby("Time", as_index=False)["Delta"].sum().sort_values("Time")
    events_df["State"] = events_df["Delta"].cumsum()
    events_df["Presence"] = (events_df["State"] > 0).astype(int)
    events_df["DayStart"] = events_df["Time"].dt.floor("D")
    
    unique_days = sorted(events_df["DayStart"].unique())

    fig, axes = plt.subplots(
        len(unique_days), 1, 
        figsize=(14, 3 * len(unique_days)), 
        sharey=True, 
        constrained_layout=True
    )
    if len(unique_days) == 1: axes = [axes]

    # 3) Plot Each Day
    for ax, start_day in zip(axes, unique_days):
        start_day = pd.Timestamp(start_day)
        end_day = start_day + pd.Timedelta(days=1)

        day_data = events_df[(events_df["Time"] >= start_day) & (events_df["Time"] < end_day)].copy()
        
        # Boundary conditions
        prev_state = events_df.loc[events_df["Time"] < start_day, "State"]
        presence_at_start = int(prev_state.iloc[-1] > 0) if not prev_state.empty else 0
        boundary_points = pd.DataFrame({"Time": [start_day, end_day], "Presence": [presence_at_start, None]})
        day_data = pd.concat([day_data[["Time", "Presence"]], boundary_points], ignore_index=True).sort_values("Time")
        day_data["Presence"] = day_data["Presence"].ffill().astype(int)

        # Onsets del día (usando el DF ya procesado)
        day_onsets = matched_df[(matched_df["onset"] >= start_day) & (matched_df["onset"] < end_day)]
        
        for _, s_row in day_onsets.iterrows():
            color = "red" if s_row["captured"] else "gray"
            ax.axvline(s_row["onset"], color=color, linestyle="--", linewidth=1.2, alpha=0.8)

        # Visuals
        ax.step(day_data["Time"], day_data["Presence"], where="post", color="steelblue", linewidth=2)
        ax.fill_between(day_data["Time"], day_data["Presence"], step="post", alpha=0.2, color="steelblue")
        ax.set_title(f"Date: {start_day.date()}", loc='left', fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(start_day, end_day)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show_plot: plt.show()
    else: plt.close()

    return df_captured_onsets
# Uso del código
#df_matches = plot_eeg_availability_with_onsets(df, df_sqEEG)

# Ahora puedes ver qué onsets están en qué archivos
#print(df_matches[['onset', 'T0', 'TF', 'file_name']]) 

# Ejemplo para graficar el primer match
# primera_convulsion = df_matches.iloc[0]
# archivo_a_cargar = primera_convulsion['file_name']
# momento_exacto = primera_convulsion['onset']
##
##---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##

import pandas as pd
from pathlib import Path
from typing import Tuple

def preprocess_seizure_data(seizure_xlsx_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts seizure sheets from Excel, saves them as CSVs in a patient-specific 
    folder, and normalizes datetime columns.

    Args:
        seizure_xlsx_path (str): Full path to the .xlsx file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleaned (df_sqEEG, df_diary)
    """
    # 1) Setup Paths and Folder Names
    xlsx_path = Path(seizure_xlsx_path)
    base_dir = xlsx_path.parent
    patient_id = base_dir.name.upper()
    
    output_folder = base_dir / f"preprocessCSV_{patient_id}"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing Patient: {patient_id}")
    print(f"Output directory: {output_folder}")

    # 2) Load Sheets
    # Using a dictionary to handle potential missing sheets gracefully
    try:
        df_sqEEG = pd.read_excel(xlsx_path, sheet_name="sqEEG")
        df_diary = pd.read_excel(xlsx_path, sheet_name="diary")
    except Exception as e:
        print(f"Error reading sheets: {e}")
        raise

    # 3) Normalize Datetime Format
    # sqEEG usually uses 'onset', diary uses 'Timestamp'
    if "onset" in df_sqEEG.columns:
        df_sqEEG["onset"] = pd.to_datetime(df_sqEEG["onset"])
        
    if "Timestamp" in df_diary.columns:
        df_diary["Timestamp"] = pd.to_datetime(df_diary["Timestamp"])

    # 4) Save to CSV
    sqeeg_csv_path = output_folder / "sqEEG.csv"
    diary_csv_path = output_folder / "diary.csv"
    
    df_sqEEG.to_csv(sqeeg_csv_path, index=False)
    df_diary.to_csv(diary_csv_path, index=False)
    
    print(f"Successfully saved: \n - {sqeeg_csv_path.name} \n - {diary_csv_path.name}")

    return df_sqEEG, df_diary

# --- Example Usage ---
# file_path = "/home/tperezsanchez/FoundationModel_EEG_Dissertation/EEG_data_vis/data/Working/XB47Y/XB47Y_seizures.xlsx"
# df_sq, df_di = preprocess_seizure_data(file_path)
def plot_recording_availability_per_day(
    df_intervals: pd.DataFrame,
    df_seizures: pd.DataFrame = None,
    save_path: str = None,
    seizure_column: str = "onset"
):
    """
    Plot EEG recording availability per day as a binary step signal (1 = recording,
    0 = no recording). Optionally overlays seizure onset times as vertical lines.

    Parameters
    ----------
    df_intervals : pandas.DataFrame
        Must contain columns:
            - "T0" (recording start datetime)
            - "TF" (recording end datetime)

    df_seizures : pandas.DataFrame, optional
        DataFrame containing seizure timestamps.
        Must contain column specified by seizure_column.

    save_path : str, optional
        If provided, figure will be saved to this path.

    seizure_column : str, optional
        Name of the column in df_seizures containing seizure timestamps.
        Default = "onset".

    Returns
    -------
    None
    """

    # Ensure datetime format
    df = df_intervals.copy()
    df["T0"] = pd.to_datetime(df["T0"])
    df["TF"] = pd.to_datetime(df["TF"])

    # --- Create start/end events ---
    events = []

    for _, row in df.iterrows():
        events.append((row["T0"], 1))  # recording starts
        events.append((row["TF"], 0))  # recording ends

    events_df = pd.DataFrame(events, columns=["Time", "Presence"])
    events_df = events_df.sort_values("Time")

    # Extract unique days
    events_df["Date"] = events_df["Time"].dt.date
    unique_days = events_df["Date"].unique()

    # Create one subplot per day
    fig, axes = plt.subplots(
        len(unique_days),
        1,
        figsize=(14, 3 * len(unique_days)),
        sharey=True
    )

    # If only one day, make axes iterable
    if len(unique_days) == 1:
        axes = [axes]

    for ax, day in zip(axes, unique_days):

        day_data = events_df[events_df["Date"] == day]

        ax.step(day_data["Time"], day_data["Presence"], where="post")

        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel("Presence")
        ax.set_title(f"Date: {day}")

        start_day = pd.Timestamp(day)
        end_day = start_day + pd.Timedelta(days=1)
        ax.set_xlim(start_day, end_day)

        # --- Overlay seizure onset markers if provided ---
        if df_seizures is not None and seizure_column in df_seizures.columns:

            df_seiz = df_seizures.copy()
            df_seiz[seizure_column] = pd.to_datetime(df_seiz[seizure_column])

            day_seizures = df_seiz[
                (df_seiz[seizure_column] >= start_day) &
                (df_seiz[seizure_column] < end_day)
            ]

            for event_time in day_seizures[seizure_column]:
                ax.axvline(
                    event_time,
                    color="red",
                    linestyle="--",
                    linewidth=1.5
                )

    axes[-1].set_xlabel("Time")

    plt.suptitle("EEG Recording Availability Per Day")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
# EXAMPLE
#plot_recording_availability_per_day(
#    df_intervals=df_intervals,
#    df_seizures=df_sqEEG,
#    save_path="/home/tperezsanchez/FoundationModel_EEG_Dissertation/EEG_data_vis/results/XB47Y_EEG_recording_availability_per_dayONSET.png"
#)

def build_eeg_array_from_mat(
    hdr,
    mat_data,
    output_dir=".",
    file_prefix="EEG_data",
    save_format="npz",   # "npy" or "npz"
    return_dataframe=True
):
    """
    Build EEG array from .mat structure and save as .npy or .npz.

    Parameters
    ----------
    hdr : dict
        Header structure from .mat file
    mat_data : dict
        Full .mat dictionary
    output_dir : str
        Directory to save output
    file_prefix : str
        Prefix for output filename
    save_format : str
        "npy" (signal only) or "npz" (signal + metadata)
    return_dataframe : bool
        If True, also returns a DataFrame

    Returns
    -------
    signal : np.ndarray
    file_path : str
    (optional) EEG_Table : pandas.DataFrame
    """

    # Sampling frequency
    fs = float(hdr['Fs'][0,0])

    # Channel labels
    channels_raw = hdr['label'][0,0]
    channels = [str(row[0][0]) for row in channels_raw]

    # Extract signal
    signal = np.asarray(mat_data['data'], dtype=np.float32)

    # Fix orientation if needed
    if signal.shape[1] != len(channels) and signal.shape[0] == len(channels):
        signal = signal.T

    n_samples = signal.shape[0]
    time = np.arange(n_samples, dtype=np.float32) / fs

    os.makedirs(output_dir, exist_ok=True)

    # -------- SAVE --------
    if save_format == "npy":
        file_path = os.path.join(output_dir, f"{file_prefix}.npy")
        np.save(file_path, signal)

    elif save_format == "npz":
        file_path = os.path.join(output_dir, f"{file_prefix}.npz")
        np.savez(
            file_path,
            signal=signal,
            fs=fs,
            channels=channels,
            time=time
        )

    else:
        raise ValueError("save_format must be 'npy' or 'npz'")

    print(f"Saved EEG data to: {file_path}")
    print(f"Shape: {signal.shape}")
    print(f"Sampling frequency: {fs} Hz")

    if return_dataframe:
        EEG_Table = pd.DataFrame(signal, columns=channels)
        EEG_Table.insert(0, "Time", time)
        return signal, file_path, EEG_Table

    return signal, file_path

def apply_amplitude_cutoff(
    EEG_Table: pd.DataFrame,
    threshold: float = 200,
    start_sec: float = None,
    end_sec: float = None
):
    """
    Clip EEG amplitudes at ±threshold (µV),
    optionally selecting a time window in seconds.

    Parameters
    ----------
    EEG_Table : pandas.DataFrame
        DataFrame containing 'Time' column OR time as index (in seconds)
    threshold : float
        Amplitude threshold in µV (default 200)
    start_sec : float, optional
        Start time of window (in seconds)
    end_sec : float, optional
        End time of window (in seconds)

    Returns
    -------
    EEG_clipped : pandas.DataFrame
        Windowed and clipped DataFrame
    """

    # Copiar para no modificar original
    EEG_clipped = EEG_Table.copy()

    # ---------------------------------------------------
    # 1) Selección de ventana temporal (si se especifica)
    # ---------------------------------------------------
    if start_sec is not None and end_sec is not None:

        if "Time" in EEG_clipped.columns:
            EEG_clipped = EEG_clipped[
                (EEG_clipped["Time"] >= start_sec) &
                (EEG_clipped["Time"] <= end_sec)
            ]
        else:
            EEG_clipped = EEG_clipped.loc[
                (EEG_clipped.index >= start_sec) &
                (EEG_clipped.index <= end_sec)
            ]

    # ---------------------------------------------------
    # 2) Aplicar clipping
    # ---------------------------------------------------
    if "Time" in EEG_clipped.columns:
        signal_cols = EEG_clipped.columns.drop("Time")
        EEG_clipped[signal_cols] = EEG_clipped[signal_cols].clip(
            lower=-threshold,
            upper=threshold
        )
    else:
        EEG_clipped = EEG_clipped.clip(
            lower=-threshold,
            upper=threshold
        )

    return EEG_clipped
import matplotlib.pyplot as plt

def plot_eeg_with_shaded_threshold(
    EEG_Table,
    threshold=200,          # µV
    time_window=None,
    figsize=(12,6)
):
    """
    Plot EEG signals with grey shaded region between ±threshold (µV).

    Parameters
    ----------
    EEG_Table : pandas.DataFrame
        Must contain 'Time' column
    threshold : float
        Amplitude threshold in µV (default 200)
    time_window : tuple or None
        (start_time, end_time) in seconds
    figsize : tuple
        Figure size
    """

    # Apply time window if provided
    if time_window is not None:
        start, end = time_window
        EEG_Table = EEG_Table[
            (EEG_Table["Time"] >= start) &
            (EEG_Table["Time"] <= end)
        ]

    EEG_TimeTable = EEG_Table.set_index("Time")

    fig, axes = plt.subplots(
        nrows=EEG_TimeTable.shape[1],
        ncols=1,
        sharex=True,
        figsize=figsize
    )

    # If only one channel
    if EEG_TimeTable.shape[1] == 1:
        axes = [axes]

    for ax, channel in zip(axes, EEG_TimeTable.columns):

        # Plot signal
        ax.plot(EEG_TimeTable.index, EEG_TimeTable[channel])

        # Grey shaded region between ±threshold
        ax.axhspan(-threshold, threshold, alpha=0.15)

        # Horizontal lines at ±threshold
        ax.axhline(threshold, linestyle="--")
        ax.axhline(-threshold, linestyle="--")

        ax.set_ylabel(channel)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt

def plot_eeg_signals(
    EEG_Table,
    time_window=None,      # tuple (start, end) in seconds
    y_limit=None,          # tuple (-200, 200)
    figsize=(12,6),
    color=None             # str or list of colors
):
    """
    Plot EEG signals from a DataFrame with Time column.

    Parameters
    ----------
    EEG_Table : pandas.DataFrame
        DataFrame containing 'Time' + EEG channels
    time_window : tuple or None
        (start_time, end_time) in seconds
    y_limit : tuple or None
        (ymin, ymax)
    figsize : tuple
        Figure size
    color : str or list
        Single color for all channels OR list of colors per channel
    """

    # Apply time window if provided
    if time_window is not None:
        start, end = time_window
        EEG_Table = EEG_Table[
            (EEG_Table["Time"] >= start) &
            (EEG_Table["Time"] <= end)
        ]

    EEG_TimeTable = EEG_Table.set_index("Time")

    fig, axes = plt.subplots(
        nrows=EEG_TimeTable.shape[1],
        ncols=1,
        sharex=True,
        figsize=figsize
    )

    if EEG_TimeTable.shape[1] == 1:
        axes = [axes]

    for i, (ax, channel) in enumerate(zip(axes, EEG_TimeTable.columns)):

        # Select color
        if isinstance(color, list):
            plot_color = color[i] if i < len(color) else None
        else:
            plot_color = color

        ax.plot(
            EEG_TimeTable.index,
            EEG_TimeTable[channel],
            color=plot_color
        )

        ax.set_ylabel(channel)

        if y_limit is not None:
            ax.set_ylim(y_limit)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

def bandpass_filter_eegwin(
    EEG_win: pd.DataFrame,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
    check_nans: bool = True
):
    """
    Band-pass robusto usando SOS + sosfiltfilt (fase cero).

    EEG_win:
      - index: tiempo en segundos (numérico, creciente)
      - columns: canales
      - values: amplitud
    """

    # --- 1) Inferir fs desde el índice ---
    t = EEG_win.index.to_numpy(dtype=float)
    if t.size < 3:
        raise ValueError("Muy pocas muestras para inferir fs y filtrar (necesitas >= 3).")

    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("El índice de tiempo debe ser numérico, finito y estrictamente creciente.")

    fs = 1.0 / dt
    nyq = fs / 2.0

    # --- 2) Validaciones de cortes ---
    if lowcut <= 0:
        raise ValueError("lowcut debe ser > 0 Hz.")
    if highcut >= nyq:
        raise ValueError(f"highcut ({highcut} Hz) debe ser < Nyquist ({nyq:.2f} Hz).")
    if lowcut >= highcut:
        raise ValueError("lowcut debe ser < highcut.")

    # --- 3) NaNs ---
    if check_nans and EEG_win.isna().any().any():
        raise ValueError("EEG_win contiene NaNs. Rellena/interpola antes de filtrar.")

    # --- 4) Diseñar filtro ---
    low  = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="band", output="sos")



    # --- 5) Filtrar ---
    X = EEG_win.to_numpy(dtype=float)  # (n_samples, n_channels)
    try:
        Xf = sosfiltfilt(sos, X, axis=0)
    except ValueError as e:
        raise ValueError(
            f"No se pudo filtrar (posible ventana corta/padding). "
            f"Prueba una ventana más larga o baja el orden. Error: {e}"
        )

    EEG_win_filt = pd.DataFrame(Xf, index=EEG_win.index, columns=EEG_win.columns)
    return EEG_win_filt, fs
def process_eeg_windows(
    EEG_Table,
    window_size=10,
    threshold=200,
    lowcut=0.5,
    highcut=40,
    order=4,
    base_name="",
    plots_dir=""
):
    start_time = EEG_Table["Time"].min()
    end_time = EEG_Table["Time"].max()

    current_start = start_time
    colors = ["steelblue", "darkorange"]
    color_idx = 0

    while current_start + window_size <= end_time + 0.1:
        current_end = current_start + window_size

        # 1. Extraer el segmento de tiempo actual
        df_segment = EEG_Table[(EEG_Table["Time"] >= current_start) & (EEG_Table["Time"] <= current_end)].copy()
        
        if df_segment.empty:
            current_start += window_size
            continue

        # 2. Filtro Bandpass (usando TEEG directamente)
        df_win_idx = df_segment.set_index("Time")
        df_win_filt, fs = TEEG.bandpass_filter_eegwin(
            df_win_idx, lowcut=lowcut, highcut=highcut, order=order
        )

        # 3. Aplicar Cutoff
        df_final = TEEG.apply_amplitude_cutoff(
            df_win_filt.reset_index(), 
            threshold=threshold, 
            start_sec=current_start, 
            end_sec=current_end
        )

        # 4. Graficar
        plt.figure(figsize=(15, 7))
        TEEG.plot_eeg_signals(
            df_final, 
            color=colors[color_idx]
        )
        
        plt.title(f"Archivo: {base_name} | Ventana: {current_start:.1f}-{current_end:.1f}s")
        plt.xlabel("Tiempo (s)")
        plt.tight_layout()
        
        # Guardar gráfico de la ventana
        plot_filename = f"{base_name}_win_{int(current_start)}.png"
        plt.savefig(os.path.join(plots_dir, plot_filename), dpi=150)
        plt.close() 

        # Alternar color y avanzar
        color_idx = 1 - color_idx
        current_start += window_size