"""
Clinical Feature Extraction for Multi-Heartbreaker LightGBM Pipeline.

Extracts cardiology-standard features from each 12-lead ECG signal:
- HYP: Sokolow-Lyon index, Cornell voltage criteria
- MI: Pathological Q-wave metrics, ST elevation per lead group
- CD: QRS duration, PR interval estimation
- STTC: ST deviation, T-wave inversion metrics
- General: Heart rate, QTc, R-wave amplitudes, axis estimation
"""

import os
import numpy as np
import pandas as pd
import wfdb
import scipy.signal as sig_proc
from scipy.signal import find_peaks

# PTB-XL 100 Hz lead order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
FS = 100  # sampling frequency


def bandpass_filter(signal_data, lowcut=0.5, highcut=40.0, fs=FS, order=4):
    """Bandpass filter for ECG signal."""
    nyq = 0.5 * fs
    b, a = sig_proc.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    filtered = np.zeros_like(signal_data)
    for i in range(signal_data.shape[1]):
        filtered[:, i] = sig_proc.filtfilt(b, a, signal_data[:, i])
    return filtered


def detect_r_peaks(lead_signal, fs=FS):
    """Detect R-peaks using scipy peak detection on lead II."""
    # Minimum distance between peaks: ~300ms (200 bpm max)
    min_dist = int(0.3 * fs)
    # Adaptive threshold: 40% of max amplitude
    threshold = 0.4 * np.max(np.abs(lead_signal))
    peaks, properties = find_peaks(lead_signal, distance=min_dist, height=threshold)
    return peaks


def compute_heart_rate(r_peaks, fs=FS):
    """Compute heart rate from R-R intervals."""
    if len(r_peaks) < 2:
        return np.nan
    rr_intervals = np.diff(r_peaks) / fs  # in seconds
    rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]  # physiological range
    if len(rr_intervals) == 0:
        return np.nan
    return 60.0 / np.mean(rr_intervals)


def compute_qrs_duration(lead_signal, r_peaks, fs=FS):
    """Estimate QRS duration from the width of the R-peak complex."""
    if len(r_peaks) == 0:
        return np.nan
    
    durations = []
    for r in r_peaks:
        # Search window: 150ms around R-peak
        window = int(0.075 * fs)
        start = max(0, r - window)
        end = min(len(lead_signal), r + window)
        segment = lead_signal[start:end]
        
        # QRS boundaries: where amplitude drops below 20% of R-peak
        r_amp = np.abs(lead_signal[r])
        threshold = 0.2 * r_amp
        
        above = np.abs(segment) > threshold
        if np.any(above):
            first = np.argmax(above)
            last = len(above) - 1 - np.argmax(above[::-1])
            duration_ms = (last - first) / fs * 1000
            if 40 < duration_ms < 200:  # physiological QRS range
                durations.append(duration_ms)
    
    return np.median(durations) if durations else np.nan


def compute_st_deviation(lead_signal, r_peaks, fs=FS):
    """Compute ST segment deviation (measured 80ms after J-point)."""
    if len(r_peaks) == 0:
        return np.nan, np.nan
    
    st_values = []
    for r in r_peaks:
        # J-point approx 40ms after R-peak, ST measurement 80ms after J
        j_point = r + int(0.04 * fs)
        st_point = j_point + int(0.08 * fs)
        
        if st_point < len(lead_signal):
            # Baseline: average of segment before QRS
            baseline_start = max(0, r - int(0.15 * fs))
            baseline_end = max(0, r - int(0.05 * fs))
            if baseline_end > baseline_start:
                baseline = np.mean(lead_signal[baseline_start:baseline_end])
                st_dev = lead_signal[st_point] - baseline
                st_values.append(st_dev)
    
    if not st_values:
        return np.nan, np.nan
    
    return np.mean(st_values), np.max(np.abs(st_values))


def compute_q_wave_metrics(lead_signal, r_peaks, fs=FS):
    """Detect pathological Q waves: depth and duration."""
    if len(r_peaks) == 0:
        return np.nan, np.nan
    
    q_depths = []
    q_durations = []
    
    for r in r_peaks:
        # Look for Q wave in the 80ms before R-peak
        q_window = int(0.08 * fs)
        start = max(0, r - q_window)
        segment = lead_signal[start:r]
        
        if len(segment) < 3:
            continue
            
        # Q wave is a negative deflection before R
        min_idx = np.argmin(segment)
        q_depth = segment[min_idx]
        
        if q_depth < 0:
            r_amp = lead_signal[r]
            q_depths.append(abs(q_depth))
            
            # Duration: time the signal stays negative
            neg_mask = segment < 0
            if np.any(neg_mask):
                q_dur_ms = np.sum(neg_mask) / fs * 1000
                q_durations.append(q_dur_ms)
    
    mean_depth = np.mean(q_depths) if q_depths else np.nan
    mean_duration = np.mean(q_durations) if q_durations else np.nan
    return mean_depth, mean_duration


def compute_t_wave_metrics(lead_signal, r_peaks, fs=FS):
    """Compute T-wave amplitude and detect inversions."""
    if len(r_peaks) == 0:
        return np.nan, 0
    
    t_amps = []
    inversions = 0
    
    for r in r_peaks:
        # T-wave typically 200-400ms after R-peak
        t_start = r + int(0.2 * fs)
        t_end = r + int(0.4 * fs)
        
        if t_end < len(lead_signal):
            t_segment = lead_signal[t_start:t_end]
            t_amp = t_segment[np.argmax(np.abs(t_segment))]
            t_amps.append(t_amp)
            if t_amp < 0:
                inversions += 1
    
    mean_amp = np.mean(t_amps) if t_amps else np.nan
    return mean_amp, inversions


def extract_features_from_signal(signal_data, fs=FS):
    """Extract all clinical features from a 12-lead ECG signal."""
    features = {}
    
    # Filter signal
    filtered = bandpass_filter(signal_data)
    
    # Use Lead II for primary rhythm analysis
    lead_ii = filtered[:, 1]
    r_peaks = detect_r_peaks(lead_ii, fs)
    
    # --- General features ---
    features['heart_rate'] = compute_heart_rate(r_peaks, fs)
    features['n_beats'] = len(r_peaks)
    
    # R-R interval variability (HRV proxy)
    if len(r_peaks) >= 3:
        rr = np.diff(r_peaks) / fs
        rr = rr[(rr > 0.3) & (rr < 2.0)]
        features['rr_mean'] = np.mean(rr) if len(rr) > 0 else np.nan
        features['rr_std'] = np.std(rr) if len(rr) > 1 else np.nan
        features['rr_range'] = np.ptp(rr) if len(rr) > 0 else np.nan
    else:
        features['rr_mean'] = np.nan
        features['rr_std'] = np.nan
        features['rr_range'] = np.nan
    
    # QRS duration from Lead II
    features['qrs_duration_ms'] = compute_qrs_duration(lead_ii, r_peaks, fs)
    
    # QTc estimation (Bazett's formula)
    if len(r_peaks) >= 2 and not np.isnan(features.get('rr_mean', np.nan)):
        rr_sec = features['rr_mean']
        # QT interval: from Q onset to T end (~40% of RR)
        qt_est = 0.4 * rr_sec  # rough estimate
        features['qtc_bazett'] = qt_est / np.sqrt(rr_sec) if rr_sec > 0 else np.nan
    else:
        features['qtc_bazett'] = np.nan
    
    # --- HYP features: Sokolow-Lyon and Cornell ---
    # Sokolow-Lyon: S(V1) + R(V5 or V6)
    v1 = filtered[:, 6]   # V1
    v5 = filtered[:, 10]  # V5
    v6 = filtered[:, 11]  # V6
    lead_avl = filtered[:, 4]  # aVL
    
    # S-wave in V1: maximum negative deflection
    s_v1 = abs(np.min(v1))
    # R-wave in V5/V6: maximum positive deflection
    r_v5 = np.max(v5)
    r_v6 = np.max(v6)
    
    features['sokolow_lyon'] = s_v1 + max(r_v5, r_v6)
    
    # Cornell voltage: R(aVL) + S(V3)
    v3 = filtered[:, 8]
    r_avl = np.max(lead_avl)
    s_v3 = abs(np.min(v3))
    features['cornell_voltage'] = r_avl + s_v3
    
    # R-wave amplitudes per precordial lead (V1-V6)
    for i, lead_name in enumerate(['V1', 'V2', 'V3', 'V4', 'V5', 'V6']):
        lead_idx = 6 + i
        features[f'r_amp_{lead_name}'] = np.max(filtered[:, lead_idx])
        features[f's_amp_{lead_name}'] = abs(np.min(filtered[:, lead_idx]))
    
    # R/S ratio progression (V1→V6)
    r_s_ratios = []
    for i in range(6):
        lead_idx = 6 + i
        r = np.max(filtered[:, lead_idx])
        s = abs(np.min(filtered[:, lead_idx]))
        ratio = r / (s + 1e-6)
        r_s_ratios.append(ratio)
        features[f'rs_ratio_V{i+1}'] = ratio
    features['rs_transition_zone'] = np.argmax(np.array(r_s_ratios) > 1.0) + 1  # V-lead where R > S
    
    # --- MI features: Q-waves and ST elevation per lead group ---
    # Anterior leads: V1-V4
    anterior_leads = [6, 7, 8, 9]
    # Inferior leads: II, III, aVF
    inferior_leads = [1, 2, 5]
    # Lateral leads: I, aVL, V5, V6
    lateral_leads = [0, 4, 10, 11]
    
    for group_name, lead_indices in [('anterior', anterior_leads), 
                                       ('inferior', inferior_leads), 
                                       ('lateral', lateral_leads)]:
        q_depths = []
        q_durs = []
        st_devs = []
        st_maxes = []
        
        for li in lead_indices:
            lead = filtered[:, li]
            lead_r_peaks = detect_r_peaks(lead, fs) if li != 1 else r_peaks
            
            qd, qdur = compute_q_wave_metrics(lead, lead_r_peaks, fs)
            if not np.isnan(qd):
                q_depths.append(qd)
            if not np.isnan(qdur):
                q_durs.append(qdur)
                
            st_mean, st_max = compute_st_deviation(lead, lead_r_peaks, fs)
            if not np.isnan(st_mean):
                st_devs.append(st_mean)
            if not np.isnan(st_max):
                st_maxes.append(st_max)
        
        features[f'q_depth_{group_name}'] = np.mean(q_depths) if q_depths else np.nan
        features[f'q_duration_{group_name}'] = np.mean(q_durs) if q_durs else np.nan
        features[f'st_deviation_{group_name}'] = np.mean(st_devs) if st_devs else np.nan
        features[f'st_max_abs_{group_name}'] = np.max(st_maxes) if st_maxes else np.nan
    
    # --- STTC features: ST/T morphology ---
    t_inversions_total = 0
    t_amps_all = []
    st_devs_all = []
    
    for li in range(12):
        lead = filtered[:, li]
        lead_r_peaks = detect_r_peaks(lead, fs) if li != 1 else r_peaks
        
        t_amp, t_inv = compute_t_wave_metrics(lead, lead_r_peaks, fs)
        if not np.isnan(t_amp):
            t_amps_all.append(t_amp)
        t_inversions_total += t_inv
        
        st_mean, _ = compute_st_deviation(lead, lead_r_peaks, fs)
        if not np.isnan(st_mean):
            st_devs_all.append(st_mean)
    
    features['t_inversion_count'] = t_inversions_total
    features['t_amp_mean'] = np.mean(t_amps_all) if t_amps_all else np.nan
    features['t_amp_std'] = np.std(t_amps_all) if len(t_amps_all) > 1 else np.nan
    features['st_deviation_global_mean'] = np.mean(st_devs_all) if st_devs_all else np.nan
    features['st_deviation_global_std'] = np.std(st_devs_all) if len(st_devs_all) > 1 else np.nan
    features['st_deviation_global_max'] = np.max(np.abs(st_devs_all)) if st_devs_all else np.nan
    
    # --- Signal quality features ---
    for li, name in enumerate(LEAD_NAMES):
        lead = filtered[:, li]
        features[f'signal_rms_{name}'] = np.sqrt(np.mean(lead**2))
    
    # Global signal energy
    features['total_signal_energy'] = np.sum(filtered**2)
    
    return features


def main():
    metadata_path = 'data/subset_multiclass_metadata.csv'
    base_dir = 'data/raw'
    
    df = pd.read_csv(metadata_path)
    print(f"Extracting clinical features from {len(df)} records...")
    
    all_features = []
    errors = 0
    
    for i, row in df.iterrows():
        if i % 500 == 0:
            print(f"  Processing record {i}/{len(df)}...")
        
        record_path = os.path.join(base_dir, row['filename_lr'])
        try:
            record = wfdb.rdrecord(record_path)
            signal_data = record.p_signal
            
            if signal_data.shape[0] < 500 or signal_data.shape[1] != 12:
                features = {f'feature_{j}': np.nan for j in range(50)}
            else:
                features = extract_features_from_signal(signal_data)
            
            features['ecg_id'] = row['ecg_id']
            all_features.append(features)
        except Exception as e:
            errors += 1
            features = {'ecg_id': row['ecg_id']}
            all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    
    # Merge with metadata to keep labels
    features_df = features_df.merge(
        df[['ecg_id'] + [f'label_{sc}' for sc in ['NORM', 'MI', 'STTC', 'CD', 'HYP']] + ['patient_id']],
        on='ecg_id',
        how='left'
    )
    
    os.makedirs('data', exist_ok=True)
    features_df.to_csv('data/clinical_features.csv', index=False)
    
    print(f"\nExtraction complete!")
    print(f"  Records processed: {len(all_features)}")
    print(f"  Errors: {errors}")
    print(f"  Features per record: {len([c for c in features_df.columns if c not in ['ecg_id', 'patient_id'] and not c.startswith('label_')])}")
    print(f"  Saved to data/clinical_features.csv")

if __name__ == '__main__':
    main()
