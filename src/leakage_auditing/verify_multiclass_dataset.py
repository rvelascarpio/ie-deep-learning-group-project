import pandas as pd

def main():
    metadata_path = 'data/subset_multiclass_metadata.csv'
    df = pd.read_csv(metadata_path)
    
    print("=== MULTICLASS LEAKAGE AUDIT ===")
    print(f"Total records: {len(df)}")
    
    unique_patients = df['patient_id'].nunique()
    print(f"Unique patient_ids: {unique_patients}")
    
    if len(df) == unique_patients:
        print("✅ NO PATIENT OVERLAP: Every record is a unique patient. Zero cross-fold leakage guaranteed for MVP!")
    else:
        duplicates = len(df) - unique_patients
        print(f"❌ LEAKAGE WARNING: Found {duplicates} overlapping patient records.")
        
    print("\nClass Distribution:")
    for sc in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
        print(f"  {sc}: {df[f'label_{sc}'].sum()}")

if __name__ == '__main__':
    main()
