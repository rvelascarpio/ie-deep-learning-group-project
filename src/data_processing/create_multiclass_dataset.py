import pandas as pd
import ast
import os

def main():
    print("Reading scp_statements.csv...")
    scp_df = pd.read_csv('scp_statements.csv', index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1.0]
    
    # Mapping from detailed scp code to 5 superclasses (NORM, MI, STTC, CD, HYP)
    scp_to_class = scp_df['diagnostic_class'].dropna().to_dict()
    
    print("Reading ptbxl_database.csv...")
    df = pd.read_csv('ptbxl_database.csv')
    df['scp_codes_dict'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
    
    # Globally drop duplicate patient_id to guarantee no patient exists in multiple rows
    df = df.drop_duplicates(subset=['patient_id'])
    
    # Define our 5 target superclasses
    superclasses = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    def get_superclasses(scp_codes_dict):
        classes = set()
        for code, conf in scp_codes_dict.items():
            if code in scp_to_class:
                if conf > 50: # Only confident diagnoses
                    classes.add(scp_to_class[code])
        return list(classes)
        
    df['diagnostic_classes'] = df['scp_codes_dict'].apply(get_superclasses)
    
    # Drop rows that don't belong to any of the 5 superclasses
    df = df[df['diagnostic_classes'].apply(len) > 0]
    
    # Create multi-label columns
    for sc in superclasses:
        df[f'label_{sc}'] = df['diagnostic_classes'].apply(lambda x: 1 if sc in x else 0)
        
    print(f"Total records with a valid superclass (before signal check): {len(df)}")
    
    # --- KEY CHANGE: Use ALL records that have signal files on disk ---
    base_dir = 'data/raw'
    valid_rows = []
    missing = 0
    for _, row in df.iterrows():
        file_path = os.path.join(base_dir, row['filename_lr'] + '.dat')
        if os.path.exists(file_path):
            valid_rows.append(row)
        else:
            missing += 1
            
    subset = pd.DataFrame(valid_rows).reset_index(drop=True)
    
    # Shuffle the final dataset
    subset = subset.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print(f"\nSignals found on disk: {len(subset)}")
    print(f"Signals missing (not downloaded): {missing}")
    print(f"\nFinal dataset: {len(subset)} unique patient records")
    for sc in superclasses:
        print(f"  {sc} count: {subset[f'label_{sc}'].sum()}")
        
    os.makedirs('data', exist_ok=True)
    subset.to_csv("data/subset_multiclass_metadata.csv", index=False)
    print("\nSaved to data/subset_multiclass_metadata.csv")

if __name__ == '__main__':
    main()
