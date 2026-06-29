import pandas as pd
import ast

def main():
    scp_df = pd.read_csv('scp_statements.csv', index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1.0]
    scp_to_class = scp_df['diagnostic_class'].dropna().to_dict()

    df = pd.read_csv('data/subset_metadata_2000.csv')
    df['scp_codes_dict'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
    
    superclasses = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    def get_superclasses(scp_codes_dict):
        classes = set()
        for code, conf in scp_codes_dict.items():
            if code in scp_to_class:
                if conf > 50:
                    classes.add(scp_to_class[code])
        return list(classes)
        
    df['diagnostic_classes'] = df['scp_codes_dict'].apply(get_superclasses)
    
    for sc in superclasses:
        df[f'label_{sc}'] = df['diagnostic_classes'].apply(lambda x: 1 if sc in x else 0)
        
    print(f"Distribution of the {len(df)} already-downloaded records:")
    for sc in superclasses:
        print(f"  {sc}: {df[f'label_{sc}'].sum()}")

    # Save it as the multiclass metadata!
    df.to_csv("data/subset_multiclass_metadata.csv", index=False)
    print("Overwrote subset_multiclass_metadata.csv with the fast 2000 records.")

if __name__ == '__main__':
    main()
