
import pandas as pd
import numpy as np

def remove_highly_correlated(df, variables, threshold=0.9):
    """
    Rimuove le variabili altamente correlate basandosi su un valore di soglia.
    
    Parametri:
      df (pd.DataFrame): Il DataFrame di input.
      variables (list): Lista di colonne da analizzare.
      threshold (float): Soglia di correlazione oltre la quale si rimuove una variabile.
                         
    Ritorna:
     Lista delle feature selezionate e lista delle feature rimosse.
    """
    remaining_vars = list(variables)
    removed_vars = []
    
    while True:
        # Calcolo matrice di correlazione
        sub_df = df[remaining_vars]
        corr_matrix = sub_df.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)  # Imposta la diagonale a zero (evita autovalori)
        
        # Trova la massima correlazione tra due variabili
        max_corr = corr_matrix.to_numpy().max()
        
        if max_corr < threshold:
            break
        
        # Trova l'indice della prima coppia con correlazione massima
        max_indices = np.where(corr_matrix == max_corr)
        i, j = max_indices[0][0], max_indices[1][0]
        var1, var2 = corr_matrix.index[i], corr_matrix.columns[j]
        
        # Calcola la media delle correlazioni per entrambe le variabili
        avg_corr_var1 = corr_matrix.loc[var1].mean()
        avg_corr_var2 = corr_matrix.loc[var2].mean()
        
        # Decide quale variabile eliminare
        drop_var = var1 if avg_corr_var1 >= avg_corr_var2 else var2
        remaining_vars.remove(drop_var)
        removed_vars.append(drop_var)
    
    return remaining_vars, removed_vars


# --- Caricamento del dataset ---
file_path = '/Users/gabriele/Desktop/TESI/CMS-databehaviour_males-females 1.xlsx'
sheets = ["maschi", "femmine"]  # Nomi dei fogli di lavoro

for sheet in sheets:
    print(f"\n--- Analisi per {sheet} ---")

    # Caricamento del foglio specifico
    df = pd.read_excel(file_path, sheet_name=sheet, header=2, usecols="C:Q")  
    # Pulizia nomi colonne
    df.columns = df.columns.str.strip()

    # Stampa colonne per verifica
    print("Colonne del dataset:", df.columns.tolist())

    # Definizione delle feature (senza target)
    feature_columns = ['body weight', 'sucrose intake', 'NOR index', 'locomotor activity', 
                       'social interaction time', 'social events', '0P (entries)', 'CL (entries)', 
                       '% OP', 'tOP', 'tCL', 'tCENT', 't%OP']

    # Pulizia dataset (rimuove righe con valori mancanti)
    df_clean = df.dropna(subset=feature_columns)

    # Rimozione delle feature altamente correlate
    selected_features, removed_features = remove_highly_correlated(df_clean, feature_columns, threshold=0.8)

    # Salvataggio delle feature selezionate in un nuovo CSV
    output_path = f"/Users/gabriele/Desktop/TESI/selected_features_{sheet.lower()}.csv"
    df_clean[selected_features].to_csv(output_path, index=False)

    print("\nFeature selezionate dopo correlation analysis:", selected_features)
    print("Feature rimosse durante correlation analysis:", removed_features)
    print(f"File salvato: {output_path}")
