import pandas as pd

# Datos Reales obtenidos de tus pruebas anteriores
data_english = {
    "Métrica": ["Accuracy", "F1-Score (Macro)", "MCC (Matthews)"],
    "Resultado": [0.724, 0.727, 0.568],
    "Interpretación": ["Estado del Arte (>70%)", "Balanceado", "Moderado"]
}

data_spanish = {
    "Métrica": ["Accuracy", "F1-Score (Macro)", "MCC (Matthews)"],
    "Resultado": [0.778, 0.776, 0.668],
    "Interpretación": ["Superior al Base (>75%)", "Muy Balanceado", "Alto (Robusto)"]
}

# Configuración visual de Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')

df_en = pd.DataFrame(data_english)
df_es = pd.DataFrame(data_spanish)

print("\n" + "="*60)
print("🇺🇸 TABLA: RESULTADOS MODELO INGLÉS (TweetEval)")
print("   Modelo: twitter-roberta-base-sentiment-latest")
print("="*60)
print(df_en.to_string(index=False))
print("\n\n")

print("="*60)
print("🇪🇨 TABLA: RESULTADOS MODELO ESPAÑOL (TweetSentMult)")
print("   Modelo: robertuito-finetuned (Optimizado)")
print("="*60)
print(df_es.to_string(index=False))
print("="*60 + "\n")