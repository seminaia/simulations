"""Look for melting-point information anywhere in the v6 workbook."""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PATH = next(p for p in (HERE / 'data' / 'ellingham' / 'Data_EllinghamDiagram_v6.xlsx',
                        HERE / 'Data_EllinghamDiagram_v6.xlsx') if p.exists())
PATTERN = re.compile(r'melt|fusion|liquid|\bTm\b|T_m|solid', re.I)

print('File:', PATH)
for sheet, df in pd.read_excel(PATH, sheet_name=None, header=None).items():
    print(f'\n=== {sheet}  {df.shape} ===')
    print('First rows:')
    print(df.iloc[:4, :14].to_string())
    hits = 0
    for r, c in zip(*df.apply(lambda col: col.astype(str).str.contains(PATTERN)).to_numpy().nonzero()):
        print(f'  match row {r} col {c}: {str(df.iat[r, c])[:120]}')
        hits += 1
        if hits >= 25:
            print('  ... (more matches)')
            break
    if not hits:
        print('  no melt/liquid/solid text found')

info = pd.read_excel(PATH, sheet_name='info-reactions')
print('\ninfo-reactions columns:', list(info.columns))
for col in info.columns:
    print(f'\n--- {col} (first 5 distinct) ---')
    print(info[col].dropna().astype(str).drop_duplicates().head(5).to_string(index=False))
