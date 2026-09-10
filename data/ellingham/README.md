# Ellingham workbook data

`ellingham_data.xlsx` stores one worksheet per family:

- `oxides`
- `carbides`
- `nitrides`
- `fluorides`
- `chlorides`
- `hydrides`
- `sulfides`
- `tellurides`
- `selenides`
- `iodides`
- `bromides`

Each sheet uses the same columns:

- `phase_code`: metal-state/compound-state code (`ss`, `ls`, `gs`, `sl`, `ll`, `gl`, `sg`, `lg`, `gg`)
- `T0`, `T1`: segment endpoint temperatures in **Kelvin**
- `G0`, `G1`: segment endpoint ΔG values in **kcal/mol** of the family reference gas (or per mol C for carbides)
- `reaction`: matplotlib mathtext label string used by `ellingham.py`
- `label_offset`: y-offset for the start label (`0` for XLS-derived families; preserved manual values for carbides)
- `element`: first element symbol in the reaction, used for filtering/reporting

`ellingham.py` keeps these source units in the workbook and still performs the existing one-time K→°C and kcal→kJ conversion after loading. Family sheets are loaded lazily: the script only parses the worksheet(s) requested by `--families` instead of reading all 11 up front.

## Sources and regeneration

- `oxides`, `nitrides`, `fluorides`, `chlorides`, `hydrides`, `sulfides`, `tellurides`, `selenides`, `iodides`, and `bromides` are parsed from `EllinghamMaker_v12-5.xls`.
- `carbides` has **no XLS source**. That sheet was transcribed once from Coltters (1985) and is preserved verbatim when regenerating the workbook.
- The source workbook's `Oxide Suboxide` and `Sulfides Sub` sheets are supplementary detail tables and are intentionally skipped.

Regenerate the XLS-backed sheets with:

```bash
python3 tools/parse_ellingham_xls.py
```

The parser rewrites `data/ellingham/ellingham_data.xlsx`, refreshing the 10 XLS-backed family sheets while preserving the existing `carbides` sheet.
