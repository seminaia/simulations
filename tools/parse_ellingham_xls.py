#!/usr/bin/env python3
"""Parse EllinghamMaker_v12-5.xls into structured multi-sheet XLSX data.

The source workbook stores ten reaction families in repeating 4-column blocks
(separated by blank spacer columns). Each 8-row block contains one reaction's
piecewise-linear data with optional state-transition markers:
  M/B -> metal melts / boils
  m/b -> compound melts / boils
  X/x -> breakpoint only (no state change; reported for review)

The generated workbook keeps one sheet per family with columns:
  phase_code, T0, T1, G0, G1, reaction, label_offset, element

Values intentionally stay in source units:
  T in K, G in kcal/mol of the family reference gas
  (O2, N2, F2, Cl2, Br2, I2, H2, S2, Se2, Te2) or per mol C for carbides.

Carbides has no XLS source; its existing sheet is preserved when regenerating
`data/ellingham/ellingham_data.xlsx`.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from matplotlib.mathtext import MathTextParser

SHEET_TO_FAMILY = {
    'Oxides': 'oxides',
    'Nitrides': 'nitrides',
    'Fluorides': 'fluorides',
    'Chlorides': 'chlorides',
    'Hydrides': 'hydrides',
    'Sulfides': 'sulfides',
    'Tellurides': 'tellurides',
    'Selenides': 'selenides',
    'Iodides': 'iodides',
    'Bromides': 'bromides',
}
FAMILY_REFERENCE_GASES = {
    'oxides': 'O2',
    'carbides': 'C',
    'nitrides': 'N2',
    'fluorides': 'F2',
    'chlorides': 'Cl2',
    'hydrides': 'H2',
    'sulfides': 'S2',
    'tellurides': 'Te2',
    'selenides': 'Se2',
    'iodides': 'I2',
    'bromides': 'Br2',
}
FAMILY_ORDER = [
    'oxides',
    'carbides',
    'nitrides',
    'fluorides',
    'chlorides',
    'hydrides',
    'sulfides',
    'tellurides',
    'selenides',
    'iodides',
    'bromides',
]
OUTPUT_COLUMNS = ['phase_code', 'T0', 'T1', 'G0', 'G1', 'reaction', 'label_offset', 'element']
PHASE_CODES = ('ss', 'ls', 'gs', 'sl', 'll', 'gl', 'sg', 'lg', 'gg')
BLOCK_HEIGHT = 8
MATH_PARSER = MathTextParser('path')


@dataclass(frozen=True)
class Point:
    row: int
    temperature_k: float
    gibbs_kcal: float
    marker: str | None


@dataclass(frozen=True)
class Segment:
    family: str
    phase_code: str
    T0: float
    T1: float
    G0: float
    G1: float
    reaction: str
    label_offset: float
    element: str


MARKER_TRANSITIONS = {
    'M': ('metal', 'l'),
    'B': ('metal', 'g'),
    'm': ('compound', 'l'),
    'b': ('compound', 'g'),
    'X': ('none', None),
    'x': ('none', None),
}


def discover_group_starts(df: pd.DataFrame) -> list[int]:
    starts = []
    for col in range(df.shape[1]):
        values = df.iloc[:, col].dropna().astype(str)
        if values.str.startswith('Part-').any():
            starts.append(col - 1)
    return starts


def is_block_start(df: pd.DataFrame, row: int, col0: int) -> bool:
    if col0 < 0 or col0 + 3 >= df.shape[1] or row + 1 >= df.shape[0]:
        return False
    idx = df.iat[row, col0]
    symbol = df.iat[row, col0 + 1]
    t_header = df.iat[row, col0 + 2]
    return (
        pd.notna(idx)
        and isinstance(idx, (int, float))
        and float(idx).is_integer()
        and isinstance(symbol, str)
        and re.fullmatch(r'[A-Z][a-z]?|[A-Z]', symbol) is not None
        and str(t_header).strip() == 'T, K'
    )


def parse_numeric(value: object) -> float:
    text = str(value).strip().replace('−', '-')
    if re.fullmatch(r'\.\d+(?:\.\d+)?', text):
        text = '-' + text[1:]
    return float(text)


def iter_blocks(df: pd.DataFrame) -> Iterable[tuple[int, int, str, str, list[Point]]]:
    for col0 in discover_group_starts(df):
        for row in range(df.shape[0]):
            if not is_block_start(df, row, col0):
                continue
            symbol = str(df.iat[row, col0 + 1]).strip()
            reaction = str(df.iat[row + 1, col0 + 1]).strip()
            points: list[Point] = []
            for point_row in range(row + 1, min(row + BLOCK_HEIGHT, df.shape[0])):
                t_value = df.iat[point_row, col0 + 2]
                g_value = df.iat[point_row, col0 + 3]
                if pd.isna(t_value) or pd.isna(g_value):
                    continue
                marker_value = df.iat[point_row, col0 + 1]
                marker = marker_value.strip() if isinstance(marker_value, str) else None
                points.append(
                    Point(
                        row=point_row,
                        temperature_k=parse_numeric(t_value),
                        gibbs_kcal=parse_numeric(g_value),
                        marker=marker,
                    )
                )
            yield row, col0, symbol, reaction, points


def transition_state(marker: str, metal_state: str, compound_state: str) -> tuple[str, str]:
    who, new_state = MARKER_TRANSITIONS[marker]
    if who == 'metal':
        return new_state, compound_state
    if who == 'compound':
        return metal_state, new_state
    return metal_state, compound_state


def subscript_formula(formula: str) -> str:
    formula = formula.replace(' ', '')
    return re.sub(r'([A-Za-z\)])(\d+(?:\.\d+)?)', lambda m: f"{m.group(1)}_{{{m.group(2)}}}", formula)


def format_coefficient(coef: str) -> str:
    coef = coef.strip()
    if not coef:
        return ''
    if '/' in coef:
        num, den = coef.split('/', 1)
        return rf'\frac{{{num}}}{{{den}}}'
    return coef


def format_term(term: str) -> str:
    term = term.strip()
    if not term:
        return term
    match = re.fullmatch(r'(?P<coef>(?:\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)?)(?P<formula>[A-Za-z].*)', term)
    if not match:
        return term
    coef = match.group('coef') or ''
    formula = subscript_formula(match.group('formula'))
    if coef:
        return f"{format_coefficient(coef)} {formula}"
    return formula


def to_mathtext_reaction(reaction: str) -> str:
    normalized = re.sub(r'\s+', '', reaction)
    pieces = re.split(r'(\+|=)', normalized)
    rendered = []
    for piece in pieces:
        if piece == '+':
            rendered.append(' + ')
        elif piece == '=':
            rendered.append(' = ')
        elif piece:
            rendered.append(format_term(piece))
    label = '$' + ''.join(rendered) + '$'
    MATH_PARSER.parse(label)
    return label


def segments_from_points(family: str, symbol: str, reaction: str, points: list[Point]) -> tuple[list[Segment], list[str]]:
    if len(points) < 2:
        return [], []
    reaction_label = to_mathtext_reaction(reaction)
    metal_state = 's'
    compound_state = 's'
    segments: list[Segment] = []
    breakpoint_markers: list[str] = []
    for left, right in zip(points, points[1:]):
        phase_code = metal_state + compound_state
        if phase_code not in PHASE_CODES:
            raise ValueError(f'Unexpected phase code {phase_code!r} for {family}:{symbol}:{reaction}')
        segments.append(
            Segment(
                family=family,
                phase_code=phase_code,
                T0=left.temperature_k,
                T1=right.temperature_k,
                G0=left.gibbs_kcal,
                G1=right.gibbs_kcal,
                reaction=reaction_label,
                label_offset=0.0,
                element=symbol,
            )
        )
        if right.marker:
            if right.marker not in MARKER_TRANSITIONS:
                raise ValueError(f'Unknown marker {right.marker!r} in {family}:{symbol}:{reaction}')
            if right.marker in {'X', 'x'}:
                breakpoint_markers.append(right.marker)
            metal_state, compound_state = transition_state(right.marker, metal_state, compound_state)
    return segments, breakpoint_markers


def segments_to_dataframe(segments: list[Segment]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'phase_code': segment.phase_code,
                'T0': segment.T0,
                'T1': segment.T1,
                'G0': segment.G0,
                'G1': segment.G1,
                'reaction': segment.reaction,
                'label_offset': segment.label_offset,
                'element': segment.element,
            }
            for segment in segments
        ],
        columns=OUTPUT_COLUMNS,
    )


def parse_sheet(workbook: Path, sheet_name: str) -> tuple[list[Segment], dict[str, set[str]], list[tuple[str, str]]]:
    family = SHEET_TO_FAMILY[sheet_name]
    df = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
    segments: list[Segment] = []
    phase_by_element: dict[str, set[str]] = defaultdict(set)
    breakpoints: list[tuple[str, str]] = []
    for _, _, symbol, reaction, points in iter_blocks(df):
        new_segments, bp_markers = segments_from_points(family, symbol, reaction, points)
        segments.extend(new_segments)
        for segment in new_segments:
            phase_by_element[segment.element].add(segment.phase_code)
        for marker in bp_markers:
            breakpoints.append((symbol, marker))
    return segments, phase_by_element, breakpoints


def read_preserved_sheets(out_path: Path, generated_families: set[str]) -> dict[str, pd.DataFrame]:
    if not out_path.exists():
        return {}
    preserved: dict[str, pd.DataFrame] = {}
    with pd.ExcelFile(out_path) as workbook:
        for sheet_name in workbook.sheet_names:
            if sheet_name not in generated_families:
                preserved[sheet_name] = workbook.parse(sheet_name=sheet_name)
    return preserved


def write_workbook(out_path: Path, family_frames: dict[str, pd.DataFrame], preserved_frames: dict[str, pd.DataFrame]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_frames = {**preserved_frames, **family_frames}
    ordered_sheets = [family for family in FAMILY_ORDER if family in all_frames]
    ordered_sheets.extend(sheet for sheet in all_frames if sheet not in ordered_sheets)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for family in ordered_sheets:
            all_frames[family].to_excel(writer, sheet_name=family, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description='Parse EllinghamMaker_v12-5.xls into structured XLSX family data.')
    parser.add_argument('--workbook', default='EllinghamMaker_v12-5.xls', help='Path to the .xls workbook')
    parser.add_argument('--out-dir', default='data/ellingham', help='Directory for generated XLSX data')
    parser.add_argument(
        '--out-file', default='ellingham_data.xlsx',
        help='Output workbook filename (written under --out-dir)'
    )
    args = parser.parse_args()

    workbook = Path(args.workbook)
    out_dir = Path(args.out_dir)
    out_path = out_dir / args.out_file

    generated_families = set(SHEET_TO_FAMILY.values())
    preserved_frames = read_preserved_sheets(out_path, generated_families)

    family_frames: dict[str, pd.DataFrame] = {}
    all_breakpoints: dict[str, list[tuple[str, str]]] = {}
    total_generated_rows = 0
    for sheet_name, family in SHEET_TO_FAMILY.items():
        segments, _, breakpoints = parse_sheet(workbook, sheet_name)
        frame = segments_to_dataframe(segments)
        family_frames[family] = frame
        all_breakpoints[sheet_name] = breakpoints
        total_generated_rows += len(frame)
        distinct_elements = sorted(frame['element'].unique()) if not frame.empty else []
        gas = FAMILY_REFERENCE_GASES[family]
        print(f'{sheet_name:10s} -> {len(frame):4d} segments, {len(distinct_elements):3d} elements, reference {gas}')

    write_workbook(out_path, family_frames, preserved_frames)
    preserved_list = ', '.join(sorted(preserved_frames)) if preserved_frames else 'none'
    print(
        f'-> Wrote {out_path} '
        f'({total_generated_rows + sum(len(df) for df in preserved_frames.values())} total rows; '
        f'preserved sheets: {preserved_list})'
    )

    for sheet_name, breakpoints in all_breakpoints.items():
        if breakpoints:
            detail = ', '.join(f'{symbol}:{marker}' for symbol, marker in breakpoints)
            print(f'{sheet_name}: breakpoint-only markers treated as no state change -> {detail}')


if __name__ == '__main__':
    main()
