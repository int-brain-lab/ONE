#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi, Miles Wells
# @Date: Monday, January 21st 2019, 6:28:49 pm
from datetime import datetime
from pathlib import Path
from typing import Union


def subjects_data_folder(folder: Path, rglob: bool = False) -> Path:
    """Given a root_data_folder will try to find a 'Subjects' data folder.
    If Subjects folder is passed will return it directly."""
    if not isinstance(folder, Path):
        folder = Path(folder)
    if rglob:
        func = folder.rglob
    else:
        func = folder.glob

    # Try to find Subjects folder one level
    if folder.name.lower() != 'subjects':
        # Try to find Subjects folder if folder.glob
        spath = [x for x in func('*') if x.name.lower() == 'subjects']
        if not spath:
            raise ValueError('No "Subjects" folder in children folders')
        elif len(spath) > 1:
            raise ValueError(f'Multiple "Subjects" folder in children folders: {spath}')
        else:
            folder = folder / spath[0]

    return folder


def remove_empty_folders(folder: Union[str, Path]) -> None:
    """Will iteratively remove any children empty folders"""
    all_folders = sorted(x for x in Path(folder).rglob('*') if x.is_dir())
    for f in reversed(all_folders):  # Reversed sorted ensures we remove deepest first
        try:
            f.rmdir()
        except Exception:
            continue


def _isdatetime(s: str) -> bool:
    try:
        datetime.strptime(s, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def session_path(path: Union[str, Path]) -> str:
    """Returns the session path from any filepath if the date/number pattern is found"""
    path = Path(path)
    sess = None
    for i, p in enumerate(path.parts):
        if p.isdigit() and _isdatetime(path.parts[i - 1]):
            sess = str(Path().joinpath(*path.parts[:i + 1]))

    return sess


def session_name(path: Union[str, Path]) -> str:
    """Returns the session name (subject/date/number) string for any filepath
    using session_path"""
    path = Path(path)
    return '/'.join(Path(session_path(path)).parts[-3:])


def next_num_folder(session_date_folder: str) -> str:
    """Return the next number for a session given a session_date_folder"""
    session_date_folder = Path(session_date_folder)
    if not session_date_folder.exists():
        return '001'
    session_nums = [
        int(x.name) for x in session_date_folder.iterdir()
        if x.is_dir() and not x.name.startswith('.') and x.name.isdigit()
    ]
    out = f'{max(session_nums or [0]) + 1:03d}'
    assert len(out) == 3, 'ALF spec does not support session numbers > 999'
    return out
