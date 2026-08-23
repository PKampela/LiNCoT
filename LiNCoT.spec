from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_data_files
# -*- mode: python ; coding: utf-8 -*-

mne_datas = collect_data_files("mne")
gui_datas = [
    ("gui/assets", "gui/assets"),
]

a = Analysis(
    ['launcher.py'],
    pathex=['.'],
    binaries=[],
    datas=mne_datas + gui_datas,
    hiddenimports=[
    *collect_submodules("cli"),
    *collect_submodules("core"),
    *collect_submodules("gui"),
    *collect_submodules("registry"),
    *collect_submodules("backends"),
    *collect_submodules("mne")
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LiNCoT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LiNCoT',
)
