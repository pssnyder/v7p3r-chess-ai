# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['S:\\Maker Stuff\\Programming\\Chess Engines\\V7P3R Chess AI\\v7p3r-chess-ai\\tournament_build\\v7p3rai_uci_main.py'],
    pathex=[],
    binaries=[],
    datas=[('S:\\Maker Stuff\\Programming\\Chess Engines\\V7P3R Chess AI\\v7p3r-chess-ai\\tournament_build\\v3.0', 'v3.0'), ('S:\\Maker Stuff\\Programming\\Chess Engines\\V7P3R Chess AI\\v7p3r-chess-ai\\tournament_build\\models', 'models'), ('S:\\Maker Stuff\\Programming\\Chess Engines\\V7P3R Chess AI\\v7p3r-chess-ai\\tournament_build\\config', 'config')],
    hiddenimports=['chess_core', 'v3.0.src.ai.thinking_brain', 'v3.0.src.ai.gameplay_brain'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=2,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [('O', None, 'OPTION'), ('O', None, 'OPTION')],
    name='V7P3RAI_v3.0',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
