# PyInstaller one-dir spec for the Familiar Python backend.
#
# Why PyInstaller (not Nuitka): the backend leans on dynamic imports
# (uvicorn lifespan/loop/protocol plugins, lazy discord/fastembed/sqlite_vec
# imports, pydantic plugin loading). PyInstaller's import analysis + hooks
# handle these; Nuitka compiles (slower builds, more fragile with namespace
# packages and native extensions like onnxruntime/PyNaCl). One-dir (not
# one-file) keeps startup fast and tracebacks debuggable.
#
# Build from the repo root with the project venv:
#   .venv/bin/python -m PyInstaller packaging/backend.spec
# Output: packaging/appimage-work/backend/dist/familiar-backend/
import os

REPO = os.path.dirname(SPECPATH)

a = Analysis(
    [os.path.join(REPO, 'dmd', 'server.py')],
    pathex=[REPO],
    binaries=[],
    datas=[
        (os.path.join(REPO, 'web'), 'web'),
        (os.path.join(REPO, 'config.example.yaml'), '.'),
    ],
    hiddenimports=[
        # uvicorn plugin surface (selected by string at runtime)
        'uvicorn.lifespan.on',
        'uvicorn.lifespan.off',
        'uvicorn.loops.auto',
        'uvicorn.loops.asyncio',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.http.h11_impl',
        'uvicorn.protocols.http.httptools_impl',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.protocols.websockets.websockets_impl',
        'uvicorn.protocols.websockets.websockets_sansio_impl',
        'uvicorn.protocols.websockets.wsproto_impl',
        # lazy/gated imports elsewhere in dmd/
        'discord',
        'discord.voice',
        'discord.sinks',
        'discord.opus',
        'discord.errors',
        'nacl',
        'fastembed',
        'sqlite_vec',
        'soundfile',
        'websockets.legacy',
        'websockets.legacy.protocol',
        'multipart',
        'yaml',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # dmd/ never imports these; they leak in from user-site packages via
    # over-broad hooks. Excluding keeps the bundle small and honest.
    excludes=[
        'tkinter', 'unittest', 'pydoc', 'doctest',
        'torch', 'torchvision', 'torchaudio', 'triton',
        'transformers',
        'nvidia', 'scipy', 'pandas', 'matplotlib', 'pyarrow',
        'yt_dlp', 'IPython', 'jupyter', 'notebook',
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='familiar-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # logs visible when launched from a terminal
    disable_windowed_traceback=False,
    argv_emulation=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='familiar-backend',
)
