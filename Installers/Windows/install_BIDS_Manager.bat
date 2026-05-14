@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

rem =======================================================================
rem BIDS-Manager Installer, Windows (x86_64)
rem Version:  1.0.1
rem License:  MIT, ANCP Lab, University of Oldenburg
rem Homepage: https://github.com/ANCPLabOldenburg/BIDS-Manager
rem
rem Downloads an embeddable Python 3.10 runtime, creates an isolated
rem virtual environment, installs bids-manager from PyPI, and places a
rem Desktop shortcut (.lnk with icon) and a Start Menu entry.
rem
rem Icon source: the installed bids-manager package ships its logo at
rem   bidsmgr/gui/assets/logo.png
rem The installer builds an .ico on the fly via Pillow (multi-resolution
rem 256/128/64/32/16). A pre-built AppIcon.ico in the package is used
rem directly if present.
rem =======================================================================

rem -- Configuration ---------------------------------------------------
set "INSTALLER_VERSION=1.0.1"
set "APP_NAME=BIDS-Manager"
set "BASEDIR=%USERPROFILE%\BIDS-Manager"
set "PYDIR=%BASEDIR%\python310"
set "ENVDIR=%BASEDIR%\env"
set "LOG_FILE=%BASEDIR%\install.log"
set "PYPI_PKG=bids-manager"
set "ICO_FILE=%BASEDIR%\bids-manager.ico"
set "ZIPURL=https://raw.githubusercontent.com/ANCPLabOldenburg/BIDS-Manager/main/external/python-embed/python-3.10.11-embed-amd64.zip"

rem Determine Desktop and Start Menu paths
set "DESKTOP="
for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`) do set "DESKTOP=%%D"
set "STARTMENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs\BIDS-Manager"

rem -- Banner --------------------------------------------------------
cls
echo.
echo     +==================================================+
echo     ^|                                                  ^|
echo     ^|   B I D S - M a n a g e r   I n s t a l l e r    ^|
echo     ^|                                                  ^|
echo     ^|    Schema-driven BIDS converter and editor       ^|
echo     ^|        ANCP Lab - University of Oldenburg        ^|
echo     ^|                                                  ^|
echo     +==================================================+
echo.
echo   Installer version:  %INSTALLER_VERSION%
echo   Platform:           Windows (x86_64)
echo   Install directory:  %BASEDIR%
echo.


rem -- Pre-flight: existing installation --------------------------------
set "UPGRADE_MODE=0"
if exist "%ENVDIR%\Scripts\activate.bat" (
    echo   [W] An existing BIDS-Manager installation was detected at:
    echo       %BASEDIR%
    echo.
    echo   [u] Upgrade: keep settings, reinstall the package
    echo   [r] Reinstall: remove everything and start fresh
    echo   [a] Abort
    echo.
)
if exist "%ENVDIR%\Scripts\activate.bat" goto :ASK_CHOICE
goto :AFTER_CHOICE

:ASK_CHOICE
set /p "CHOICE=  Your choice [u/r/a]: "
if /i "!CHOICE!"=="u" (
    echo   [*] Upgrading existing installation...
    set "UPGRADE_MODE=1"
    goto :AFTER_CHOICE
)
if /i "!CHOICE!"=="r" (
    echo   [*] Removing existing installation...
    del /q "%DESKTOP%\BIDS-Manager.lnk" 2>nul
    del /q "%DESKTOP%\Uninstall BIDS-Manager.lnk" 2>nul
    rd /s /q "%STARTMENU%" 2>nul
    rmdir /s /q "%BASEDIR%" 2>nul
    set "UPGRADE_MODE=0"
    goto :AFTER_CHOICE
)
if /i "!CHOICE!"=="a" (
    echo   Installation aborted.
    pause
    exit /b 0
)
echo   Please enter u, r, or a.
goto :ASK_CHOICE

:AFTER_CHOICE

rem -- Confirmation ----------------------------------------------------
echo.
echo   The installer will:
echo     1. Download an embeddable Python 3.10 runtime (~12 MB)
echo     2. Create an isolated virtual environment
echo     3. Install bids-manager and all dependencies (~1.5 GB)
echo     4. Create a Desktop shortcut and Start Menu entry with icon
echo        BIDS-Manager opens in a terminal window so you can follow its progress.
echo.
set /p "CONFIRM=  Continue with installation? [Y/n] "
if /i "!CONFIRM!"=="n" (
    echo   Installation cancelled.
    pause
    exit /b 0
)

rem -- Prepare ---------------------------------------------------------
if not exist "%BASEDIR%" mkdir "%BASEDIR%"
if %ERRORLEVEL% NEQ 0 (
    echo   [X] ERROR: Could not create base folder.
    pause
    exit /b 1
)
cd /d "%BASEDIR%"

echo BIDS-Manager installation started at %DATE% %TIME% > "%LOG_FILE%"
echo Installer version: %INSTALLER_VERSION% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

rem =====================================================================
rem Step 1/5 - Download embeddable Python
rem =====================================================================
if "%UPGRADE_MODE%"=="1" if exist "%PYDIR%\python.exe" (
    echo   [OK] Portable Python already present, skipping download.
    goto :SKIP_PYTHON_DL
)

echo   [*] Step 1/5 - Downloading embeddable Python 3.10...

where curl.exe >nul 2>&1
if %ERRORLEVEL%==0 (
    curl.exe -fSL --retry 3 --retry-delay 5 -o python310.zip "%ZIPURL%" >> "%LOG_FILE%" 2>&1
) else (
    powershell -NoProfile -Command "Invoke-WebRequest -Uri '%ZIPURL%' -OutFile 'python310.zip'" >> "%LOG_FILE%" 2>&1
)
if not exist "python310.zip" (
    echo   [X] ERROR: Download failed. Please check your internet connection.
    echo       See log: %LOG_FILE%
    pause
    exit /b 1
)

echo   [*] Step 2/5 - Extracting Python...
powershell -NoProfile -Command "Expand-Archive -Force 'python310.zip' 'python310'" >> "%LOG_FILE%" 2>&1
if not exist "%PYDIR%\python.exe" (
    echo   [X] ERROR: Extraction failed, python.exe not found.
    pause
    exit /b 1
)
del /q python310.zip
echo   [OK] Python 3.10 ready.

:SKIP_PYTHON_DL

rem =====================================================================
rem Step 3/5 - Bootstrap pip and create virtual environment
rem =====================================================================
echo   [*] Step 3/5 - Bootstrapping pip and creating virtual environment...

powershell -NoProfile -Command "(Get-Content '%PYDIR%\python310._pth') -replace '^#\s*import site','import site' | Set-Content '%PYDIR%\python310._pth'" >> "%LOG_FILE%" 2>&1

if not exist "%PYDIR%\Scripts\pip.exe" (
    echo       Downloading get-pip.py...
    where curl.exe >nul 2>&1
    if %ERRORLEVEL%==0 (
        curl.exe -fSL -o "%PYDIR%\get-pip.py" "https://bootstrap.pypa.io/get-pip.py" >> "%LOG_FILE%" 2>&1
    ) else (
        powershell -NoProfile -Command "Invoke-WebRequest 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PYDIR%\get-pip.py'" >> "%LOG_FILE%" 2>&1
    )
    if not exist "%PYDIR%\get-pip.py" (
        echo   [X] ERROR: Could not download get-pip.py.
        pause
        exit /b 1
    )
    "%PYDIR%\python.exe" "%PYDIR%\get-pip.py" >> "%LOG_FILE%" 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo   [X] ERROR: pip bootstrap failed.
        pause
        exit /b 1
    )
    del /q "%PYDIR%\get-pip.py"
)

echo       Creating virtual environment...
"%PYDIR%\python.exe" -m pip install virtualenv >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   [X] ERROR: virtualenv installation failed.
    pause
    exit /b 1
)
"%PYDIR%\python.exe" -m virtualenv "%ENVDIR%" >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   [X] ERROR: Virtual environment creation failed.
    pause
    exit /b 1
)
echo   [OK] Virtual environment ready.

rem =====================================================================
rem Step 4/5 - Install bids-manager
rem =====================================================================
echo   [*] Step 4/5 - Installing bids-manager (this may take several minutes)...
call "%ENVDIR%\Scripts\activate.bat"

echo       Upgrading pip...
python -m pip install --upgrade pip setuptools wheel >> "%LOG_FILE%" 2>&1

echo       Installing %PYPI_PKG%...
python -m pip install %PYPI_PKG% >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   [X] ERROR: bids-manager installation failed.
    echo       See log: %LOG_FILE%
    pause
    exit /b 1
)
echo   [OK] bids-manager installed successfully.

rem -- App icon --------------------------------------------------------
echo   [*] Locating app icon...
set "PKG_PATH="
if exist "%ICO_FILE%" del /q "%ICO_FILE%"

if not exist "%ENVDIR%\Scripts\python.exe" goto :ICON_NO_PY
"%ENVDIR%\Scripts\python.exe" -c "import bidsmgr,os; print(os.path.dirname(bidsmgr.__file__))" > "%TEMP%\bidsmgr_pkgpath.txt" 2>nul
for /f "usebackq tokens=*" %%P in ("%TEMP%\bidsmgr_pkgpath.txt") do set "PKG_PATH=%%P"
del /q "%TEMP%\bidsmgr_pkgpath.txt" 2>nul

if not defined PKG_PATH goto :ICON_NO_PKG

rem Prefer a pre-built .ico if one ever ships in the package.
if exist "%PKG_PATH%\gui\assets\windows\AppIcon.ico" (
    copy /y "%PKG_PATH%\gui\assets\windows\AppIcon.ico" "%ICO_FILE%" >nul
    echo   [OK] App icon installed from bundled assets (AppIcon.ico).
    goto :ICON_DONE
)

rem Fallback: build a multi-resolution .ico from bidsmgr/gui/assets/logo.png
rem using Pillow (already in the venv, pulled in by matplotlib / mne).
if not exist "%PKG_PATH%\gui\assets\logo.png" goto :ICON_NO_LOGO
echo   [*] Building app icon from package logo via Pillow...
"%ENVDIR%\Scripts\python.exe" -c "from PIL import Image; im=Image.open(r'%PKG_PATH%\gui\assets\logo.png').convert('RGBA'); w,h=im.size; side=max(w,h); canv=Image.new('RGBA',(side,side),(0,0,0,0)); canv.paste(im,((side-w)//2,(side-h)//2)); canv=canv.resize((256,256), Image.LANCZOS); canv.save(r'%ICO_FILE%', sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)])" >> "%LOG_FILE%" 2>&1
if exist "%ICO_FILE%" (
    echo   [OK] App icon built from package logo.
) else (
    echo   [W] Icon build failed, shortcuts will use the default icon.
)
goto :ICON_DONE

:ICON_NO_LOGO
echo   [W] bidsmgr/gui/assets/logo.png not found, shortcuts will use the default icon.
goto :ICON_DONE

:ICON_NO_PKG
echo   [W] Could not locate package path, shortcuts will use the default icon.
goto :ICON_DONE

:ICON_NO_PY
echo   [W] Python not found, shortcuts will use the default icon.

:ICON_DONE

rem =====================================================================
rem Step 5/5 - Create launcher, uninstaller, Desktop and Start Menu
rem =====================================================================
echo   [*] Step 5/5 - Creating shortcuts...

rem -- Launcher bat ----------------------------------------------------
echo   [*] Creating run_BIDS_Manager.bat...
(
    echo @echo off
    echo title BIDS-Manager
    echo call "%ENVDIR%\Scripts\activate.bat"
    echo bidsmgr %%*
) > "%BASEDIR%\run_BIDS_Manager.bat"
if exist "%BASEDIR%\run_BIDS_Manager.bat" (
    echo   [OK] run_BIDS_Manager.bat created.
) else (
    echo   [X] ERROR: Could not create run_BIDS_Manager.bat
    pause
    exit /b 1
)

rem -- Uninstaller bat (written inline, no subroutine) -----------------
echo   [*] Creating uninstall_BIDS_Manager.bat...
(
    echo @echo off
    echo chcp 65001 ^>nul 2^>^&1
    echo setlocal
    echo echo.
    echo echo   BIDS-Manager Uninstaller
    echo echo.
    echo echo   This will remove BIDS-Manager from %BASEDIR%
    echo echo.
    echo set /p "CONF=  Are you sure? [y/N] "
    echo if /i "%%CONF%%"=="y" goto :DO_UNINSTALL
    echo echo   Uninstallation cancelled.
    echo pause
    echo exit /b 0
    echo :DO_UNINSTALL
    echo echo   [*] Removing Desktop shortcuts...
    echo del /q "%DESKTOP%\BIDS-Manager.lnk" 2^>nul
    echo del /q "%DESKTOP%\Uninstall BIDS-Manager.lnk" 2^>nul
    echo echo   [*] Removing Start Menu entry...
    echo rd /s /q "%STARTMENU%" 2^>nul
    echo echo   [*] Removing installation directory...
    echo rd /s /q "%BASEDIR%" 2^>nul
    echo ie4uinit.exe -show ^>nul 2^>^&1
    echo echo   BIDS-Manager has been completely removed.
    echo pause
) > "%BASEDIR%\uninstall_BIDS_Manager.bat"
if exist "%BASEDIR%\uninstall_BIDS_Manager.bat" (
    echo   [OK] uninstall_BIDS_Manager.bat created.
) else (
    echo   [X] ERROR: Could not create uninstall_BIDS_Manager.bat
    pause
    exit /b 1
)

rem -- Start Menu folder -----------------------------------------------
if not exist "%STARTMENU%" mkdir "%STARTMENU%"

rem -- Create shortcuts via temp .ps1 file ------------------------------
echo   [*] Writing PowerShell shortcut script...
(
    echo $WshShell = New-Object -ComObject WScript.Shell
    echo.
    echo $lnk = $WshShell.CreateShortcut('%DESKTOP%\BIDS-Manager.lnk'^)
    echo $lnk.TargetPath = '%BASEDIR%\run_BIDS_Manager.bat'
    echo $lnk.WorkingDirectory = '%BASEDIR%'
    echo $lnk.Description = 'Launch BIDS-Manager'
    echo if (Test-Path '%ICO_FILE%'^) { $lnk.IconLocation = '%ICO_FILE%,0' }
    echo $lnk.Save(^)
    echo.
    echo $lnk = $WshShell.CreateShortcut('%DESKTOP%\Uninstall BIDS-Manager.lnk'^)
    echo $lnk.TargetPath = '%BASEDIR%\uninstall_BIDS_Manager.bat'
    echo $lnk.WorkingDirectory = '%BASEDIR%'
    echo $lnk.Description = 'Uninstall BIDS-Manager'
    echo $lnk.Save(^)
    echo.
    echo $lnk = $WshShell.CreateShortcut('%STARTMENU%\BIDS-Manager.lnk'^)
    echo $lnk.TargetPath = '%BASEDIR%\run_BIDS_Manager.bat'
    echo $lnk.WorkingDirectory = '%BASEDIR%'
    echo $lnk.Description = 'Launch BIDS-Manager'
    echo if (Test-Path '%ICO_FILE%'^) { $lnk.IconLocation = '%ICO_FILE%,0' }
    echo $lnk.Save(^)
) > "%TEMP%\bidsmgr_shortcuts.ps1"

echo   [*] Running PowerShell shortcut script...
powershell -NoProfile -ExecutionPolicy Bypass -File "%TEMP%\bidsmgr_shortcuts.ps1"
set "PS_ERR=%ERRORLEVEL%"
del /q "%TEMP%\bidsmgr_shortcuts.ps1" 2>nul

if %PS_ERR% NEQ 0 (
    echo   [W] PowerShell returned error %PS_ERR%, shortcuts may be missing.
) else (
    echo   [OK] PowerShell script completed.
)

rem -- Verify shortcuts ------------------------------------------------
set "SHORTCUTS_OK=1"
if exist "%DESKTOP%\BIDS-Manager.lnk" (
    echo   [OK] Desktop shortcut created.
) else (
    echo   [W] Desktop shortcut could not be created.
    set "SHORTCUTS_OK=0"
)
if not exist "%DESKTOP%\Uninstall BIDS-Manager.lnk" set "SHORTCUTS_OK=0"
if not exist "%STARTMENU%\BIDS-Manager.lnk" set "SHORTCUTS_OK=0"

if "!SHORTCUTS_OK!"=="0" (
    echo   [W] Some shortcuts could not be created.
    echo       You can still launch BIDS-Manager manually: %BASEDIR%\run_BIDS_Manager.bat
)

rem -- Refresh Windows icon cache --------------------------------------
ie4uinit.exe -show >nul 2>&1

rem =====================================================================
rem Done.
rem =====================================================================
echo.
echo   =======================================================
echo     BIDS-Manager was successfully installed!
echo   =======================================================
echo.
echo   Launch BIDS-Manager:
echo     * Double-click BIDS-Manager on your Desktop
echo     * Or search BIDS-Manager in the Start Menu
echo     A terminal window will open showing BIDS-Manager's progress.
echo.
echo   Uninstall:    Double-click  Uninstall BIDS-Manager  on your Desktop
echo   Log file:     %LOG_FILE%
echo.
echo   Thank you for using BIDS-Manager!
echo.
pause
exit /b 0
