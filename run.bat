@echo off
chcp 65001 >nul 2>nul

echo.
echo ========================================
echo    AI Browser Agent
echo ========================================
echo.


:: Check for Go
where go >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Go is not installed. Install Go 1.21+ from https://go.dev
    pause
    exit /b 1
)

echo [OK] Go found
go version
echo.

:: Check for Chrome
where chrome >nul 2>nul
if %ERRORLEVEL% neq 0 (
    if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
        echo [OK] Chrome found in Program Files
    ) else if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
        echo [OK] Chrome found in Program Files x86
    ) else if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" (
        echo [OK] Chrome found in LocalAppData
    ) else (
        echo [WARNING] Chrome not found. Make sure Google Chrome is installed.
    )
) else (
    echo [OK] Chrome found
)

echo.

:: Check API key (support multiple env vars)
if "%OPENROUTER_API_KEY%"=="" (
    if "%DEEPSEEK_API_KEY%"=="" (
        if "%AI_API_KEY%"=="" (
            echo [WARNING] API key is not set!
            echo.
            echo For OpenRouter: set OPENROUTER_API_KEY=sk-or-v1-...
            echo For DeepSeek:   set DEEPSEEK_API_KEY=sk-...
            echo.
            echo Get OpenRouter key at: https://openrouter.ai/keys
            echo.
            set /p apikey="Enter your OpenRouter API key (or press Enter to skip): "
            if not "!apikey!"=="" (
                set OPENROUTER_API_KEY=!apikey!
                echo [OK] API key set
            ) else (
                echo [WARNING] Continuing without API key - agent will not work
            )
        ) else (
            echo [OK] AI_API_KEY is set
        )
    ) else (
        echo [OK] DEEPSEEK_API_KEY is set
    )
) else (
    echo [OK] OPENROUTER_API_KEY is set
)

echo.
echo [INFO] Downloading dependencies...
go mod download
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to download dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo    Starting AI Browser Agent...
echo ========================================
echo.
echo    Open http://localhost:8080 in your browser
echo.
echo    Press Ctrl+C to stop
echo.
echo ========================================
echo.

go run .

pause
