#!/bin/bash

# AI Browser Agent - Startup Script

echo "🤖 AI Browser Agent"
echo "==================="
echo ""

# Check for Go
if ! command -v go &> /dev/null; then
    echo "❌ Go не установлен. Установите Go 1.21+ с https://go.dev"
    exit 1
fi


echo "✓ Go найден: $(go version)"

# Check for Chrome
if command -v google-chrome &> /dev/null; then
    echo "✓ Chrome найден: $(google-chrome --version)"
elif command -v chromium &> /dev/null; then
    echo "✓ Chromium найден: $(chromium --version)"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if [ -d "/Applications/Google Chrome.app" ]; then
        echo "✓ Chrome найден (macOS)"
    else
        echo "⚠️  Chrome не найден. Установите Google Chrome"
    fi
else
    echo "⚠️  Chrome не найден. Убедитесь, что Google Chrome установлен"
fi

# Check API key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo ""
    echo "⚠️  DEEPSEEK_API_KEY не установлен!"
    echo ""
    echo "Установите API ключ DeepSeek:"
    echo "  export DEEPSEEK_API_KEY='sk-your-key-here'"
    echo ""
    echo "Получить ключ: https://platform.deepseek.com"
    echo ""
    read -p "Введите ваш API ключ (или Enter для пропуска): " apikey
    if [ -n "$apikey" ]; then
        export DEEPSEEK_API_KEY="$apikey"
        echo "✓ API ключ установлен"
    else
        echo "⚠️  Продолжаю без API ключа (агент не будет работать)"
    fi
else
    echo "✓ DEEPSEEK_API_KEY установлен"
fi

echo ""
echo "📦 Загрузка зависимостей..."
go mod download

echo ""
echo "🚀 Запуск агента..."
echo "📍 Откройте http://localhost:8080 в браузере"
echo ""
echo "Нажмите Ctrl+C для остановки"
echo "---"
echo ""

go run .
