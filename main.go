package main

import (
	"embed"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

//go:embed static/*
var staticFiles embed.FS

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

type Server struct {
	agent   *Agent
	clients map[*websocket.Conn]bool
	mu      sync.RWMutex

	confirmMu      sync.Mutex
	pendingConfirm chan bool
}

func NewServer() *Server {
	// Поддержка разных переменных окружения для API ключа
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("DEEPSEEK_API_KEY")
	}
	if apiKey == "" {
		apiKey = os.Getenv("AI_API_KEY")
	}

	if apiKey == "" {
		log.Println("⚠️  API ключ не установлен!")
		log.Println("   Для OpenRouter: set OPENROUTER_API_KEY=sk-or-...")
		log.Println("   Для DeepSeek:   set DEEPSEEK_API_KEY=sk-...")
	} else {
		keyPreview := apiKey[:8] + "..." + apiKey[len(apiKey)-4:]
		log.Printf("✓ API ключ установлен: %s", keyPreview)
	}

	// Показываем настройки API
	apiURL := os.Getenv("AI_API_URL")
	if apiURL == "" {
		apiURL = "https://openrouter.ai/api/v1/chat/completions"
	}
	model := os.Getenv("AI_MODEL")
	if model == "" {
		model = "deepseek/deepseek-chat"
	}
	log.Printf("✓ API URL: %s", apiURL)
	log.Printf("✓ Модель: %s", model)

	s := &Server{
		clients: make(map[*websocket.Conn]bool),
	}

	s.agent = NewAgent(apiKey, s.broadcast, s.requestConfirmation)
	return s
}

func (s *Server) broadcast(msg Message) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	for client := range s.clients {
		if err := client.WriteJSON(msg); err != nil {
			log.Printf("Ошибка отправки: %v", err)
		}
	}
}

func (s *Server) requestConfirmation(action, details string) bool {
	s.broadcast(Message{
		Type:    "confirmation_request",
		Content: fmt.Sprintf("⚠️ Подтвердите действие: %s\n%s", action, details),
	})

	// Ждём ответа от UI. Если подтверждение не пришло — считаем, что действие отменено.
	s.confirmMu.Lock()
	// если уже был pending confirm, заменим его (старый больше не актуален)
	if s.pendingConfirm != nil {
		close(s.pendingConfirm)
	}
	ch := make(chan bool, 1)
	s.pendingConfirm = ch
	s.confirmMu.Unlock()

	select {
	case ok, more := <-ch:
		if !more {
			return false
		}
		return ok
	case <-time.After(2 * time.Minute):
		s.confirmMu.Lock()
		if s.pendingConfirm == ch {
			close(s.pendingConfirm)
			s.pendingConfirm = nil
		}
		s.confirmMu.Unlock()
		return false
	}
}

func (s *Server) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("Ошибка WebSocket: %v", err)
		return
	}
	defer conn.Close()

	s.mu.Lock()
	s.clients[conn] = true
	s.mu.Unlock()

	defer func() {
		s.mu.Lock()
		delete(s.clients, conn)
		s.mu.Unlock()
	}()

	s.broadcast(Message{Type: "status", Content: "🔌 Подключено к агенту"})

	for {
		var msg struct {
			Type    string `json:"type"`
			Content string `json:"content"`
		}

		if err := conn.ReadJSON(&msg); err != nil {
			if websocket.IsCloseError(err, websocket.CloseGoingAway, websocket.CloseNormalClosure) {
				return
			}
			log.Printf("Ошибка чтения: %v", err)
			return
		}

		log.Printf("Получено сообщение: type=%s", msg.Type)

		switch msg.Type {
		case "task":
			log.Printf("Запуск задачи: %s", msg.Content)
			go s.agent.ExecuteTask(msg.Content)
		case "stop":
			log.Println("Остановка агента")
			s.agent.Stop()
		case "close_browser":
			log.Println("Закрытие браузера")
			s.agent.CloseBrowser()
		case "confirmation_response":
			log.Printf("Ответ подтверждения: %s", msg.Content)
			confirmed := strings.TrimSpace(strings.ToLower(msg.Content)) == "yes"
			s.confirmMu.Lock()
			if s.pendingConfirm != nil {
				// не блокируемся даже если агент уже ушёл
				select {
				case s.pendingConfirm <- confirmed:
				default:
				}
				close(s.pendingConfirm)
				s.pendingConfirm = nil
			}
			s.confirmMu.Unlock()
		}
	}
}

func main() {
	server := NewServer()

	// Статические файлы
	http.Handle("/static/", http.FileServer(http.FS(staticFiles)))
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		data, err := staticFiles.ReadFile("static/index.html")
		if err != nil {
			http.Error(w, "File not found", 404)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(data)
	})

	// WebSocket
	http.HandleFunc("/ws", server.handleWebSocket)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	fmt.Println("")
	fmt.Println("========================================")
	fmt.Println("   🤖 AI Browser Agent")
	fmt.Println("========================================")
	fmt.Println("")
	fmt.Printf("   📍 Откройте: http://localhost:%s\n", port)
	fmt.Println("")
	fmt.Println("   Нажмите Ctrl+C для остановки")
	fmt.Println("========================================")
	fmt.Println("")

	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatal(err)
	}
}
