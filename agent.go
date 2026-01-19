package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/chromedp/cdproto/dom"
	"github.com/chromedp/cdproto/network"
	"github.com/chromedp/cdproto/page"
	"github.com/chromedp/cdproto/target"
	"github.com/chromedp/chromedp"
)

type Message struct {
	Type    string `json:"type"`
	Content string `json:"content"`
}

type Agent struct {
	apiKey           string
	broadcast        func(Message)
	requestConfirm   func(string, string) bool
	conversationHist []ChatMessage

	// Decomposed goals for the current task (helps avoid merging multiple requested items)
	goalChecklist  []string
	currentGoalIdx int
	goalDone       []bool

	// Goal guard: avoid duplicate "add to cart" actions.
	// If the current checklist goal looks like adding one item to cart, we capture a lightweight
	// cart signature before we start working on that goal. As soon as the signature changes,
	// we auto-mark the goal as completed (even if the LLM forgets to set goal_completed),
	// preventing repeated clicks that add duplicates/extra items.
	goalCartArmed    bool
	goalCartGoalIdx  int
	goalCartBaseline string
	goalCartBaseCtr  uint64

	// Cart mutation detector. We use this as the primary signal that an item was added
	// (works on modern SPAs where DOM-based "extract" is unreliable).
	cartEventCounter atomic.Uint64

	// Browser state
	allocCtx      context.Context
	allocCancel   context.CancelFunc
	browserCtx    context.Context
	browserCancel context.CancelFunc
	browserReady  bool
	browserMu     sync.Mutex

	// Task state
	isRunning    bool
	taskStartURL string
	stopChan     chan struct{}
	taskMu       sync.Mutex
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type AIRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
	MaxTokens   int           `json:"max_tokens"`
}

type OpenRouterRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
	MaxTokens   int           `json:"max_tokens"`
}

type TaskDecomposition struct {
	Goals []string `json:"goals"`
	Notes string   `json:"notes,omitempty"`
}

type AIResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
		Code    int    `json:"code"`
	} `json:"error"`
}

type BrowserAction struct {
	Action   string `json:"action"`
	Selector string `json:"selector,omitempty"`
	Value    string `json:"value,omitempty"`
	URL      string `json:"url,omitempty"`
	Reason   string `json:"reason"`
}

type AgentResponse struct {
	Thinking    string        `json:"thinking"`
	Action      BrowserAction `json:"action"`
	IsComplete  bool          `json:"is_complete"`
	FinalReport string        `json:"final_report,omitempty"`
	// When processing a multi-item checklist, set this to true ONLY when the CURRENT checklist item is completed.
	GoalCompleted bool   `json:"goal_completed,omitempty"`
	CompletedGoal string `json:"completed_goal,omitempty"`
	NeedsConfirm  bool   `json:"needs_confirm,omitempty"`
	ConfirmMsg    string `json:"confirm_message,omitempty"`
}

// SearchTargets describes candidates on the current page to run an in-page search.
// Many sites (especially food delivery) have a menu search input inside the restaurant page.
// We find it dynamically from the DOM (no hardcoded selectors) and use it as a robust fallback.
type SearchTargets struct {
	Open  string `json:"open"`
	Input string `json:"input"`
}

func NewAgent(apiKey string, broadcast func(Message), confirm func(string, string) bool) *Agent {
	return &Agent{
		apiKey:           apiKey,
		broadcast:        broadcast,
		requestConfirm:   confirm,
		conversationHist: make([]ChatMessage, 0),
		goalChecklist:    nil,
		currentGoalIdx:   0,
		goalDone:         nil,
	}
}

func (a *Agent) Stop() {
	a.taskMu.Lock()
	if a.isRunning && a.stopChan != nil {
		close(a.stopChan)
	}
	a.isRunning = false
	a.taskMu.Unlock()

	a.broadcast(Message{Type: "status", Content: "⏹️ Агент остановлен"})
}

func (a *Agent) CloseBrowser() {
	a.browserMu.Lock()
	defer a.browserMu.Unlock()

	if a.browserCancel != nil {
		a.browserCancel()
	}
	if a.allocCancel != nil {
		a.allocCancel()
	}
	a.browserCtx = nil
	a.browserCancel = nil
	a.allocCtx = nil
	a.allocCancel = nil
	a.browserReady = false

	a.broadcast(Message{Type: "status", Content: "🔒 Браузер закрыт"})
}

func (a *Agent) initBrowser() error {
	a.browserMu.Lock()
	defer a.browserMu.Unlock()

	// Если браузер уже запущен и работает
	if a.browserReady && a.browserCtx != nil {
		// Проверяем, жив ли контекст
		select {
		case <-a.browserCtx.Done():
			// Контекст мёртв, нужно перезапустить
			a.browserReady = false
		default:
			// Контекст жив, проверяем браузер
			testCtx, cancel := context.WithTimeout(a.browserCtx, 3*time.Second)
			var url string
			err := chromedp.Run(testCtx, chromedp.Location(&url))
			cancel()
			if err == nil {
				return nil // Браузер работает
			}
			a.browserReady = false
		}
	}

	// Закрываем старые контексты если есть
	if a.browserCancel != nil {
		a.browserCancel()
	}
	if a.allocCancel != nil {
		a.allocCancel()
	}

	// Определяем путь для данных Chrome
	userDataDir := filepath.Join(os.TempDir(), "ai-browser-agent-chrome")
	if cwd, err := os.Getwd(); err == nil {
		userDataDir = filepath.Join(cwd, "chrome-user-data")
	}

	// Создаём директорию
	if err := os.MkdirAll(userDataDir, 0755); err != nil {
		a.broadcast(Message{Type: "error", Content: fmt.Sprintf("⚠️ Ошибка создания директории: %v", err)})
	}

	a.broadcast(Message{Type: "status", Content: fmt.Sprintf("📂 Chrome данные: %s", userDataDir)})

	// Опции Chrome
	opts := append(chromedp.DefaultExecAllocatorOptions[:],
		chromedp.Flag("headless", false),
		chromedp.Flag("disable-gpu", false),
		chromedp.Flag("enable-automation", false),
		chromedp.Flag("disable-extensions", false),
		chromedp.Flag("no-first-run", true),
		chromedp.Flag("no-default-browser-check", true),
		chromedp.Flag("disable-background-networking", false),
		chromedp.Flag("disable-sync", true),
		chromedp.Flag("disable-translate", true),
		chromedp.Flag("mute-audio", true),
		chromedp.Flag("disable-infobars", true),
		chromedp.Flag("disable-features", "TranslateUI"),
		chromedp.Flag("disable-blink-features", "AutomationControlled"),
		chromedp.UserDataDir(userDataDir),
		chromedp.WindowSize(1400, 900),
	)

	// Создаём allocator context (живёт пока браузер открыт)
	a.allocCtx, a.allocCancel = chromedp.NewExecAllocator(context.Background(), opts...)

	// Создаём browser context
	a.browserCtx, a.browserCancel = chromedp.NewContext(a.allocCtx,
		chromedp.WithLogf(func(s string, i ...interface{}) {
			// Подавляем лишние логи
		}),
	)

	// --- Anti-tab-spam layer ---
	// По умолчанию держим ОДНУ вкладку, чтобы агент не плодил десятки новых табов
	// (частая причина: кликабельные карточки/ссылки с target=_blank на сайтах доставки).
	// Можно разрешить вкладки, установив AI_ALLOW_NEW_TABS=true|1.
	allowTabs := strings.ToLower(strings.TrimSpace(os.Getenv("AI_ALLOW_NEW_TABS")))
	allowNewTabs := allowTabs == "1" || allowTabs == "true" || allowTabs == "yes"

	// В chromedp TargetID в момент NewContext ещё может быть не инициализирован,
	// поэтому определяем mainTargetID ЛЕНИВО по первому созданному page-target.
	var mainTargetMu sync.Mutex
	var mainTargetID target.ID

	// Слушаем события браузера (не только текущего таба), чтобы ловить любые новые вкладки.
	chromedp.ListenBrowser(a.browserCtx, func(ev interface{}) {
		if allowNewTabs {
			return
		}
		switch e := ev.(type) {
		case *target.EventTargetCreated:
			if e.TargetInfo == nil || e.TargetInfo.Type != "page" {
				return
			}
			tid := e.TargetInfo.TargetID

			mainTargetMu.Lock()
			if mainTargetID == "" {
				// Первый page-target считаем главным и оставляем.
				mainTargetID = tid
				mainTargetMu.Unlock()
				return
			}
			// Любой другой page-target закрываем.
			isMain := tid == mainTargetID
			mainTargetMu.Unlock()
			if isMain {
				return
			}

			ctx, cancel := context.WithTimeout(a.browserCtx, 3*time.Second)
			defer cancel()
			_ = chromedp.Run(ctx, chromedp.ActionFunc(func(ctx context.Context) error {
				return target.CloseTarget(tid).Do(ctx)
			}))
		}
	})

	// Дополнительно: переопределяем window.open и клики по <a target=_blank>,
	// чтобы открывать всё в текущей вкладке (срабатывает даже если вкладка НЕ создаётся напрямую).
	antiTabScript := `(function(){
		// Максимально агрессивно запрещаем открытие новых вкладок/окон.
		// Это нужно, потому что на доставках еды клики по карточкам/баннерам часто открывают target=_blank.
		try {
			var origOpen = window.open;
			window.open = function(url){
				try { if (url) { window.location.href = url; } } catch(e) {}
				return null;
			};
			// на всякий случай блокируем сохранённые ссылки на open
			try { Object.defineProperty(window, 'open', { value: window.open, writable: false, configurable: false }); } catch(e) {}
		} catch(e) {}
		function normalizeAnchor(a){
			try {
				if (!a) return;
				var tgt = (a.getAttribute('target')||'').toLowerCase();
				if (tgt && tgt !== '_self') a.setAttribute('target','_self');
				// rel=noopener/noreferrer часто сопровождает target=_blank
				if (a.getAttribute('rel')) a.removeAttribute('rel');
			} catch(e) {}
		}
		try {
			// почистим уже существующие ссылки
			Array.prototype.forEach.call(document.querySelectorAll('a[target]'), normalizeAnchor);
			// и будущие тоже
			var mo = new MutationObserver(function(muts){
				for (var i=0;i<muts.length;i++){
					var m = muts[i];
					if (m.addedNodes) {
						for (var j=0;j<m.addedNodes.length;j++){
							var n = m.addedNodes[j];
							if (!n || !n.querySelectorAll) continue;
							Array.prototype.forEach.call(n.querySelectorAll('a[target]'), normalizeAnchor);
						}
					}
					if (m.target && m.target.tagName && m.target.tagName.toLowerCase()==='a') normalizeAnchor(m.target);
				}
			});
			mo.observe(document.documentElement || document.body, {subtree:true, childList:true, attributes:true, attributeFilter:['target','rel']});
		} catch(e) {}
		try {
			// перехватываем клики по ссылкам, которые пытаются открыть новую вкладку
			document.addEventListener('click', function(ev){
				var t = ev.target;
				if (!t || !t.closest) return;
				var a = t.closest('a[href]');
				if (!a || !a.href) return;
				var tgt = (a.getAttribute('target')||'').toLowerCase();
				if (tgt && tgt !== '_self') {
					ev.preventDefault();
					ev.stopPropagation();
					window.location.href = a.href;
					return;
				}
				// на всякий случай блокируем ctrl/meta клик
				if (ev.ctrlKey || ev.metaKey) {
					ev.preventDefault();
					ev.stopPropagation();
					window.location.href = a.href;
					return;
				}
			}, true);
		} catch(e) {}
	})();`

	// Cart-mutation hook (JS fallback). Some SPAs do not expose a stable DOM for cart content.
	// We track cart-related network calls via fetch/XHR and expose a monotonic counter.
	cartHookScript := `(function(){
		try {
			if (window.__cartHooked) return;
			window.__cartHooked = true;
			window.__CART_MUTATIONS = 0;
			function mark(url){
				try {
					var u = String(url||'').toLowerCase();
					if (!u) return;
					var isCart = (u.indexOf('cart')>=0 || u.indexOf('basket')>=0 || u.indexOf('checkout')>=0);
					if (!isCart && u.indexOf('order')>=0) {
						if (u.indexOf('item')>=0 || u.indexOf('items')>=0 || u.indexOf('position')>=0 || u.indexOf('positions')>=0 || u.indexOf('line')>=0 || u.indexOf('basket')>=0 || u.indexOf('cart')>=0) isCart = true;
					}
					if (isCart) window.__CART_MUTATIONS = (window.__CART_MUTATIONS||0) + 1;
				} catch(e) {}
			}
			var origFetch = window.fetch;
			if (origFetch) {
				window.fetch = async function(){
					var url = arguments && arguments.length ? arguments[0] : '';
					var res = await origFetch.apply(this, arguments);
					mark(url);
					return res;
				}
			}
			try {
				var origOpen = XMLHttpRequest.prototype.open;
				var origSend = XMLHttpRequest.prototype.send;
				XMLHttpRequest.prototype.open = function(m,u){
					try { this.__url = u; } catch(e) {}
					return origOpen.apply(this, arguments);
				};
				XMLHttpRequest.prototype.send = function(){
					try {
						this.addEventListener('load', function(){ mark(this.__url); });
					} catch(e) {}
					return origSend.apply(this, arguments);
				};
			} catch(e) {}
		} catch(e) {}
	})();`

	a.broadcast(Message{Type: "status", Content: "🌐 Запуск браузера..."})

	// Запускаем браузер с начальной страницей
	// Listen to network traffic on the active target to detect cart mutations.
	// We watch request/response URLs that look cart-related. This is more reliable than
	// trying to extract cart DOM from modern SPAs.
	chromedp.ListenTarget(a.browserCtx, func(ev interface{}) {
		switch e := ev.(type) {
		case *network.EventRequestWillBeSent:
			if e.Request != nil {
				m := strings.ToUpper(e.Request.Method)
				if (m == "POST" || m == "PUT" || m == "PATCH" || m == "DELETE") && isCartishURL(e.Request.URL) {
					a.cartEventCounter.Add(1)
				}
			}
		case *network.EventResponseReceived:
			if isCartishURL(e.Response.URL) {
				a.cartEventCounter.Add(1)
			}
		}
	})

	err := chromedp.Run(a.browserCtx,
		chromedp.ActionFunc(func(ctx context.Context) error {
			// ВАЖНО: без discover targets CDP не присылает TargetCreated события,
			// и анти-таб слой не сможет закрывать новые вкладки.
			// Это ключевой фикс "миллион вкладок".
			return target.SetDiscoverTargets(true).Do(ctx)
		}),
		chromedp.ActionFunc(func(ctx context.Context) error {
			// Инжектим скрипт в каждый новый документ (навигации/SPA переходы)
			_, err := page.AddScriptToEvaluateOnNewDocument(antiTabScript).Do(ctx)
			if err != nil {
				return err
			}
			_, err = page.AddScriptToEvaluateOnNewDocument(cartHookScript).Do(ctx)
			return err
		}),
		network.Enable(),
		chromedp.Navigate("about:blank"),
		// И сразу применяем к текущей странице
		chromedp.Evaluate(antiTabScript, nil),
		chromedp.Evaluate(cartHookScript, nil),
	)
	if err != nil {
		a.allocCancel()
		a.browserCancel()
		a.browserCtx = nil
		a.allocCtx = nil
		return fmt.Errorf("ошибка запуска браузера: %v", err)
	}

	// Даём браузеру время на инициализацию
	time.Sleep(1 * time.Second)

	a.browserReady = true
	a.broadcast(Message{Type: "status", Content: "✅ Браузер запущен"})
	return nil
}

func (a *Agent) ExecuteTask(task string) {
	a.taskMu.Lock()
	if a.isRunning {
		a.taskMu.Unlock()
		a.broadcast(Message{Type: "error", Content: "⚠️ Агент уже выполняет задачу"})
		return
	}
	a.isRunning = true
	a.stopChan = make(chan struct{})
	a.taskMu.Unlock()

	defer func() {
		a.taskMu.Lock()
		a.isRunning = false
		a.taskMu.Unlock()
	}()

	a.broadcast(Message{Type: "status", Content: "🚀 Начинаю выполнение задачи..."})
	a.broadcast(Message{Type: "task", Content: task})

	// Инициализация браузера
	if err := a.initBrowser(); err != nil {
		a.broadcast(Message{Type: "error", Content: fmt.Sprintf("❌ %v", err)})
		return
	}

	// Сбрасываем историю для новой задачи
	a.conversationHist = []ChatMessage{
		{Role: "system", Content: a.getSystemPrompt()},
		{Role: "user", Content: fmt.Sprintf("Задача пользователя: %s", task)},
	}

	// ВСЕГДА сбрасываем чеклист/гарды между задачами.
	// Иначе если декомпозиция не вернёт целей (например, "просто открой сайт"),
	// у агента может остаться старый чеклист, и он будет считать задачу незавершённой.
	a.goalChecklist = nil
	a.currentGoalIdx = 0
	a.goalDone = nil
	a.goalCartArmed = false
	a.goalCartGoalIdx = -1
	a.goalCartBaseline = ""
	a.goalCartBaseCtr = 0
	// Reset cart mutation counter for a clean per-task baseline.
	a.cartEventCounter.Store(0)

	// Capture the starting URL for this task (used to auto-close simple navigation goals).
	a.taskStartURL = ""
	if u, err := a.getCurrentURL(); err == nil {
		a.taskStartURL = strings.TrimSpace(u)
	}

	// Декомпозируем задачу в атомарные цели, чтобы агент не склеивал несколько позиций в одну.
	if goals, err := a.decomposeTaskGoals(task); err == nil && len(goals) > 0 {
		a.goalChecklist = goals
		a.currentGoalIdx = 0
		a.goalDone = make([]bool, len(goals))
		var b strings.Builder
		b.WriteString("Чеклист целей (не объединяй разные позиции; закрой все пункты перед завершением):\n")
		for i, g := range goals {
			b.WriteString(fmt.Sprintf("%d) %s\n", i+1, g))
		}
		b.WriteString("\nПравило выполнения чеклиста: работай СТРОГО по порядку. Сейчас можно пытаться выполнить только текущий пункт.\n" +
			"Когда убедишься, что текущий пункт выполнен (например, блюдо добавлено в корзину), верни goal_completed:true и completed_goal с текстом пункта.\n" +
			"Не переходи к следующему пункту, пока текущий не закрыт.\n")
		a.conversationHist = append(a.conversationHist, ChatMessage{Role: "user", Content: b.String()})
		a.broadcast(Message{Type: "status", Content: "🧾 Сформировал чеклист целей из запроса"})
	}

	// Основной цикл агента
	maxSteps := 50
	consecutiveErrors := 0
	maxConsecutiveErrors := 5

	for step := 0; step < maxSteps; step++ {
		// Проверяем остановку
		select {
		case <-a.stopChan:
			a.broadcast(Message{Type: "status", Content: "⏹️ Задача отменена пользователем"})
			return
		default:
		}

		a.broadcast(Message{Type: "step", Content: fmt.Sprintf("📍 Шаг %d/%d", step+1, maxSteps)})

		// Пауза перед получением состояния (даём странице загрузиться)
		time.Sleep(500 * time.Millisecond)

		// Получаем состояние страницы
		pageState, err := a.getPageState()
		if err != nil {
			consecutiveErrors++
			a.broadcast(Message{Type: "error", Content: fmt.Sprintf("⚠️ Ошибка получения состояния: %v", err)})

			if consecutiveErrors >= maxConsecutiveErrors {
				a.broadcast(Message{Type: "error", Content: "❌ Слишком много ошибок подряд, останавливаюсь"})
				return
			}

			time.Sleep(2 * time.Second)
			continue
		}
		consecutiveErrors = 0

		// Arm "add-to-cart" guard once per checklist goal, so a single successful add
		// auto-completes the goal and prevents duplicate additions.
		a.armGoalCartGuard()

		// Спрашиваем AI что делать
		response, err := a.askAI(pageState)
		if err != nil {
			a.broadcast(Message{Type: "error", Content: fmt.Sprintf("❌ Ошибка AI: %v", err)})
			time.Sleep(3 * time.Second)
			continue
		}

		// Логируем мышление агента
		if response.Thinking != "" {
			a.broadcast(Message{Type: "thinking", Content: fmt.Sprintf("🤔 %s", response.Thinking)})
		}

		// Проверяем завершение
		if response.IsComplete {
			// Если есть чеклист целей, не даём завершить задачу, пока все пункты не закрыты.
			if len(a.goalChecklist) > 0 {
				allDone := true
				for _, v := range a.goalDone {
					if !v {
						allDone = false
						break
					}
				}
				if !allDone {
					// Auto-close a single simple navigation goal if we clearly navigated away from the task start URL.
					// This prevents the "opened site but checklist not closed" loop for tasks like "открой Яндекс Еду".
					if len(a.goalChecklist) == 1 && !a.goalDone[0] && a.isNavigationGoal() {
						curURL, _ := a.getCurrentURL()
						curURL = strings.TrimSpace(curURL)
						if curURL != "" && !strings.HasPrefix(curURL, "about:") && a.taskStartURL != "" && curURL != a.taskStartURL {
							cur := a.goalChecklist[0]
							a.goalDone[0] = true
							a.currentGoalIdx = 1
							a.broadcast(Message{Type: "status", Content: fmt.Sprintf("✅ Чеклист: навигационный пункт выполнен автоматически — %s", cur)})
							a.conversationHist = append(a.conversationHist, ChatMessage{Role: "user", Content: fmt.Sprintf("Мы уже перешли на нужную страницу (URL изменился). Автоматически закрываем пункт чеклиста: %s.", cur)})
							allDone = true
						}
					}
					if !allDone {
						a.broadcast(Message{Type: "status", Content: "ℹ️ Модель попыталась завершить задачу, но чеклист ещё не закрыт. Продолжаю."})
						a.conversationHist = append(a.conversationHist, ChatMessage{Role: "user", Content: "Ты попытался завершить задачу, но чеклист целей ещё не закрыт. Продолжай выполнять текущий пункт."})
						continue
					}
				}
			}
			a.broadcast(Message{Type: "complete", Content: fmt.Sprintf("✅ Задача выполнена!\n\n%s", response.FinalReport)})
			return
		}

		// Проверяем необходимость подтверждения
		if response.NeedsConfirm {
			ok := true
			if a.requestConfirm != nil {
				ok = a.requestConfirm(response.Action.Action, response.ConfirmMsg)
			}
			if !ok {
				a.broadcast(Message{Type: "status", Content: "❌ Действие не подтверждено пользователем. Останавливаюсь."})
				return
			}
		}

		// Выполняем действие
		a.sanitizeActionForChecklist(&response.Action)
		a.broadcast(Message{Type: "action", Content: fmt.Sprintf("🎯 %s: %s", response.Action.Action, response.Action.Reason)})

		result, err := a.executeAction(response.Action)
		if err != nil {
			// Safe auto-recovery: if a click failed (often due to category navigation), try searching in-page.
			if recMsg, ok := a.tryAutoRecovery(response.Action, err); ok {
				a.broadcast(Message{Type: "status", Content: fmt.Sprintf("🛠️ Автовосстановление: %s", recMsg)})
				a.conversationHist = append(a.conversationHist, ChatMessage{
					Role:    "user",
					Content: fmt.Sprintf("Результат действия: ОШИБКА при %s (%v). Выполнено автовосстановление: %s.", response.Action.Action, err, recMsg),
				})
			} else {
				errMsg := fmt.Sprintf("Ошибка выполнения действия %s: %v", response.Action.Action, err)
				a.broadcast(Message{Type: "error", Content: fmt.Sprintf("⚠️ %s", errMsg)})
				a.conversationHist = append(a.conversationHist, ChatMessage{
					Role:    "user",
					Content: fmt.Sprintf("Результат действия: ОШИБКА - %s. Попробуй другой подход.", errMsg),
				})
			}
		} else {
			a.broadcast(Message{Type: "status", Content: fmt.Sprintf("✓ %s", result)})
			a.conversationHist = append(a.conversationHist, ChatMessage{
				Role:    "user",
				Content: fmt.Sprintf("Результат действия: %s", result),
			})
		}

		autoClosedGoal := false
		// If the current goal is "add to cart" and we detect that the cart changed since the goal baseline,
		// auto-close the goal even if the LLM forgot to set goal_completed.
		if len(a.goalChecklist) > 0 && a.isAddToCartGoal() && a.goalCartGuardTriggered() && a.currentGoalIdx < len(a.goalChecklist) {
			cur := a.goalChecklist[a.currentGoalIdx]
			a.goalDone[a.currentGoalIdx] = true
			a.broadcast(Message{Type: "status", Content: fmt.Sprintf("✅ Чеклист: пункт %d/%d выполнен автоматически (корзина изменилась) — %s", a.currentGoalIdx+1, len(a.goalChecklist), cur)})
			a.conversationHist = append(a.conversationHist, ChatMessage{Role: "user", Content: fmt.Sprintf("Корзина изменилась по сравнению с началом пункта. Считай пункт выполненным: %s. НЕ добавляй ничего лишнего.", cur)})
			a.goalCartArmed = false
			a.goalCartGoalIdx = -1
			a.goalCartBaseline = ""
			a.currentGoalIdx++
			if a.currentGoalIdx >= len(a.goalChecklist) {
				a.broadcast(Message{Type: "status", Content: "🧾 Все пункты чеклиста закрыты. Можно завершать задачу."})
				a.conversationHist = append(a.conversationHist, ChatMessage{Role: "user", Content: "Все пункты чеклиста закрыты. Теперь можно завершить задачу, сформировав финальный отчёт."})
			}
			autoClosedGoal = true
		}

		// Если модель сообщает, что текущий пункт чеклиста выполнен — фиксируем и переходим к следующему.
		if !autoClosedGoal && response.GoalCompleted && len(a.goalChecklist) > 0 && a.currentGoalIdx < len(a.goalChecklist) {
			cur := a.goalChecklist[a.currentGoalIdx]
			a.goalDone[a.currentGoalIdx] = true
			a.broadcast(Message{Type: "status", Content: fmt.Sprintf("✅ Чеклист: выполнен пункт %d/%d — %s", a.currentGoalIdx+1, len(a.goalChecklist), cur)})
			// Подскажем LLM, что можно перейти к следующей цели.
			a.conversationHist = append(a.conversationHist, ChatMessage{Role: "user", Content: fmt.Sprintf("Отмечаем пункт чеклиста как выполненный: %s. Переходи к следующему пункту.", cur)})
			a.currentGoalIdx++
			if a.currentGoalIdx >= len(a.goalChecklist) {
				a.broadcast(Message{Type: "status", Content: "🧾 Все пункты чеклиста закрыты. Можно завершать задачу."})
				a.conversationHist = append(a.conversationHist, ChatMessage{Role: "user", Content: "Все пункты чеклиста закрыты. Теперь можно завершить задачу, сформировав финальный отчёт."})
			}
		}

		// Пауза между действиями
		time.Sleep(1500 * time.Millisecond)
	}

	a.broadcast(Message{Type: "status", Content: "⚠️ Достигнут лимит шагов"})
}

// callLLM выполняет единичный запрос к LLM и возвращает content ответа.
func (a *Agent) callLLM(messages []ChatMessage, maxTokens int, temperature float64) (string, error) {
	apiURL := os.Getenv("AI_API_URL")
	model := os.Getenv("AI_MODEL")
	if apiURL == "" {
		apiURL = "https://openrouter.ai/api/v1/chat/completions"
	}
	if model == "" {
		model = "deepseek/deepseek-chat"
	}

	reqBody := OpenRouterRequest{
		Model:       model,
		Messages:    messages,
		Temperature: temperature,
		MaxTokens:   maxTokens,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", a.apiKey))
	req.Header.Set("HTTP-Referer", "http://localhost:8080")
	req.Header.Set("X-Title", "AI Browser Agent")

	client := &http.Client{Timeout: 90 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	var aiResp AIResponse
	if err := json.Unmarshal(body, &aiResp); err != nil {
		return "", fmt.Errorf("ошибка парсинга ответа: %v, body: %s", err, string(body))
	}
	if aiResp.Error != nil {
		return "", fmt.Errorf("API ошибка: %s", aiResp.Error.Message)
	}
	if len(aiResp.Choices) == 0 {
		return "", fmt.Errorf("пустой ответ от API, body: %s", string(body))
	}
	return aiResp.Choices[0].Message.Content, nil
}

// decomposeTaskGoals извлекает атомарные цели/объекты из пользовательской задачи.
// Ключевой кейс: «маффин с яйцом и ветчиной и морковные палочки» -> 2 пункта, но ингредиенты не дробятся.
func (a *Agent) decomposeTaskGoals(task string) ([]string, error) {
	if a.apiKey == "" {
		return nil, fmt.Errorf("API ключ не установлен")
	}

	msgs := []ChatMessage{
		{Role: "system", Content: `Ты извлекаешь из запроса пользователя список АТОМАРНЫХ целей/объектов.

Верни СТРОГО валидный JSON без markdown:
{"goals":["..."],"notes":"..."}

Правила:
- Если пользователь перечисляет несколько объектов через запятые/«и»/списком — сделай их ОТДЕЛЬНЫМИ пунктами.
- НЕ дели ингредиенты внутри одной позиции (например «маффин с яйцом и ветчиной» — одна цель).
- Для заказа еды/товаров: goals должны быть ТОЛЬКО названиями позиций так, как их можно ввести в поиск/меню. НЕ добавляй глаголы и контекст вроде «закажи», «добавь в корзину», «на Яндекс Еде», «на сайте», «оформи».
  Пример: «закажи маффин с яйцом и ветчиной и морковные палочки» -> goals=["маффин с яйцом и ветчиной","морковные палочки"].
- Если задача не про список объектов, всё равно декомпозируй на несколько проверяемых целей (например «прочитать 10 писем» и «удалить спам»).
- Если совсем не уверен — верни goals из одного пункта: исходный запрос.`},
		{Role: "user", Content: task},
	}

	content, err := a.callLLM(msgs, 400, 0.0)
	if err != nil {
		return nil, err
	}
	content = strings.TrimSpace(content)
	content = strings.TrimPrefix(content, "```json")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(content, "```")
	content = strings.TrimSpace(content)

	var dec TaskDecomposition
	if err := json.Unmarshal([]byte(content), &dec); err != nil {
		// Фолбэк: один пункт
		return []string{task}, nil
	}
	// Чистим и убираем пустые
	out := make([]string, 0, len(dec.Goals))
	seen := map[string]bool{}
	for _, g := range dec.Goals {
		g = strings.TrimSpace(g)
		if g == "" {
			continue
		}
		lg := strings.ToLower(g)
		if seen[lg] {
			continue
		}
		seen[lg] = true
		out = append(out, g)
	}
	if len(out) == 0 {
		out = []string{task}
	}
	return out, nil
}

func (a *Agent) getSystemPrompt() string {
	return `Ты — автономный AI-агент, управляющий веб-браузером для выполнения задач пользователя.

ВАЖНЫЕ ПРАВИЛА:
1. Анализируй текущее состояние страницы и принимай решения самостоятельно
2. НЕ используй заготовленные селекторы — определяй их из DOM
3. Действуй пошагово, проверяя результат каждого действия
4. При ошибках пробуй альтернативные подходы
5. Для ОПАСНЫХ действий (удаление, оплата, отправка) устанавливай needs_confirm: true

ПРАКТИЧЕСКИЙ СОВЕТ ДЛЯ МЕНЮ/КАТАЛОГОВ (еда, магазины):
- Если ты находишься внутри страницы ресторана/магазина, ПРЕДПОЧИТАЙ поиск по меню через поисковую строку (input/поле поиска) вместо навигации по категориям.
- Категории часто требуют скролла/ленивой загрузки и клики могут срываться; поиск по меню обычно надёжнее.

КРИТИЧЕСКИ ВАЖНО ДЛЯ "ДОБАВИТЬ В КОРЗИНУ":
- Когда нужно добавить блюдо/товар, НЕ кликай по карточке товара, если рядом есть отдельная кнопка (например "В корзину", "Добавить", "+", "Order/Add").
- Ищи именно управляющую кнопку добавления и кликай по ней; после клика проверь, что корзина/счётчик действительно изменился.
 - Если пользователь пишет "любой ..." / "any ..." (например "любой воппер"), нужно добавить РОВНО ОДИН подходящий товар и сразу прекратить добавления.
 - Если ты не уверен, добавился ли товар, сначала проверь корзину/счётчик и только потом пробуй другое действие. Не спамь повторными кликами.

КРИТИЧЕСКИ ВАЖНО ПРО МНОЖЕСТВЕННЫЕ ПОЗИЦИИ:
- Если пользователь перечисляет несколько объектов (например несколько блюд, несколько писем, несколько вакансий), считай это ОТДЕЛЬНЫМИ пунктами.
- НЕ объединяй разные позиции в одну. Например: «маффин с яйцом и ветчиной и морковные палочки» — это 2 позиции: (1) маффин с яйцом и ветчиной, (2) морковные палочки.
- При этом НЕ дели ингредиенты внутри названия одной позиции (например «с яйцом и ветчиной» — часть одного блюда).
- Перед завершением задачи мысленно пройди чеклист целей и убедись, что все пункты закрыты.

ПРАВИЛО СТРОГОЙ ПОСЛЕДОВАТЕЛЬНОСТИ ДЛЯ СПИСКОВ (например заказ еды из 2+ позиций):
- Если в контексте есть сообщение "ТЕКУЩАЯ ЦЕЛЬ ЧЕКЛИСТА" — работай ТОЛЬКО над этой целью.
- Сначала найди и добавь в корзину текущую цель. Только после этого переходи к следующей.
- Когда текущая цель выполнена, установи goal_completed:true и completed_goal:"<текст текущей цели>".

ДОСТУПНЫЕ ДЕЙСТВИЯ:
- navigate: перейти по URL {"action": "navigate", "url": "https://..."}
- click: кликнуть по элементу {"action": "click", "selector": "селектор"}
- type: ввести текст {"action": "type", "selector": "селектор", "value": "текст"}

ВАЖНО про действие type: поле value — это ТОЛЬКО то, что должно оказаться в поле ввода (поисковый запрос, адрес, имя и т.п.). Никогда не пиши туда инструкции вроде «добавь в корзину ...», «на сайте ...», «в яндекс еде ...».
- scroll: прокрутить {"action": "scroll", "value": "down|up|top|bottom"}
- wait: подождать {"action": "wait", "value": "2000"} (миллисекунды)
- extract: извлечь данные {"action": "extract", "selector": "селектор"}
- screenshot: сделать скриншот для анализа

ВАЖНО ПРО СЕЛЕКТОРЫ:
- Предпочитай CSS селекторы (querySelector): #id, input[name="..."], button[aria-label="..."], [data-qa="..."]
- Псевдоселектор :contains("текст") НЕ является стандартным CSS — вместо него используй XPath.
- XPath поддерживается, если selector начинается с // (например: //button[contains(., "Купить")])
- Если элемент сложно найти, выбирай селектор по стабильным атрибутам (aria-label, name, placeholder, data-testid и т.п.)

ФОРМАТ ОТВЕТА (строго JSON без markdown):
{
  "thinking": "Анализ ситуации и план действий",
  "action": {
    "action": "тип_действия",
    "selector": "селектор (если нужен)",
    "value": "значение (если нужно)",
    "url": "URL (для navigate)",
    "reason": "почему выбрано это действие"
  },
  "is_complete": false,
  "needs_confirm": false,
  "confirm_message": "сообщение для подтверждения",
  "goal_completed": false,
  "completed_goal": ""
}

Когда задача ПОЛНОСТЬЮ выполнена:
{
  "thinking": "Задача выполнена",
  "action": {"action": "none", "reason": "завершено"},
  "is_complete": true,
  "final_report": "Подробный отчёт о выполненной работе"
}

ОТВЕЧАЙ ТОЛЬКО ВАЛИДНЫМ JSON БЕЗ MARKDOWN-РАЗМЕТКИ!`
}

func (a *Agent) getPageState() (string, error) {
	a.browserMu.Lock()
	ctx := a.browserCtx
	a.browserMu.Unlock()

	if ctx == nil {
		return "", fmt.Errorf("браузер не запущен")
	}

	// Создаём контекст с таймаутом для операции.
	// На тяжёлых SPA (еда/почта/HH) 15s часто недостаточно и приводит к context deadline exceeded.
	opCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	var url, title string

	err := chromedp.Run(opCtx,
		chromedp.Location(&url),
		chromedp.Title(&title),
	)
	if err != nil {
		return "", fmt.Errorf("ошибка получения URL/title: %v", err)
	}

	// Получаем компактное описание страницы (видимый текст + интерактивные элементы).
	pageOverview, err := a.extractPageOverview(opCtx)
	if err != nil {
		pageOverview = fmt.Sprintf("Ошибка извлечения структуры: %v", err)
	}

	state := fmt.Sprintf(`=== СОСТОЯНИЕ СТРАНИЦЫ ===
URL: %s
Заголовок: %s

=== ОБЗОР СТРАНИЦЫ (компактно) ===
%s`, url, title, pageOverview)

	return state, nil
}

// getCurrentURL возвращает текущий URL активной вкладки.
// Нужен, чтобы принимать решения (например, автозакрывать простой навигационный пункт чеклиста)
// без парсинга строкового pageState.
func (a *Agent) getCurrentURL() (string, error) {
	a.browserMu.Lock()
	ctx := a.browserCtx
	a.browserMu.Unlock()

	if ctx == nil {
		return "", fmt.Errorf("браузер не запущен")
	}

	opCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	var url string
	if err := chromedp.Run(opCtx, chromedp.Location(&url)); err != nil {
		return "", err
	}
	return url, nil
}

// extractPageOverview возвращает компактное, пригодное для LLM описание страницы:
// 1) фрагмент видимого текста
// 2) список интерактивных элементов с разумными селекторами
func (a *Agent) extractPageOverview(ctx context.Context) (string, error) {
	var visibleText string
	// Видимый текст (как контекст, без полного DOM)
	err := chromedp.Run(ctx,
		chromedp.Evaluate(`(function(){
			const t = (document.body && document.body.innerText) ? document.body.innerText : '';
			return t.replace(/\s+/g,' ').trim().slice(0, 3000);
		})()`, &visibleText),
	)
	if err != nil {
		visibleText = ""
	}

	// Интерактивные элементы
	var elementsJSON string
	err = chromedp.Run(ctx,
		chromedp.Evaluate(`(function(){
			function escCssIdent(s){
				// минимальный экранировщик, чтобы не падать без CSS.escape
				return String(s).replace(/[^a-zA-Z0-9_\-]/g, function(ch){
					return '\\' + ch;
				});
			}
			function q(v){
				return String(v).replace(/\\/g,'\\\\').replace(/"/g,'\\"');
			}
			function cssPath(el){
				if (!el || !el.tagName) return '';
				if (el.id) return '#' + escCssIdent(el.id);
				const tag = el.tagName.toLowerCase();
				const attrs = ['data-testid','data-test','data-qa','data-automation','data-qaid','name','aria-label','placeholder','role','type'];
				for (const a of attrs){
					const v = el.getAttribute && el.getAttribute(a);
					if (v && v.length <= 80) return tag + '[' + a + '="' + q(v) + '"]';
				}
				let sel = tag;
				if (typeof el.className === 'string'){
					const cls = el.className.trim().split(/\s+/).filter(Boolean).slice(0,2).map(c=>'.'+escCssIdent(c)).join('');
					if (cls) sel += cls;
				}
				// добавим nth-of-type, чтобы селектор был стабильнее
				const p = el.parentElement;
				if (p){
					const sib = Array.from(p.children).filter(ch => ch.tagName === el.tagName);
					if (sib.length > 1){
						const idx = sib.indexOf(el) + 1;
						sel += ':nth-of-type(' + idx + ')';
					}
				}
				return sel;
			}
			const max = 120;
			const all = Array.from(document.querySelectorAll('input, textarea, button, a[href], select, [role="button"], [contenteditable="true"], [tabindex]'));
			const els = [];
			for (const el of all){
				try {
					const r = el.getBoundingClientRect();
					if (!r || r.width < 2 || r.height < 2) continue;
					// берём и видимые, и близкие к видимой области (на случай скролла)
					if (r.bottom < -200 || r.top > (window.innerHeight + 1200)) continue;
					els.push(el);
					if (els.length >= max) break;
				} catch(e){}
			}
			const out = els.map(el => {
				const tag = el.tagName.toLowerCase();
				const text = ((el.innerText || el.value || '') + '').replace(/\s+/g,' ').trim().slice(0,80);
				return {
					tag,
					text,
					aria: (el.getAttribute('aria-label') || '').slice(0,80),
					placeholder: (el.getAttribute('placeholder') || '').slice(0,80),
					name: (el.getAttribute('name') || '').slice(0,80),
					type: (el.getAttribute('type') || '').slice(0,30),
					href: (el.getAttribute('href') || '').slice(0,120),
					selector: cssPath(el)
				};
			});
			return JSON.stringify(out);
		})()`, &elementsJSON),
	)
	if err != nil {
		return "", err
	}

	// Форматируем список элементов как читаемые строки
	type elInfo struct {
		Tag         string `json:"tag"`
		Text        string `json:"text"`
		Aria        string `json:"aria"`
		Placeholder string `json:"placeholder"`
		Name        string `json:"name"`
		Type        string `json:"type"`
		Href        string `json:"href"`
		Selector    string `json:"selector"`
	}
	var infos []elInfo
	if strings.TrimSpace(elementsJSON) != "" {
		_ = json.Unmarshal([]byte(elementsJSON), &infos)
	}

	var b strings.Builder
	if visibleText != "" {
		b.WriteString("Видимый текст (фрагмент):\n")
		b.WriteString(visibleText)
		b.WriteString("\n\n")
	}
	b.WriteString("Интерактивные элементы (кандидаты):\n")
	limit := 80
	if len(infos) < limit {
		limit = len(infos)
	}
	for i := 0; i < limit; i++ {
		it := infos[i]
		label := it.Text
		if label == "" {
			if it.Aria != "" {
				label = it.Aria
			} else if it.Placeholder != "" {
				label = "placeholder: " + it.Placeholder
			}
		}
		if label == "" {
			label = "(no text)"
		}
		b.WriteString(fmt.Sprintf("%d) <%s> %s | selector: %s\n", i+1, it.Tag, label, it.Selector))
		if it.Href != "" {
			b.WriteString(fmt.Sprintf("   href: %s\n", it.Href))
		}
		if it.Name != "" || it.Type != "" {
			b.WriteString(fmt.Sprintf("   name: %s type: %s\n", it.Name, it.Type))
		}
	}
	if len(infos) > limit {
		b.WriteString(fmt.Sprintf("... и ещё %d элементов\n", len(infos)-limit))
	}

	// Ограничиваем общий объём на всякий случай
	out := b.String()
	if len(out) > 20000 {
		out = out[:20000] + "\n... [обзор обрезан]"
	}
	return out, nil
}

func (a *Agent) extractSimplifiedDOM(ctx context.Context) (string, error) {
	var html string

	err := chromedp.Run(ctx,
		chromedp.ActionFunc(func(ctx context.Context) error {
			node, err := dom.GetDocument().Do(ctx)
			if err != nil {
				return err
			}
			html, err = dom.GetOuterHTML().WithNodeID(node.NodeID).Do(ctx)
			return err
		}),
	)

	if err != nil {
		return "", err
	}

	return a.simplifyHTML(html), nil
}

func (a *Agent) simplifyHTML(html string) string {
	// Удаляем скрипты
	scriptRe := regexp.MustCompile(`(?is)<script[^>]*>.*?</script>`)
	html = scriptRe.ReplaceAllString(html, "")

	// Удаляем стили
	styleRe := regexp.MustCompile(`(?is)<style[^>]*>.*?</style>`)
	html = styleRe.ReplaceAllString(html, "")

	// Удаляем SVG
	svgRe := regexp.MustCompile(`(?is)<svg[^>]*>.*?</svg>`)
	html = svgRe.ReplaceAllString(html, "[SVG]")

	// Удаляем комментарии
	commentRe := regexp.MustCompile(`<!--[\s\S]*?-->`)
	html = commentRe.ReplaceAllString(html, "")

	// Удаляем noscript
	noscriptRe := regexp.MustCompile(`(?is)<noscript[^>]*>.*?</noscript>`)
	html = noscriptRe.ReplaceAllString(html, "")

	// Упрощаем атрибуты style (удаляем их)
	styleAttrRe := regexp.MustCompile(`\s+style="[^"]*"`)
	html = styleAttrRe.ReplaceAllString(html, "")

	// Удаляем data-reactid и подобные
	reactRe := regexp.MustCompile(`\s+data-react[^=]*="[^"]*"`)
	html = reactRe.ReplaceAllString(html, "")

	// Удаляем лишние пробелы и переводы строк
	spaceRe := regexp.MustCompile(`\s+`)
	html = spaceRe.ReplaceAllString(html, " ")

	// Делаем более читаемым
	html = strings.ReplaceAll(html, "> <", ">\n<")

	// Ограничиваем размер
	if len(html) > 20000 {
		html = html[:20000] + "\n... [DOM обрезан]"
	}

	return html
}

func (a *Agent) askAI(pageState string) (*AgentResponse, error) {
	if a.apiKey == "" {
		return nil, fmt.Errorf("API ключ не установлен")
	}

	// Добавляем состояние страницы в контекст
	messages := make([]ChatMessage, len(a.conversationHist))
	copy(messages, a.conversationHist)

	// Если есть чеклист (несколько целей), каждый шаг явно задаём текущую цель.
	if len(a.goalChecklist) > 0 && a.currentGoalIdx < len(a.goalChecklist) {
		cur := a.goalChecklist[a.currentGoalIdx]
		// Сводка по прогрессу (коротко)
		done := 0
		for _, v := range a.goalDone {
			if v {
				done++
			}
		}
		messages = append(messages, ChatMessage{Role: "user", Content: fmt.Sprintf(
			"ТЕКУЩАЯ ЦЕЛЬ ЧЕКЛИСТА (выполняй только её): #%d/%d: %s\nПрогресс: выполнено %d/%d.\nПравила: (1) Если вводишь текст в поиск/поле — вводи ТОЛЬКО текущую цель, не перечисляй несколько. (2) Добавь/выполни текущую цель полностью. (3) Когда цель выполнена — верни goal_completed:true и completed_goal с текстом текущего пункта.",
			a.currentGoalIdx+1, len(a.goalChecklist), cur, done, len(a.goalChecklist)),
		})
	}

	messages = append(messages, ChatMessage{
		Role:    "user",
		Content: fmt.Sprintf("Текущее состояние:\n%s\n\nЧто делаем дальше? Ответь JSON.", pageState),
	})

	// Ограничиваем историю
	if len(messages) > 20 {
		messages = append(messages[:1], messages[len(messages)-18:]...)
	}

	// Определяем API URL и модель
	apiURL := os.Getenv("AI_API_URL")
	model := os.Getenv("AI_MODEL")

	if apiURL == "" {
		// По умолчанию используем OpenRouter
		apiURL = "https://openrouter.ai/api/v1/chat/completions"
	}
	if model == "" {
		model = "deepseek/deepseek-chat"
	}

	reqBody := OpenRouterRequest{
		Model:       model,
		Messages:    messages,
		Temperature: 0.3,
		MaxTokens:   2000,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", a.apiKey))
	// Заголовки для OpenRouter
	req.Header.Set("HTTP-Referer", "http://localhost:8080")
	req.Header.Set("X-Title", "AI Browser Agent")

	client := &http.Client{Timeout: 90 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var aiResp AIResponse
	if err := json.Unmarshal(body, &aiResp); err != nil {
		return nil, fmt.Errorf("ошибка парсинга ответа: %v, body: %s", err, string(body))
	}

	if aiResp.Error != nil {
		return nil, fmt.Errorf("API ошибка: %s", aiResp.Error.Message)
	}

	if len(aiResp.Choices) == 0 {
		return nil, fmt.Errorf("пустой ответ от API, body: %s", string(body))
	}

	content := aiResp.Choices[0].Message.Content

	// Сохраняем ответ в историю
	a.conversationHist = append(a.conversationHist, ChatMessage{
		Role:    "assistant",
		Content: content,
	})

	// Парсим JSON из ответа
	content = strings.TrimSpace(content)
	content = strings.TrimPrefix(content, "```json")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(content, "```")
	content = strings.TrimSpace(content)

	var agentResp AgentResponse
	if err := json.Unmarshal([]byte(content), &agentResp); err != nil {
		return nil, fmt.Errorf("ошибка парсинга JSON агента: %v\nОтвет: %s", err, content)
	}

	// Автоматически требуем подтверждение для опасных действий
	if a.isDestructiveAction(agentResp.Action) && !agentResp.NeedsConfirm {
		agentResp.NeedsConfirm = true
		agentResp.ConfirmMsg = fmt.Sprintf("Подтвердите: %s - %s", agentResp.Action.Action, agentResp.Action.Reason)
	}

	return &agentResp, nil
}

func (a *Agent) isDestructiveAction(action BrowserAction) bool {
	// Keep this intentionally narrow. Words like "заказ"/"order" appear in benign navigation steps
	// (e.g. "перейти на страницу ... для заказа") and caused false confirmation prompts.
	destructiveKeywords := []string{"удал", "delete", "оплат", "pay", "купи", "buy", "отправ", "submit", "подтверд", "confirm"}

	reason := strings.ToLower(action.Reason)
	for _, kw := range destructiveKeywords {
		if strings.Contains(reason, kw) {
			return true
		}
	}

	return false
}

// sanitizeActionForChecklist makes multi-item tasks behave sequentially.
// If we have a goal checklist and the model tries to type multiple items at once,
// we force the typed value to be ONLY the current goal.

func cleanSearchQuery(s string) string {
	s = strings.TrimSpace(s)
	// Remove surrounding quotes
	s = strings.Trim(s, "\"'“”«»")
	lower := strings.ToLower(s)
	// Common directive / boilerplate phrases that should never be typed into a search box
	badPhrases := []string{
		"добавить в корзину", "добавь в корзину", "добавьте в корзину",
		"заказать", "закажи", "закажите",
		"найти", "найди", "найдите",
		"введи", "введите",
		"на сайте", "в приложении",
		"на яндекс еде", "в яндекс еде", "яндекс еда", "яндекс.еда",
	}
	for _, bp := range badPhrases {
		if strings.Contains(lower, bp) {
			s = strings.ReplaceAll(s, bp, "")
			lower = strings.ToLower(s)
		}
	}
	// Strip trailing context like "на <service>" if it looks like a service mention
	for _, tail := range []string{" на ", " в "} {
		idx := strings.LastIndex(strings.ToLower(s), tail)
		if idx != -1 {
			// If the tail part contains a dot or the word "еда"/"hh"/"почта"/"доставка", it's likely context.
			tailPart := strings.ToLower(strings.TrimSpace(s[idx+len(tail):]))
			if anySubstr(tailPart, []string{".", "еда", "hh", "почт", "достав", "пицц", "mail", "food"}) {
				s = strings.TrimSpace(s[:idx])
				break
			}
		}
	}
	// Collapse whitespace
	s = strings.Join(strings.Fields(s), " ")
	// If still too long, keep first 8 tokens (search queries shouldn't be essays)
	parts := strings.Fields(s)
	if len(parts) > 8 {
		s = strings.Join(parts[:8], " ")
	}
	return strings.TrimSpace(s)
}

// jsEscapeForSingleQuotes escapes a string so it can be safely embedded into a JavaScript string literal wrapped in single quotes.
func jsEscapeForSingleQuotes(s string) string {
	// Escape backslashes first.
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "'", "\\'")
	s = strings.ReplaceAll(s, "\n", "\\n")
	s = strings.ReplaceAll(s, "\r", "")
	return s
}

func anySubstr(s string, subs []string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

// isCartishURL returns true if URL likely corresponds to cart mutations.
// We keep this heuristic conservative to avoid false "item added" signals.
func isCartishURL(u string) bool {
	l := strings.ToLower(u)
	if strings.Contains(l, "cart") || strings.Contains(l, "basket") || strings.Contains(l, "checkout") {
		return true
	}
	if strings.Contains(l, "order") {
		if strings.Contains(l, "item") || strings.Contains(l, "items") || strings.Contains(l, "position") || strings.Contains(l, "positions") || strings.Contains(l, "line") || strings.Contains(l, "basket") || strings.Contains(l, "cart") {
			return true
		}
	}
	return false
}

// allowComboForQuery returns true if the user goal/query explicitly asks for a combo/meal.
func (a *Agent) allowComboForQuery(q string) bool {
	l := strings.ToLower(q)
	combo := []string{"комбо", "combo", "meal", "набор", "сет", "set"}
	return anySubstr(l, combo)
}

func (a *Agent) isProbablySearchField(selector string) bool {
	selector = strings.TrimSpace(selector)
	if selector == "" {
		return false
	}
	// Use JS to inspect element attributes. Support both CSS and XPath selectors.
	js := ""
	if strings.HasPrefix(selector, "//") {
		js = fmt.Sprintf(`(function(){
			try{
				const xp = %q;
				const r = document.evaluate(xp, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
				const el = r.singleNodeValue;
				if(!el) return false;
				const tag = (el.tagName||'').toLowerCase();
				if(tag!=='input' && tag!=='textarea') return false;
				const attrs = [el.getAttribute('placeholder'), el.getAttribute('aria-label'), el.getAttribute('name'), el.getAttribute('id'), el.getAttribute('role'), el.getAttribute('type')].join(' ').toLowerCase();
				return attrs.includes('search') || attrs.includes('иск') || attrs.includes('поиск') || attrs.includes('найд');
			}catch(e){return false;}
		})()`, selector)
	} else {
		js = fmt.Sprintf(`(function(){
			try{
				const el = document.querySelector(%q);
				if(!el) return false;
				const tag = (el.tagName||'').toLowerCase();
				if(tag!=='input' && tag!=='textarea') return false;
				const attrs = [el.getAttribute('placeholder'), el.getAttribute('aria-label'), el.getAttribute('name'), el.getAttribute('id'), el.getAttribute('role'), el.getAttribute('type')].join(' ').toLowerCase();
				return attrs.includes('search') || attrs.includes('иск') || attrs.includes('поиск') || attrs.includes('найд');
			}catch(e){return false;}
		})()`, selector)
	}
	var ok bool
	a.browserMu.Lock()
	ctx := a.browserCtx
	a.browserMu.Unlock()
	if ctx == nil {
		return false
	}
	opCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	_ = chromedp.Run(opCtx, chromedp.Evaluate(js, &ok))
	return ok
}

func (a *Agent) sanitizeActionForChecklist(action *BrowserAction) {
	if action == nil {
		return
	}
	if action.Action != "type" {
		return
	}
	// If we have a checklist, enforce sequential entry.
	hasChecklist := len(a.goalChecklist) > 0 && a.currentGoalIdx < len(a.goalChecklist)
	cur := ""
	if hasChecklist {
		cur = strings.TrimSpace(a.goalChecklist[a.currentGoalIdx])
	}

	val := strings.TrimSpace(action.Value)
	if val == "" {
		// If model forgot the value but we have a checklist goal, use it.
		if hasChecklist && cur != "" {
			val = cur
		}
	}
	if val == "" {
		return
	}

	lowerVal := strings.ToLower(val)
	lowerCur := strings.ToLower(cur)

	// Detect "instruction-like" values that should never be typed into a search box.
	looksInstruction := anySubstr(lowerVal, []string{"добав", "корзин", "закаж", "найд", "введ", "на сайте", "в приложении", "http", "яндекс"})
	looksLikeList := strings.Contains(lowerVal, " и ") || strings.Contains(val, ",") || strings.Contains(val, ";")

	// If we're typing into a search-like field, always type a CLEAN query (not a full instruction).
	if a.isProbablySearchField(action.Selector) {
		q := val
		if hasChecklist && cur != "" {
			q = cur
		}
		q = cleanSearchQuery(q)
		if q != "" {
			action.Value = q
			if action.Reason == "" {
				action.Reason = "auto-sanitized: ввожу только поисковый запрос"
			} else {
				action.Reason += " (auto-sanitized: ввожу только поисковый запрос)"
			}
		}
		return
	}

	// Otherwise: if we have a checklist and model merged multiple goals or typed instruction-like garbage — clamp to current goal.
	if hasChecklist && cur != "" {
		merged := false
		if looksLikeList {
			merged = !strings.Contains(lowerVal, lowerCur) || a.containsOtherGoal(lowerVal)
		}
		if looksInstruction {
			merged = true
		}
		if merged {
			clean := cleanSearchQuery(cur)
			if clean == "" {
				clean = cur
			}
			action.Value = clean
			if action.Reason == "" {
				action.Reason = "auto-sanitized: ввожу только текущую цель чеклиста"
			} else {
				action.Reason += " (auto-sanitized: ввожу только текущую цель чеклиста)"
			}
		}
	}
}

func (a *Agent) containsOtherGoal(lowerText string) bool {
	if len(a.goalChecklist) <= 1 {
		return false
	}
	stop := map[string]bool{"и": true, "с": true, "в": true, "на": true, "по": true, "из": true, "the": true, "with": true, "and": true, "a": true, "an": true}
	for i, g := range a.goalChecklist {
		if i == a.currentGoalIdx {
			continue
		}
		g = strings.ToLower(strings.TrimSpace(g))
		if g == "" {
			continue
		}
		// If the whole other-goal string appears, it's definitely a merge.
		if strings.Contains(lowerText, g) {
			return true
		}
		// Otherwise check a few informative keywords.
		words := strings.FieldsFunc(g, func(r rune) bool {
			return r == ' ' || r == ',' || r == '.' || r == ':' || r == ';' || r == '(' || r == ')' || r == '/' || r == '\\'
		})
		picked := 0
		for _, w := range words {
			w = strings.TrimSpace(w)
			if len([]rune(w)) < 4 {
				continue
			}
			if stop[w] {
				continue
			}
			if strings.Contains(lowerText, w) {
				return true
			}
			picked++
			if picked >= 3 {
				break
			}
		}
	}
	return false
}

// getCurrentSearchQuery returns the most relevant short query for in-page search.
// Prefer the current checklist goal; otherwise fallback to the original task text.
func (a *Agent) getCurrentSearchQuery() string {
	if len(a.goalChecklist) > 0 && a.currentGoalIdx < len(a.goalChecklist) {
		q := strings.TrimSpace(a.goalChecklist[a.currentGoalIdx])
		if q != "" {
			return q
		}
	}
	// conversationHist[1] is usually: "Задача пользователя: ..."
	if len(a.conversationHist) >= 2 {
		s := strings.TrimSpace(a.conversationHist[1].Content)
		pref := "Задача пользователя:"
		if strings.HasPrefix(s, pref) {
			s = strings.TrimSpace(strings.TrimPrefix(s, pref))
		}
		return s
	}
	return ""
}

// currentGoalText returns the active checklist goal text (if any).
func (a *Agent) currentGoalText() string {
	if len(a.goalChecklist) > 0 && a.currentGoalIdx >= 0 && a.currentGoalIdx < len(a.goalChecklist) {
		return strings.TrimSpace(a.goalChecklist[a.currentGoalIdx])
	}
	return ""
}

// isAddToCartGoal heuristically detects goals that intend to add exactly one item to a cart.
// This is NOT site-specific: it only looks at the goal text.
func (a *Agent) isAddToCartGoal() bool {
	g := strings.ToLower(a.currentGoalText())
	if g == "" {
		return false
	}
	// Be strict: treat only explicit "add to cart" intentions as add-to-cart goals.
	// Words like "оформить"/"order" are often checkout/navigation and should NOT arm the add-to-cart guard.
	keys := []string{"в корз", "корзин", "добав", "add to cart", "basket", "cart"}
	for _, k := range keys {
		if strings.Contains(g, k) {
			return true
		}
	}
	return false
}

func (a *Agent) isNavigationGoal() bool {
	g := strings.ToLower(a.currentGoalText())
	if g == "" {
		return false
	}
	// Do not treat explicit add-to-cart goals as pure navigation goals.
	if a.isAddToCartGoal() {
		return false
	}
	keys := []string{"открой", "откры", "перейд", "перейти", "зайд", "зайди", "open", "go to", "navigate"}
	for _, k := range keys {
		if strings.Contains(g, k) {
			return true
		}
	}
	return false
}

// getCartSignature returns a lightweight cart badge signature if present, otherwise empty string.
// We use it only as a monotonic "something changed" signal to stop duplicate adds.
func (a *Agent) getCartSignature(ctx context.Context) (string, error) {
	js := `(function(){
		function txt(el){
			try { return (el.textContent||'') + ' ' + (el.getAttribute('aria-label')||'') + ' ' + (el.getAttribute('title')||''); } catch(e){ return ''; }
		}
		function vis(el){
			if (!el) return false;
			try {
				var st = getComputedStyle(el);
				if (!st || st.display==='none' || st.visibility==='hidden' || Number(st.opacity)===0) return false;
				var r = el.getBoundingClientRect();
				return r && r.width>2 && r.height>2;
			} catch(e){ return false; }
		}
		var kw = /(корзин|cart|basket)/i;
		var bestN = -1;
		var bestS = '';
		var els = Array.from(document.querySelectorAll('a,button,[role="button"],div[role="button"],span[role="button"]'));
		for (var i=0;i<els.length;i++){
			var el = els[i];
			if (!vis(el)) continue;
			var b = txt(el);
			if (!kw.test(b)) continue;
			var around = b + ' ' + (el.parentElement ? (el.parentElement.textContent||'') : '');
			var m = around.match(/\d{1,3}/g);
			if (!m) continue;
			var n = parseInt(m[m.length-1], 10);
			if (!isFinite(n)) continue;
			if (n > bestN) { bestN = n; bestS = String(n); }
		}
		return bestS;
	})()`
	var sig string
	err := chromedp.Run(ctx, chromedp.Evaluate(js, &sig))
	return strings.TrimSpace(sig), err
}

// armGoalCartGuard captures the cart signature at the beginning of an add-to-cart goal.
func (a *Agent) armGoalCartGuard() {
	if !a.isAddToCartGoal() {
		a.goalCartArmed = false
		a.goalCartGoalIdx = -1
		a.goalCartBaseline = ""
		a.goalCartBaseCtr = 0
		return
	}
	if a.goalCartArmed && a.goalCartGoalIdx == a.currentGoalIdx {
		return
	}
	// Capture both baselines:
	// 1) a monotonic cart-mutation counter (network/fetch hooks)
	// 2) a best-effort cart badge signature (UI), used only as fallback.
	ctr := a.cartEventCounter.Load()
	ctx, cancel := context.WithTimeout(a.browserCtx, 4*time.Second)
	defer cancel()
	baseline, _ := a.getCartSignature(ctx)
	a.goalCartArmed = true
	a.goalCartGoalIdx = a.currentGoalIdx
	a.goalCartBaseline = baseline
	a.goalCartBaseCtr = ctr
}

// goalCartGuardTriggered checks whether cart signature changed since baseline.
func (a *Agent) goalCartGuardTriggered() bool {
	if !a.goalCartArmed || a.goalCartGoalIdx != a.currentGoalIdx {
		return false
	}
	// Primary: cart mutation counter advanced (reliable for SPAs).
	if a.cartEventCounter.Load() > a.goalCartBaseCtr {
		return true
	}
	ctx, cancel := context.WithTimeout(a.browserCtx, 4*time.Second)
	defer cancel()
	cur, err := a.getCartSignature(ctx)
	if err != nil {
		return false
	}
	return cur != a.goalCartBaseline
}

// tryInPageSearch finds a search input on the current page (without hardcoded selectors)
// and types the query into it (including Enter). Optionally clicks a "search" opener button
// if the input is not visible yet.
func (a *Agent) tryInPageSearch(query string) (string, error) {
	a.browserMu.Lock()
	ctx := a.browserCtx
	a.browserMu.Unlock()
	if ctx == nil {
		return "", fmt.Errorf("браузер не запущен")
	}
	query = strings.TrimSpace(query)
	if query == "" {
		return "", fmt.Errorf("пустой поисковый запрос")
	}

	opCtx, cancel := context.WithTimeout(ctx, 25*time.Second)
	defer cancel()

	js := `(function(){
		function escCssIdent(s){
			return String(s).replace(/[^a-zA-Z0-9_\-]/g, function(ch){ return '\\' + ch; });
		}
		function q(v){
			return String(v).replace(/\\/g,'\\\\').replace(/"/g,'\\"');
		}
		function isVisible(el){
			if (!el) return false;
			var st = window.getComputedStyle(el);
			if (!st) return false;
			if (st.display === 'none' || st.visibility === 'hidden' || Number(st.opacity) === 0) return false;
			var r = el.getBoundingClientRect();
			if (!r) return false;
			return r.width > 8 && r.height > 8;
		}
		function kw(s){
			return /(поиск|search|найти|искать|filter|фильтр|по меню|в меню)/i.test(String(s||''));
		}
		function cssPath(el){
			if (!el || !el.tagName) return '';
			if (el.id) return '#' + escCssIdent(el.id);
			var tag = el.tagName.toLowerCase();
			var attrs = ['data-testid','data-test','data-qa','data-automation','name','aria-label','placeholder','role','type','inputmode'];
			for (var i=0;i<attrs.length;i++){
				var a = attrs[i];
				var v = el.getAttribute && el.getAttribute(a);
				if (v && v.length <= 80){
					var sel = tag + '[' + a + '="' + q(v) + '"]';
					try { if (document.querySelectorAll(sel).length === 1) return sel; } catch(e) {}
					// even if not unique, keep as a reasonable guess
					return sel;
				}
			}
			var sel2 = tag;
			if (typeof el.className === 'string'){
				var cls = el.className.trim().split(/\s+/).filter(Boolean).slice(0,2).map(function(c){return '.'+escCssIdent(c)}).join('');
				if (cls) sel2 += cls;
			}
			var p = el.parentElement;
			if (p){
				var sib = Array.from(p.children).filter(function(ch){ return ch.tagName === el.tagName; });
				if (sib.length > 1){
					var idx = sib.indexOf(el) + 1;
					sel2 += ':nth-of-type(' + idx + ')';
				}
			}
			return sel2;
		}
		function scoreInput(el){
			var s = 0;
			if (!isVisible(el)) return -1000;
			var tag = (el.tagName||'').toLowerCase();
			if (tag === 'input') s += 5;
			if (tag === 'textarea') s += 2;
			var type = (el.getAttribute('type')||'').toLowerCase();
			if (type === 'search') s += 7;
			if (type === 'text') s += 2;
			if ((el.getAttribute('role')||'').toLowerCase() === 'searchbox') s += 6;
			var blob = [el.getAttribute('placeholder'), el.getAttribute('aria-label'), el.getAttribute('name'), el.id, el.className].join(' ');
			if (kw(blob)) s += 8;
			if (el.disabled) s -= 30;
			if (el.readOnly) s -= 10;
			// Prefer inputs near top of viewport
			try { var r = el.getBoundingClientRect(); if (r && r.top >= -20 && r.top < 250) s += 2; } catch(e) {}
			return s;
		}
		function scoreOpen(el){
			if (!isVisible(el)) return -1000;
			var s = 0;
			var tag = (el.tagName||'').toLowerCase();
			if (tag === 'button') s += 4;
			var blob = [el.getAttribute('aria-label'), el.getAttribute('title'), el.textContent, el.className, el.id].join(' ');
			if (kw(blob)) s += 10;
			return s;
		}
		var bestIn = null, bestInScore = -999;
		var inputs = Array.from(document.querySelectorAll('input, textarea, [contenteditable="true"]'));
		for (var i=0;i<inputs.length;i++){
			var el = inputs[i];
			var sc = scoreInput(el);
			if (sc > bestInScore){ bestInScore = sc; bestIn = el; }
		}
		var bestOpen = null, bestOpenScore = -999;
		var clicks = Array.from(document.querySelectorAll('button, [role="button"], a[href], div[role="button"], span[role="button"]'));
		for (var j=0;j<clicks.length;j++){
			var el2 = clicks[j];
			var sc2 = scoreOpen(el2);
			if (sc2 > bestOpenScore){ bestOpenScore = sc2; bestOpen = el2; }
		}
		return {
			open: (bestOpenScore >= 10 && bestOpen) ? cssPath(bestOpen) : '',
			input: (bestInScore >= 8 && bestIn) ? cssPath(bestIn) : ''
		};
	})();`

	var tgt SearchTargets
	if err := chromedp.Run(opCtx, chromedp.Evaluate(js, &tgt)); err != nil {
		return "", fmt.Errorf("не удалось найти элементы поиска: %v", err)
	}

	// If input not found but there is an opener, click it and retry once.
	if strings.TrimSpace(tgt.Input) == "" && strings.TrimSpace(tgt.Open) != "" {
		_ = chromedp.Run(opCtx,
			chromedp.ScrollIntoView(tgt.Open, chromedp.ByQuery),
			chromedp.Click(tgt.Open, chromedp.ByQuery),
		)
		time.Sleep(600 * time.Millisecond)
		_ = chromedp.Run(opCtx, chromedp.Evaluate(js, &tgt))
	}

	inputSel := strings.TrimSpace(tgt.Input)
	if inputSel == "" {
		return "", fmt.Errorf("на странице не найдено поле поиска")
	}

	// Type query and press Enter. First clear the field via JS to avoid "ГамбургерГамбургер"-style concatenation.
	clearJS := fmt.Sprintf(`(function(){
		var q = '%s';
		var el = null;
		try { el = document.querySelector(q); } catch(e) {}
		if (!el) return false;
		try { el.scrollIntoView({behavior:'instant', block:'center'}); } catch(e) {}
		try { el.focus(); } catch(e) {}
		try {
			if (el.isContentEditable) {
				el.innerText = '';
				el.dispatchEvent(new Event('input', {bubbles:true}));
				return true;
			}
			if ('value' in el) {
				el.value = '';
				el.dispatchEvent(new Event('input', {bubbles:true}));
				el.dispatchEvent(new Event('change', {bubbles:true}));
				return true;
			}
		} catch(e) {}
		return false;
	})()`, jsEscapeForSingleQuotes(inputSel))
	err := chromedp.Run(opCtx,
		chromedp.WaitReady(inputSel, chromedp.ByQuery),
		chromedp.ScrollIntoView(inputSel, chromedp.ByQuery),
		chromedp.Click(inputSel, chromedp.ByQuery),
		chromedp.Focus(inputSel, chromedp.ByQuery),
		chromedp.Evaluate(clearJS, nil),
		chromedp.SendKeys(inputSel, query, chromedp.ByQuery),
		chromedp.SendKeys(inputSel, "\r", chromedp.ByQuery),
	)
	if err != nil {
		return "", fmt.Errorf("не удалось выполнить поиск в меню: %v", err)
	}
	return fmt.Sprintf("Выполнил поиск по меню через строку поиска: '%s'", query), nil
}

// tryAutoRecovery attempts a safe fallback when a click fails.
// Primary use-case: food delivery sites where category clicks time out; searching in the restaurant menu is more reliable.
func (a *Agent) tryAutoRecovery(failedAction BrowserAction, actionErr error) (string, bool) {
	if failedAction.Action != "click" {
		return "", false
	}
	// Only attempt on typical click timeouts / not-found.
	msg := strings.ToLower(fmt.Sprintf("%v", actionErr))
	if !(strings.Contains(msg, "deadline") || strings.Contains(msg, "timeout") || strings.Contains(msg, "не удалось кликнуть") || strings.Contains(msg, "not found")) {
		return "", false
	}
	q := a.getCurrentSearchQuery()
	if q == "" {
		return "", false
	}
	res, err := a.tryInPageSearch(q)
	if err != nil {
		return "", false
	}
	return res, true
}

func (a *Agent) executeAction(action BrowserAction) (string, error) {
	a.browserMu.Lock()
	ctx := a.browserCtx
	a.browserMu.Unlock()

	if ctx == nil {
		return "", fmt.Errorf("браузер не запущен")
	}

	// Общий таймаут на действие. Для тяжёлых сайтов 30s часто мало.
	opCtx, cancel := context.WithTimeout(ctx, 90*time.Second)
	defer cancel()

	// Помощники
	runWithTimeout := func(d time.Duration, tasks ...chromedp.Action) error {
		stepCtx, stepCancel := context.WithTimeout(opCtx, d)
		defer stepCancel()
		return chromedp.Run(stepCtx, tasks...)
	}

	// Нормализация селектора + выбор стратегии поиска.
	// Возвращает:
	//  - нормализованный selector
	//  - chromedp QueryOption
	//  - isCSSQuery=true, если изначально это CSS (тогда имеет смысл fallback на BySearch)
	//
	// Правила:
	//  - XPath ("//" или "xpath=") и явный "search:" -> BySearch
	//  - "xpath:" -> BySearch
	//  - button:contains("Text") -> XPath -> BySearch
	//  - иначе CSS -> ByQuery
	normalizeSelector := func(sel string) (string, chromedp.QueryOption, bool) {
		s := strings.TrimSpace(sel)
		if s == "" {
			return s, chromedp.ByQuery, false
		}
		lower := strings.ToLower(s)
		if strings.HasPrefix(lower, "search:") {
			return strings.TrimSpace(s[len("search:"):]), chromedp.BySearch, false
		}
		if strings.HasPrefix(lower, "xpath:") {
			return strings.TrimSpace(s[len("xpath:"):]), chromedp.BySearch, false
		}
		// простая поддержка button:contains("Text") -> XPath
		if strings.Contains(s, ":contains(") {
			re := regexp.MustCompile(`^\s*([a-zA-Z0-9_\-]+)\s*:contains\((?:"([^"]*)"|'([^']*)')\)\s*$`)
			m := re.FindStringSubmatch(s)
			if len(m) > 0 {
				tag := m[1]
				text := m[2]
				if text == "" {
					text = m[3]
				}
				x := strings.ReplaceAll(text, `"`, "\\\"")
				s = fmt.Sprintf(`//%s[contains(normalize-space(.), "%s")]`, tag, x)
				return s, chromedp.BySearch, false
			}
		}
		if strings.HasPrefix(s, "//") || strings.HasPrefix(lower, "xpath=") {
			if strings.HasPrefix(lower, "xpath=") {
				s = strings.TrimSpace(s[6:])
			}
			return s, chromedp.BySearch, false
		}
		return s, chromedp.ByQuery, true
	}

	switch action.Action {
	case "navigate":
		err := runWithTimeout(60*time.Second,
			chromedp.Navigate(action.URL),
			chromedp.WaitReady("body", chromedp.ByQuery),
		)
		if err != nil {
			return "", err
		}
		// Ждём загрузки страницы
		time.Sleep(2 * time.Second)
		return fmt.Sprintf("Перешёл на %s", action.URL), nil

	case "click":
		sel, opt, isCSSQuery := normalizeSelector(action.Selector)
		if sel == "" {
			return "", fmt.Errorf("пустой selector")
		}

		// Для действий типа "добавить в корзину" нам нужен пост-контроль: изменился ли индикатор корзины.
		// Это НЕ хардкод под конкретный сайт — это общий эвристический счётчик (badge/цифра возле корзины).
		addIntent := func() bool {
			r := strings.ToLower(action.Reason)
			// If the reason is about navigation (open/go to/find a restaurant/page), DO NOT treat it as add-to-cart.
			neg := []string{"перей", "откр", "зайд", "страниц", "ресторан", "restaurant", "найд", "список", "open", "go to", "navigate", "link"}
			for _, k := range neg {
				if strings.Contains(r, k) {
					return false
				}
			}
			// Positive add-to-cart intent signals. Note: we intentionally do NOT include "заказ"/"order" here,
			// because they are often used in navigation goals ("перейти на страницу ... для заказа").
			pos := []string{"в корз", "корзин", "добав", "add", "to cart", "basket", "cart", "plus", "плюс"}
			for _, k := range pos {
				if strings.Contains(r, k) {
					return true
				}
			}
			return false
		}()

		cartSigJS := `(function(){
			function txt(el){
				try { return (el.textContent||'') + ' ' + (el.getAttribute('aria-label')||'') + ' ' + (el.getAttribute('title')||''); } catch(e){ return ''; }
			}
			function vis(el){
				if (!el) return false;
				try {
					var st = getComputedStyle(el);
					if (!st || st.display==='none' || st.visibility==='hidden' || Number(st.opacity)===0) return false;
					var r = el.getBoundingClientRect();
					return r && r.width>2 && r.height>2;
				} catch(e){ return false; }
			}
			var kw = /(корзин|cart|basket)/i;
			var bestN = -1;
			var bestS = '';
			var els = Array.from(document.querySelectorAll('a,button,[role="button"],div[role="button"],span[role="button"]'));
			for (var i=0;i<els.length;i++){
				var el = els[i];
				if (!vis(el)) continue;
				var b = txt(el);
				if (!kw.test(b)) continue;
				var around = b + ' ' + (el.parentElement ? (el.parentElement.textContent||'') : '');
				var m = around.match(/\d{1,3}/g);
				if (!m) continue;
				var n = parseInt(m[m.length-1], 10);
				if (!isFinite(n)) continue;
				if (n > bestN) { bestN = n; bestS = String(n); }
			}
			return bestS;
		})()`
		var preCartSig string
		_ = chromedp.Run(opCtx, chromedp.Evaluate(cartSigJS, &preCartSig))
		// Primary baseline for "item added" is cart-related network traffic (more reliable than DOM).
		preCartCtr := a.cartEventCounter.Load()
		var preJSCtr int
		_ = chromedp.Run(opCtx, chromedp.Evaluate(`(function(){try{return window.__CART_MUTATIONS||0}catch(e){return 0}})()`, &preJSCtr))

		waitForCartChange := func(timeout time.Duration) (string, bool) {
			deadline := time.Now().Add(timeout)
			var sig string
			for time.Now().Before(deadline) {
				// Network / fetch hooks
				if a.cartEventCounter.Load() > preCartCtr {
					return "network", true
				}
				var jsCtr int
				_ = chromedp.Run(opCtx, chromedp.Evaluate(`(function(){try{return window.__CART_MUTATIONS||0}catch(e){return 0}})()`, &jsCtr))
				if jsCtr > preJSCtr {
					return "js", true
				}
				_ = chromedp.Run(opCtx, chromedp.Evaluate(cartSigJS, &sig))
				sig = strings.TrimSpace(sig)
				if sig != preCartSig {
					return sig, true
				}
				time.Sleep(250 * time.Millisecond)
			}
			return strings.TrimSpace(sig), false
		}

		// If the user didn't ask for a combo/meal, avoid clicking upsell buttons in modals.
		query := strings.TrimSpace(a.getCurrentSearchQuery())
		allowCombo := a.allowComboForQuery(query)

		// Если после клика корзина не меняется, часто открывается модал с настройками/комбо.
		// Универсально пытаемся нажать кнопку подтверждения внутри модала (без хардкода селекторов под конкретный сайт).
		modalConfirmJS := fmt.Sprintf(`(function(){
			var allowCombo = %t;
			function vis(el){
				if(!el) return false;
				try{var st=getComputedStyle(el); if(!st||st.display==='none'||st.visibility==='hidden'||Number(st.opacity)===0) return false; var r=el.getBoundingClientRect(); return r&&r.width>6&&r.height>6;}catch(e){return false;}
			}
			function text(el){
				try{return (el.textContent||'')+' '+(el.getAttribute('aria-label')||'')+' '+(el.getAttribute('title')||'');}catch(e){return (el&&el.textContent)||'';}
			}
			function scoreBtn(b){
				var t=text(b).toLowerCase();
				var s=0;
				if(t.indexOf('в корз')>=0) s+=14;
				if(t.indexOf('добав')>=0) s+=12;
				if(t.indexOf('заказ')>=0) s+=9;
				if(t.indexOf('оформ')>=0) s+=7;
				if(t.indexOf('продолж')>=0) s+=6;
				if(t.indexOf('готов')>=0) s+=5;
				if(t.indexOf('подтверд')>=0) s+=8;
				if(t.indexOf('choose')>=0 || t.indexOf('выб')>=0) s+=4;
				if(t.indexOf('ok')>=0 || t.trim()==='ок') s+=4;
				if(t.indexOf('continue')>=0) s+=6;
				if(t.indexOf('done')>=0) s+=5;
				if(t.indexOf('add')>=0) s+=10;
				if(t.indexOf('cart')>=0 || t.indexOf('basket')>=0) s+=10;
				if(t.indexOf('отмена')>=0 || t.indexOf('cancel')>=0 || t.indexOf('закры')>=0) s-=10;
				// Strong penalty for upsells when combo/meal wasn't requested.
				if(!allowCombo){
					if(t.indexOf('комбо')>=0 || t.indexOf('combo')>=0 || t.indexOf('набор')>=0 || t.indexOf('meal')>=0 || t.indexOf('сет')>=0 || t.indexOf('set')>=0) s-=25;
					if(t.indexOf('напит')>=0 || t.indexOf('drink')>=0 || t.indexOf('картош')>=0 || t.indexOf('fries')>=0 || t.indexOf('соус')>=0 || t.indexOf('sauce')>=0 || t.indexOf('доп')>=0 || t.indexOf('extra')>=0) s-=10;
				}
				return s;
			}
			function isOverlay(el){
				if(!el) return false;
				try{
					var st=getComputedStyle(el); if(!st) return false;
					if(st.display==='none'||st.visibility==='hidden'||Number(st.opacity)===0) return false;
					var pos=st.position;
					if(pos!=='fixed' && pos!=='absolute') return false;
					var r=el.getBoundingClientRect();
					if(!r||r.width<200||r.height<120) return false;
					return true;
				}catch(e){return false;}
			}
			function z(el){
				try{var zi=parseInt(getComputedStyle(el).zIndex,10); return isFinite(zi)?zi:0;}catch(e){return 0;}
			}
			var cands = Array.from(document.querySelectorAll('[role="dialog"],[aria-modal="true"],dialog,div,section')).filter(function(el){
				try{
					var isDlg = (el.getAttribute && (el.getAttribute('role')==='dialog' || el.getAttribute('aria-modal')==='true')) || (el.tagName && el.tagName.toLowerCase()==='dialog');
					if(!isDlg && !isOverlay(el)) return false;
					return !!el.querySelector('button,[role="button"],input[type=submit],a[href]');
				}catch(e){return false;}
			});
			cands.sort(function(a,b){ return z(b)-z(a); });
			for(var i=0;i<cands.length && i<6;i++){
				var root=cands[i];
				var btns = Array.from(root.querySelectorAll('button,[role="button"],input[type=submit],a[href]')).filter(vis);
				var best=null, bestS=-999;
				for(var j=0;j<btns.length;j++){
					var b=btns[j];
					var sc=scoreBtn(b);
					if(b.tagName && b.tagName.toLowerCase()==='input') sc+=2;
					if(sc>bestS){ bestS=sc; best=b; }
				}
				if(best && bestS>=8){
					try{best.scrollIntoView({behavior:'instant',block:'center'});}catch(e){}
					try{best.focus();}catch(e){}
					try{best.click();}catch(e){}
					try{['pointerdown','mousedown','pointerup','mouseup','click'].forEach(function(tp){var ev=new MouseEvent(tp,{bubbles:true,cancelable:true,view:window}); best.dispatchEvent(ev);});}catch(e){}
					return true;
				}
			}
			return false;
		})()`, allowCombo)

		// Если агент кликает по карточке/ссылке товара, он часто открывает страницу товара вместо добавления.
		// Здесь делаем "умный клик": пытаемся найти ВНУТРИ этой карточки кнопку добавления в корзину и кликнуть по ней.
		query = strings.TrimSpace(a.getCurrentSearchQuery())
		smartAddJS := fmt.Sprintf(`(function(){
				var qSel = '%s';
				var query = '%s';
				var qt = String(query||'').toLowerCase();
				var words = qt.split(/\s+/).filter(function(w){ return w && w.length>=3; }).slice(0,5);

				function findEl(q){
					var el=null;
					try {
						if (q && q.startsWith('//')) el = document.evaluate(q, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
						else el = document.querySelector(q);
					} catch(e) {}
					return el;
				}
				function vis(el){
					if(!el) return false;
					try {
						var st = getComputedStyle(el);
						if(!st || st.display==='none' || st.visibility==='hidden' || Number(st.opacity)===0) return false;
						var r = el.getBoundingClientRect();
						return r && r.width>6 && r.height>6;
					} catch(e){ return false; }
				}
				function text(el){
					try { return (el.textContent||'') + ' ' + (el.getAttribute('aria-label')||'') + ' ' + (el.getAttribute('title')||''); } catch(e){ return (el&&el.textContent)||''; }
				}
				function hasSvg(el){
					try { return !!(el.querySelector && el.querySelector('svg,svg *,path,use')); } catch(e){ return false; }
				}
				function addScore(el){
					var t = text(el).toLowerCase();
					var cls = '';
					try { cls = String(el.className||''); } catch(e) { cls=''; }
					cls = cls.toLowerCase();
					var s = 0;
					if (t.indexOf('в корз')>=0) s += 14;
					if (t.indexOf('добав')>=0) s += 12;
					if (t.indexOf('заказ')>=0) s += 8;
					if (t.indexOf('куп')>=0) s += 6;
					if (t.indexOf('add')>=0) s += 12;
					if (t.indexOf('basket')>=0 || t.indexOf('cart')>=0) s += 12;
					var tt = ((el.textContent||'').trim());
					if (tt === '+') s += 8;
					if (!t.trim() && hasSvg(el)) s += 6;
					if (/(add|plus|cart|basket|buy|order|to-?cart)/.test(cls)) s += 5;
					if (t.indexOf('подробнее')>=0 || t.indexOf('details')>=0 || t.indexOf('описан')>=0 || t.indexOf('review')>=0) s -= 10;
					try {
						if (el.tagName && el.tagName.toLowerCase()==='a'){
							var href = (el.getAttribute('href')||'').trim().toLowerCase();
							if (href && href !== '#' && href.indexOf('javascript:')!==0) s -= 3;
						}
					} catch(e) {}
					return s;
				}
				function kwScore(root){
					if (!root || !words.length) return 0;
					var txt = (root.textContent||'').toLowerCase();
					var s = 0;
					for (var i=0;i<words.length;i++) if (txt.indexOf(words[i])>=0) s++;
					return s;
				}
				function looksLikeProductCard(root){
					if(!root) return false;
					try{
						var t = (root.textContent||'').toLowerCase();
						// price-ish signals (руб, ₽, р., $, €) or a bunch of digits
						if (/(₽|руб|\br\.|\$|€)/.test(t)) return true;
						if ((t.match(/\d{2,}/g)||[]).length>=1) return true;
					}catch(e){}
					return false;
				}
				function hardClick(el){
					if(!el) return;
					try { el.scrollIntoView({behavior:'instant', block:'center'}); } catch(e) {}
					try { el.focus(); } catch(e) {}
					try {
						['pointerdown','mousedown','pointerup','mouseup','click'].forEach(function(tp){
							var ev;
							if (window.PointerEvent && tp.indexOf('pointer')===0) ev = new PointerEvent(tp, {bubbles:true, cancelable:true, view:window});
							else ev = new MouseEvent(tp, {bubbles:true, cancelable:true, view:window});
							el.dispatchEvent(ev);
						});
					} catch(e) {}
					try { el.click(); } catch(e) {}
				}
				function nearestCard(el){
					var cur = el;
					for (var i=0;i<10 && cur;i++){
						try {
							var t = (cur.textContent||'');
							if (t.length>30 && t.length<8000 && cur.querySelector && cur.querySelector('button,[role="button"],a[href]')) return cur;
						} catch(e) {}
						cur = cur.parentElement;
					}
					return (el && el.parentElement) ? el.parentElement : null;
				}
				function bestAddIn(root){
					if(!root) return null;
					var cand = Array.from(root.querySelectorAll('button,[role="button"],a[href]')).filter(vis);
					var best=null, bestT=-999;
					var ks = kwScore(root);
					for (var i=0;i<cand.length;i++){
						var c=cand[i];
						var base = addScore(c);
						if (c.tagName && c.tagName.toLowerCase()==='button') base += 1;
						var total = base + ks*3;
						if (total>bestT){ bestT=total; best=c; }
					}
					if (best){
						var base = addScore(best);
						if (base>=10) return best;
						if (ks>=1 && base>=6) return best;
						// Важно для доставок еды: кнопка добавления часто без текста (иконка "+"/svg).
						// Внутри "похожей на товар" карточки разрешаем такие кнопки даже без матчей по словам.
						if (looksLikeProductCard(root) && base>=6 && (!words.length || ks>=1)) return best;
						if (bestT>=12) return best;
					}
					return null;
				}

				// 1) prefer within clicked element's card
				var el = findEl(qSel);
				if (el){
					var card = nearestCard(el);
					var b = bestAddIn(card);
					if (!b && card && card.parentElement) b = bestAddIn(card.parentElement);
					if (b){ hardClick(b); return true; }
				}

				// 1b) если есть текстовый запрос, ищем карточку по названию и кликаем "+/в корзину" внутри неё.
				if (words.length){
					function matchScore(t){
						if(!t) return 0;
						t = t.toLowerCase();
						var s=0;
						for (var i=0;i<words.length;i++) if (t.indexOf(words[i])>=0) s += (words[i].length>=6?2:1);
						return s;
					}
					function hasAddHint(card){
						try{
							var btns = Array.from(card.querySelectorAll('button,[role="button"],a[href]')).filter(vis);
							for (var i=0;i<btns.length && i<18;i++) if (addScore(btns[i])>=6) return true;
						}catch(e){}
						return false;
					}
					var texts = Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,span,div,p,a')).filter(vis);
					var bestCard=null, bestS=-1;
					for (var k=0;k<texts.length && k<700;k++){
						var te = texts[k];
						var sc = matchScore(te.textContent||'');
						if(sc<=0) continue;
						var card2=null;
						try { card2 = te.closest('li,article,section,div'); } catch(e) { card2 = te.parentElement; }
						if(!card2) continue;
						if(!hasAddHint(card2)) continue;
						// небольшой бонус за близость к верху (обычно первый экран списка)
						try { var r = card2.getBoundingClientRect(); if(r && r.top>=-40 && r.top<650) sc += 1; } catch(e) {}
						if(sc>bestS){ bestS=sc; bestCard=card2; }
					}
					if(bestCard){
						var bb2 = bestAddIn(bestCard);
						if(!bb2 && bestCard.parentElement) bb2 = bestAddIn(bestCard.parentElement);
						if(bb2){ hardClick(bb2); return true; }
					}
				}

				// 2) global scoring
				var btns = Array.from(document.querySelectorAll('button,[role="button"],a[href]')).filter(vis);
				var best2=null, best2T=-999;
				for (var j=0;j<btns.length;j++){
					var bb = btns[j];
					var base = addScore(bb);
					if (base<4 && words.length===0) continue;
					var root=null;
					try { root = bb.closest('li,article,section,div') || bb.parentElement; } catch(e){ root = bb.parentElement; }
					var ks = kwScore(root);
					if (ks===0 && root && root.parentElement) ks = kwScore(root.parentElement);
					// If we have a query, never add a random unrelated item.
					if (words.length && ks===0) continue;
					var total = base + ks*3;
					if (base>=10 || total>=12 || (ks>=1 && base>=6) || (looksLikeProductCard(root) && base>=6)){
						if (total>best2T){ best2T=total; best2=bb; }
					}
				}
				if (best2){ hardClick(best2); return true; }

				return false;
			})()`, jsEscapeForSingleQuotes(sel), jsEscapeForSingleQuotes(query))
		var smartClicked bool
		if addIntent {
			_ = chromedp.Run(opCtx, chromedp.Evaluate(smartAddJS, &smartClicked))
			if smartClicked {
				if addIntent {
					// For add-to-cart, do not perform any additional clicks in this action.
					// Poll for a UI update to avoid accidental double-adds.
					if post, changed := waitForCartChange(4 * time.Second); changed {
						return fmt.Sprintf("Добавил товар в корзину (умный клик; корзина: %s)", post), nil
					}
					var modalDid bool
					_ = chromedp.Run(opCtx, chromedp.Evaluate(modalConfirmJS, &modalDid))
					if modalDid {
						if post, changed := waitForCartChange(4 * time.Second); changed {
							return fmt.Sprintf("Добавил товар в корзину (подтверждение в модале; корзина: %s)", post), nil
						}
					}
					return "Нажал кнопку добавления (умный клик), жду обновления корзины", nil
				}
			}
		}

		// Перед кликом стараемся не плодить вкладки: снимаем target=_blank у элемента/ближайшей ссылки.
		prep := fmt.Sprintf(`(function(){
			var q = '%s';
			var el = null;
			try {
				if (q.startsWith('//')) {
					el = document.evaluate(q, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
				} else {
					el = document.querySelector(q);
				}
			} catch(e) {}
			if (!el) return false;
			try {
				var a = null;
				if (el.tagName && el.tagName.toLowerCase() === 'a') a = el;
				if (!a && el.closest) a = el.closest('a');
				if (a) {
					a.setAttribute('target', '_self');
					a.removeAttribute('rel');
				}
			} catch(e) {}
			return true;
		})();`, jsEscapeForSingleQuotes(sel))
		_ = chromedp.Run(opCtx, chromedp.Evaluate(prep, nil))

		// Попытка 1: дождаться наличия и кликнуть (короткий таймаут, чтобы успеть сделать fallback)
		err := runWithTimeout(12*time.Second,
			chromedp.WaitReady(sel, opt),
			chromedp.ScrollIntoView(sel, opt),
			chromedp.Click(sel, opt),
		)
		if err != nil {
			// Попытка 2: если селектор CSS — попробуем через BySearch (DOM.performSearch часто находит в сложных DOM/Shadow DOM)
			if isCSSQuery {
				err = runWithTimeout(12*time.Second,
					chromedp.WaitReady(sel, chromedp.BySearch),
					chromedp.ScrollIntoView(sel, chromedp.BySearch),
					chromedp.Click(sel, chromedp.BySearch),
				)
			}
		}
		if err != nil {
			// Попытка 3: JavaScript click (как крайний вариант)
			jsClick := fmt.Sprintf(`(function(){
				var q = '%s';
				var el = null;
				try {
					// Если похоже на XPath, используем document.evaluate
					if (q.startsWith('//')) {
						el = document.evaluate(q, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
					} else {
						el = document.querySelector(q);
					}
				} catch(e) {}
				if (el) {
					try { el.scrollIntoView({behavior:'instant', block:'center'}); } catch(e) {}
					try { el.click(); return true; } catch(e) {}
				}
				return false;
			})()`, jsEscapeForSingleQuotes(sel))

			var ok bool
			jerr := runWithTimeout(10*time.Second, chromedp.Evaluate(jsClick, &ok))
			if jerr != nil || !ok {
				return "", fmt.Errorf("не удалось кликнуть на %s: %v", action.Selector, err)
			}
		}

		time.Sleep(1 * time.Second)
		// If this click was intended to add to cart, we must avoid multi-click cascades
		// (they often lead to duplicates). We therefore poll for a cart badge change,
		// optionally confirm a modal once, and then stop.
		if addIntent {
			if post, changed := waitForCartChange(4 * time.Second); changed {
				return fmt.Sprintf("Клик выполнен, корзина обновилась (%s)", post), nil
			}
			var modalDid2 bool
			_ = chromedp.Run(opCtx, chromedp.Evaluate(modalConfirmJS, &modalDid2))
			if modalDid2 {
				if post, changed := waitForCartChange(4 * time.Second); changed {
					return fmt.Sprintf("Добавил в корзину после подтверждения в модале (%s)", post), nil
				}
			}
			return "Клик выполнен, но корзина не изменилась (возможно нужно выбрать опции/размер)", nil
		}
		return fmt.Sprintf("Кликнул на %s", sel), nil

	case "type":
		sel, opt, isCSSQuery := normalizeSelector(action.Selector)
		if sel == "" {
			return "", fmt.Errorf("пустой selector")
		}
		val := action.Value

		setJS := fmt.Sprintf(`(function(){
			var q = '%s';
			var el = null;
			try {
				if (q.startsWith('//')) el = document.evaluate(q, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
				else el = document.querySelector(q);
			} catch(e) {}
			if (!el) return false;
			try { el.scrollIntoView({behavior:'instant', block:'center'}); } catch(e) {}
			try { el.focus(); } catch(e) {}
			try {
				var v = '%s';
				if (el.isContentEditable) {
					el.innerText = v;
					el.dispatchEvent(new Event('input', {bubbles:true}));
					return true;
				}
				if ('value' in el) {
					// Use the native value setter to satisfy React/Vue controlled inputs.
					try{
						var proto = Object.getPrototypeOf(el);
						var desc = proto && Object.getOwnPropertyDescriptor(proto, 'value');
						if (desc && desc.set) desc.set.call(el, v);
						else el.value = v;
					}catch(e){ el.value = v; }
					try { el.setAttribute('value', v); } catch(e) {}
					try { if (el.setSelectionRange) el.setSelectionRange(v.length, v.length); } catch(e) {}
					el.dispatchEvent(new Event('input', {bubbles:true}));
					el.dispatchEvent(new Event('change', {bubbles:true}));
					return true;
				}
			} catch(e) {}
			return false;
		})()`, jsEscapeForSingleQuotes(sel), jsEscapeForSingleQuotes(val))

		clearJS := fmt.Sprintf(`(function(){
			var q = '%s';
			var el = null;
			try {
				if (q.startsWith('//')) el = document.evaluate(q, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
				else el = document.querySelector(q);
			} catch(e) {}
			if (!el) return false;
			try { el.scrollIntoView({behavior:'instant', block:'center'}); } catch(e) {}
			try { el.focus(); } catch(e) {}
			try {
				if (el.isContentEditable) {
					el.innerText = '';
					el.dispatchEvent(new Event('input', {bubbles:true}));
					return true;
				}
				if ('value' in el) {
					try{
						var proto = Object.getPrototypeOf(el);
						var desc = proto && Object.getOwnPropertyDescriptor(proto, 'value');
						if (desc && desc.set) desc.set.call(el, '');
						else el.value = '';
					}catch(e){ el.value = ''; }
					el.dispatchEvent(new Event('input', {bubbles:true}));
					el.dispatchEvent(new Event('change', {bubbles:true}));
					return true;
				}
			} catch(e) {}
			return false;
		})()`, jsEscapeForSingleQuotes(sel))

		// Попытка 1: стандартный ввод (короткий таймаут)
		err := runWithTimeout(15*time.Second,
			chromedp.WaitReady(sel, opt),
			chromedp.ScrollIntoView(sel, opt),
			chromedp.Click(sel, opt),
			chromedp.Focus(sel, opt),
			chromedp.Evaluate(clearJS, nil),
			chromedp.Evaluate(setJS, nil),
		)
		if err != nil {
			// Попытка 2: через BySearch
			if isCSSQuery {
				err = runWithTimeout(15*time.Second,
					chromedp.WaitReady(sel, chromedp.BySearch),
					chromedp.ScrollIntoView(sel, chromedp.BySearch),
					chromedp.Click(sel, chromedp.BySearch),
					chromedp.Focus(sel, chromedp.BySearch),
					chromedp.Evaluate(clearJS, nil),
					chromedp.Evaluate(setJS, nil),
				)
			}
		}
		if err != nil {
			// Попытка 3: JS выставить value + события (SPA часто слушают input/change)
			js := fmt.Sprintf(`(function(){
				var q = '%s';
				var el = null;
				try {
					if (q.startsWith('//')) {
						el = document.evaluate(q, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
					} else {
						el = document.querySelector(q);
					}
				} catch(e) {}
				if (!el) return false;
				try { el.scrollIntoView({behavior:'instant', block:'center'}); } catch(e) {}
				try { el.focus(); } catch(e) {}
				try {
					el.value = '%s';
					el.dispatchEvent(new Event('input', {bubbles:true}));
					el.dispatchEvent(new Event('change', {bubbles:true}));
					return true;
				} catch(e) {}
				return false;
			})()`, jsEscapeForSingleQuotes(sel), jsEscapeForSingleQuotes(val))
			var ok bool
			jerr := runWithTimeout(10*time.Second, chromedp.Evaluate(js, &ok))
			if jerr != nil || !ok {
				return "", fmt.Errorf("не удалось ввести текст: %v", err)
			}
		}
		return fmt.Sprintf("Ввёл '%s' в %s", val, sel), nil

	case "scroll":
		var script string
		switch action.Value {
		case "down":
			script = "window.scrollBy(0, 500)"
		case "up":
			script = "window.scrollBy(0, -500)"
		case "top":
			script = "window.scrollTo(0, 0)"
		case "bottom":
			script = "window.scrollTo(0, document.body.scrollHeight)"
		default:
			script = "window.scrollBy(0, 300)"
		}
		err := chromedp.Run(opCtx, chromedp.Evaluate(script, nil))
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("Прокрутил %s", action.Value), nil

	case "wait":
		duration := 2000
		fmt.Sscanf(action.Value, "%d", &duration)
		if duration > 10000 {
			duration = 10000
		}
		time.Sleep(time.Duration(duration) * time.Millisecond)
		return fmt.Sprintf("Подождал %dms", duration), nil

	case "extract":
		sel, opt, isCSSQuery := normalizeSelector(action.Selector)
		if sel == "" {
			return "", fmt.Errorf("пустой selector")
		}
		var text string
		err := runWithTimeout(30*time.Second,
			chromedp.WaitReady(sel, opt),
			chromedp.Text(sel, &text, opt),
		)
		if err != nil && isCSSQuery {
			// fallback через BySearch
			err = runWithTimeout(15*time.Second,
				chromedp.WaitReady(sel, chromedp.BySearch),
				chromedp.Text(sel, &text, chromedp.BySearch),
			)
		}
		if err != nil {
			// Fallback: direct JS extraction (avoids WaitReady hangs on shadow-DOM/overlays).
			js := fmt.Sprintf(`(function(){
				function find(q){
					try{if(q.startsWith('//'))return document.evaluate(q,document,null,XPathResult.FIRST_ORDERED_NODE_TYPE,null).singleNodeValue;}
					catch(e){}
					try{return document.querySelector(q);}catch(e){}
					return null;
				}
				var q='%s';
				var el=find(q);
				var t='';
				try{t=(el?el.innerText:document.body.innerText)||'';}catch(e){try{t=(el?el.textContent:document.body.textContent)||'';}catch(e2){t='';}}
				return t;
			})()`, jsEscapeForSingleQuotes(sel))
			var t2 string
			_ = runWithTimeout(15*time.Second, chromedp.Evaluate(js, &t2))
			text = t2
		}
		if len(text) > 1000 {
			text = text[:1000] + "..."
		}
		return fmt.Sprintf("Извлечённый текст: %s", text), nil

	case "screenshot":
		var buf []byte
		err := chromedp.Run(opCtx, chromedp.CaptureScreenshot(&buf))
		if err != nil {
			return "", err
		}
		// Можно сохранить в файл если нужно
		return "Скриншот сделан", nil

	case "none":
		return "Ожидание", nil

	default:
		return "", fmt.Errorf("неизвестное действие: %s", action.Action)
	}
}
