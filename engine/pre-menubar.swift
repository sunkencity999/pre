// PRE Menu Bar — macOS status bar app for managing the PRE server
// Provides start/stop/restart, status indicator, and quick browser launch.
// Compile: swiftc -O -o PRE.app/Contents/MacOS/pre-menubar pre-menubar.swift -framework Cocoa

import Cocoa

// ── Configuration ──────────────────────────────────────────────────────────

let WEB_PORT = ProcessInfo.processInfo.environment["PRE_WEB_PORT"] ?? "7749"
let OLLAMA_PORT = ProcessInfo.processInfo.environment["PRE_PORT"] ?? "11434"
let PRE_URL = "http://localhost:\(WEB_PORT)"
let STATUS_CHECK_INTERVAL: TimeInterval = 5.0

// Resolve paths relative to this binary's location
let binaryPath = CommandLine.arguments[0]
let engineDir = (binaryPath as NSString).deletingLastPathComponent
    .replacingOccurrences(of: "/PRE.app/Contents/MacOS", with: "")
let webDir = (engineDir as NSString).deletingLastPathComponent + "/web"
let preServerScript = webDir + "/pre-server.sh"
let homeDir = FileManager.default.homeDirectoryForCurrentUser.path
let preDir = homeDir + "/.pre"

// ── App Delegate ───────────────────────────────────────────────────────────

class PreMenuBarApp: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem!
    var statusMenuItem: NSMenuItem!
    var ollamaMenuItem: NSMenuItem!
    var startItem: NSMenuItem!
    var stopItem: NSMenuItem!
    var restartItem: NSMenuItem!
    var timer: Timer?

    var serverRunning = false
    var ollamaRunning = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

        if let button = statusItem.button {
            updateIcon(running: false)
            button.toolTip = "PRE — Personal Reasoning Engine"
        }

        buildMenu()
        checkStatus()

        timer = Timer.scheduledTimer(withTimeInterval: STATUS_CHECK_INTERVAL, repeats: true) { [weak self] _ in
            self?.checkStatus()
        }
        RunLoop.main.add(timer!, forMode: .common)
    }

    // ── Menu Construction ──────────────────────────────────────────────────

    func buildMenu() {
        let menu = NSMenu()

        // Status section
        statusMenuItem = NSMenuItem(title: "Server: checking...", action: nil, keyEquivalent: "")
        statusMenuItem.isEnabled = false
        menu.addItem(statusMenuItem)

        ollamaMenuItem = NSMenuItem(title: "Ollama: checking...", action: nil, keyEquivalent: "")
        ollamaMenuItem.isEnabled = false
        menu.addItem(ollamaMenuItem)

        menu.addItem(NSMenuItem.separator())

        // Open in browser
        let openItem = NSMenuItem(title: "Open PRE", action: #selector(openPRE), keyEquivalent: "o")
        openItem.target = self
        menu.addItem(openItem)

        menu.addItem(NSMenuItem.separator())

        // Server controls
        startItem = NSMenuItem(title: "Start Server", action: #selector(startServer), keyEquivalent: "s")
        startItem.target = self
        menu.addItem(startItem)

        stopItem = NSMenuItem(title: "Stop Server", action: #selector(stopServer), keyEquivalent: "")
        stopItem.target = self
        menu.addItem(stopItem)

        restartItem = NSMenuItem(title: "Restart Server", action: #selector(restartServer), keyEquivalent: "r")
        restartItem.target = self
        menu.addItem(restartItem)

        menu.addItem(NSMenuItem.separator())

        // Launch CLI
        let cliItem = NSMenuItem(title: "Open CLI in Terminal", action: #selector(openCLI), keyEquivalent: "t")
        cliItem.target = self
        menu.addItem(cliItem)

        menu.addItem(NSMenuItem.separator())

        // Quit
        let quitItem = NSMenuItem(title: "Quit PRE Menu", action: #selector(quitApp), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)

        statusItem.menu = menu
    }

    // ── Icon ───────────────────────────────────────────────────────────────

    func updateIcon(running: Bool) {
        guard let button = statusItem.button else { return }

        // Clear any previous state
        button.image = nil
        button.title = ""

        // Try SF Symbol first (macOS 11+), then fall back to text
        var usedSymbol = false
        if #available(macOS 11.0, *) {
            if let img = NSImage(systemSymbolName: "brain.head.profile", accessibilityDescription: "PRE") {
                let config = NSImage.SymbolConfiguration(pointSize: 14, weight: .medium)
                if let configured = img.withSymbolConfiguration(config) {
                    configured.isTemplate = true
                    button.image = configured
                    button.imagePosition = .imageLeft
                    usedSymbol = true
                }
            }
        }

        // Status text — shown next to the icon, or standalone if no SF Symbol
        let dot = running ? "\u{25CF}" : "\u{25CB}"  // ● or ○
        let label = usedSymbol ? " \(dot)" : "PRE \(dot)"
        let color = running ? NSColor.systemGreen : NSColor.systemRed
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedSystemFont(ofSize: 12, weight: .medium),
            .foregroundColor: color,
        ]
        button.attributedTitle = NSAttributedString(string: label, attributes: attrs)
    }

    // ── Status Check ───────────────────────────────────────────────────────

    func checkStatus() {
        // Check PRE server
        checkHTTP(url: "\(PRE_URL)/api/status") { [weak self] ok in
            DispatchQueue.main.async {
                self?.serverRunning = ok
                self?.updateIcon(running: ok)
                self?.statusMenuItem.title = ok ? "Server: running on port \(WEB_PORT)" : "Server: stopped"
                self?.startItem.isHidden = ok
                self?.stopItem.isHidden = !ok
                self?.restartItem.isEnabled = ok
            }
        }

        // Check Ollama
        checkHTTP(url: "http://localhost:\(OLLAMA_PORT)/v1/models") { [weak self] ok in
            DispatchQueue.main.async {
                self?.ollamaRunning = ok
                self?.ollamaMenuItem.title = ok ? "Ollama: running" : "Ollama: not running"
            }
        }
    }

    func checkHTTP(url: String, completion: @escaping (Bool) -> Void) {
        guard let reqURL = URL(string: url) else {
            completion(false)
            return
        }
        var request = URLRequest(url: reqURL)
        request.timeoutInterval = 3
        request.httpMethod = "GET"

        URLSession.shared.dataTask(with: request) { _, response, error in
            if let httpResponse = response as? HTTPURLResponse, error == nil {
                completion((200...399).contains(httpResponse.statusCode))
            } else {
                completion(false)
            }
        }.resume()
    }

    // ── Actions ────────────────────────────────────────────────────────────

    @objc func openPRE() {
        if serverRunning {
            NSWorkspace.shared.open(URL(string: PRE_URL)!)
        } else {
            // Start server first, then open browser after a delay
            startServer()
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                NSWorkspace.shared.open(URL(string: PRE_URL)!)
            }
        }
    }

    @objc func startServer() {
        statusMenuItem.title = "Server: starting..."

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            // Ensure Ollama is running
            if !(self?.ollamaRunning ?? false) {
                let ollamaApp = "/Applications/Ollama.app"
                if FileManager.default.fileExists(atPath: ollamaApp) {
                    self?.runShell("open -g -a Ollama")
                } else {
                    self?.runShell("ollama serve >/dev/null 2>&1 &")
                }
                // Wait for Ollama to be ready
                for _ in 0..<15 {
                    Thread.sleep(forTimeInterval: 1)
                    let sem = DispatchSemaphore(value: 0)
                    var ready = false
                    self?.checkHTTP(url: "http://localhost:\(OLLAMA_PORT)/v1/models") { ok in
                        ready = ok
                        sem.signal()
                    }
                    sem.wait()
                    if ready { break }
                }
            }

            // Start the web server
            let serverJS = webDir + "/server.js"
            if FileManager.default.fileExists(atPath: serverJS) {
                self?.runShell("cd '\(webDir)' && nohup node server.js > /tmp/pre-server.log 2>&1 &")
            }

            Thread.sleep(forTimeInterval: 2)

            DispatchQueue.main.async {
                self?.checkStatus()
            }
        }
    }

    @objc func stopServer() {
        statusMenuItem.title = "Server: stopping..."
        runShell("lsof -ti :\(WEB_PORT) -sTCP:LISTEN | xargs kill 2>/dev/null")

        DispatchQueue.main.asyncAfter(deadline: .now() + 1) { [weak self] in
            self?.checkStatus()
        }
    }

    @objc func restartServer() {
        statusMenuItem.title = "Server: restarting..."
        runShell("lsof -ti :\(WEB_PORT) -sTCP:LISTEN | xargs kill 2>/dev/null")

        DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
            self?.startServer()
        }
    }

    @objc func openCLI() {
        let preLaunch = engineDir + "/pre-launch"
        let script: String
        if FileManager.default.fileExists(atPath: preLaunch) {
            script = "tell application \"Terminal\" to do script \"\(preLaunch)\""
        } else {
            script = "tell application \"Terminal\" to do script \"cd '\(webDir)' && node server.js\""
        }
        runAppleScript(script)
    }

    @objc func quitApp() {
        NSApplication.shared.terminate(nil)
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    @discardableResult
    func runShell(_ command: String) -> String {
        let task = Process()
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        task.executableURL = URL(fileURLWithPath: "/bin/zsh")
        task.arguments = ["-c", command]
        task.environment = ProcessInfo.processInfo.environment
        do {
            try task.run()
            task.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: data, encoding: .utf8) ?? ""
        } catch {
            return ""
        }
    }

    func runAppleScript(_ source: String) {
        if let script = NSAppleScript(source: source) {
            var error: NSDictionary?
            script.executeAndReturnError(&error)
        }
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

let app = NSApplication.shared
app.setActivationPolicy(.accessory) // No dock icon
let delegate = PreMenuBarApp()
app.delegate = delegate
app.run()
