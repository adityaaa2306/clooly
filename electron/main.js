const { app, BrowserWindow, ipcMain } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const http = require("http");
const path = require("path");

const projectRoot = path.join(__dirname, "..");

let mainWindow;
let pythonProcess;
let healthCheckInterval;

app.disableHardwareAcceleration();

function backendHost() {
  return process.env.BACKEND_HOST || "localhost";
}

function backendPort() {
  return process.env.BACKEND_PORT || "8001";
}

function healthHost() {
  return backendHost() === "0.0.0.0" ? "localhost" : backendHost();
}

function findPythonExecutable() {
  if (process.env.PYTHON_EXECUTABLE) {
    return process.env.PYTHON_EXECUTABLE;
  }

  const venvPython = path.join(projectRoot, ".venv", "Scripts", "python.exe");
  if (fs.existsSync(venvPython)) {
    return venvPython;
  }

  return "python";
}

function checkHealth(retries = 30, delayMs = 500) {
  const port = backendPort();
  const host = healthHost();

  return new Promise((resolve, reject) => {
    let attempts = 0;

    const tryHealth = () => {
      attempts += 1;

      const req = http.get(`http://${host}:${port}/health`, (res) => {
        res.resume();

        if (res.statusCode === 200) {
          console.log(`[Electron] Backend health check passed on ${host}:${port}`);
          resolve();
          return;
        }

        console.log(`[Electron] Health returned ${res.statusCode}, retrying...`);
        if (attempts < retries) {
          setTimeout(tryHealth, delayMs);
        } else {
          reject(new Error(`Health check failed after ${retries} attempts`));
        }
      });

      req.on("error", (err) => {
        console.log(
          `[Electron] Health attempt ${attempts}/${retries} failed: ${err.message}`
        );

        if (attempts < retries) {
          setTimeout(tryHealth, delayMs);
        } else {
          reject(err);
        }
      });

      req.setTimeout(1200, () => {
        req.destroy(new Error("health check timed out"));
      });
    };

    tryHealth();
  });
}

function startPythonBackend() {
  const pythonExecutable = findPythonExecutable();
  const port = backendPort();
  const host = backendHost();

  console.log(`[Electron] Spawning Python backend with ${pythonExecutable}`);
  console.log(`[Electron] Backend target: ${host}:${port}`);

  pythonProcess = spawn(
    pythonExecutable,
    ["-m", "uvicorn", "backend.main:app", "--host", host, "--port", port],
    {
      cwd: projectRoot,
      env: {
        ...process.env,
        BACKEND_HOST: host,
        BACKEND_PORT: port,
      },
      stdio: ["ignore", "pipe", "pipe"],
      shell: false,
      windowsHide: true,
    }
  );

  pythonProcess.stdout.on("data", (data) => {
    console.log(`[Python] ${data.toString().trim()}`);
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error(`[Python] ${data.toString().trim()}`);
  });

  pythonProcess.on("error", (err) => {
    console.error("[Electron] Failed to start Python backend:", err);
  });

  pythonProcess.on("close", (code) => {
    console.log(`[Python] Process exited with code ${code}`);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 720,
    height: 360,
    x: 50,
    y: 50,
    frame: false,
    alwaysOnTop: true,
    transparent: true,
    backgroundColor: "#00000000",
    minWidth: 560,
    minHeight: 260,
    resizable: true,
    movable: true,
    show: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.setAlwaysOnTop(true, "screen-saver");

  mainWindow.webContents.on("console-message", (_event, level, message, line, sourceId) => {
    const levels = ["verbose", "info", "warning", "error"];
    const levelName = levels[level] || "log";
    console.log(`[Renderer ${levelName}] ${message} (${sourceId}:${line})`);
  });

  mainWindow.webContents.on("did-fail-load", (_event, errorCode, errorDescription) => {
    console.error(`[Electron] Renderer failed to load: ${errorCode} ${errorDescription}`);
  });

  mainWindow.webContents.on("did-finish-load", () => {
    console.log("[Electron] Renderer finished loading");
    if (!mainWindow.isVisible()) {
      mainWindow.show();
    }
  });

  const indexPath = path.join(projectRoot, "frontend", "index.html");
  console.log("[Electron] Loading index.html from:", indexPath);
  mainWindow.loadFile(indexPath);

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  healthCheckInterval = setInterval(() => {
    const req = http.get(`http://${healthHost()}:${backendPort()}/health`, (res) => {
      res.resume();
      if (res.statusCode !== 200) {
        console.warn("[Electron] Backend health check failed, status:", res.statusCode);
      }
    });

    req.on("error", (err) => {
      console.warn("[Electron] Backend health check error:", err.message);
    });

    req.setTimeout(1200, () => {
      req.destroy(new Error("health check timed out"));
    });
  }, 5000);
}

ipcMain.on("resize-window", (event, requestedHeight) => {
  if (!mainWindow || event.sender !== mainWindow.webContents) {
    return;
  }

  const height = Number(requestedHeight);
  if (!Number.isFinite(height)) {
    return;
  }

  const [width, currentHeight] = mainWindow.getSize();
  const nextHeight = Math.max(280, Math.min(760, Math.ceil(height)));
  if (Math.abs(nextHeight - currentHeight) > 4) {
    mainWindow.setSize(width, nextHeight, true);
  }
});

function cleanupAndQuit() {
  console.log("[Electron] Cleaning up...");

  if (healthCheckInterval) {
    clearInterval(healthCheckInterval);
    healthCheckInterval = null;
  }

  if (pythonProcess && !pythonProcess.killed) {
    try {
      pythonProcess.kill("SIGTERM");
      setTimeout(() => {
        if (pythonProcess && !pythonProcess.killed) {
          pythonProcess.kill("SIGKILL");
        }
      }, 3000);
    } catch (err) {
      console.error("[Electron] Error killing Python process:", err);
    }
  }

  app.quit();
}

app.on("ready", async () => {
  console.log("[Electron] App ready");
  startPythonBackend();

  try {
    await checkHealth();
    console.log("[Electron] Backend is healthy, creating window...");
    createWindow();
  } catch (err) {
    console.error("[Electron] Backend failed to start:", err);
    cleanupAndQuit();
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    cleanupAndQuit();
  }
});

app.on("activate", () => {
  if (mainWindow === null) {
    createWindow();
  }
});

app.on("before-quit", () => {
  if (pythonProcess && !pythonProcess.killed) {
    try {
      pythonProcess.kill("SIGTERM");
    } catch (err) {
      console.error("[Electron] Error during before-quit cleanup:", err);
    }
  }
});

process.on("uncaughtException", (err) => {
  console.error("[Electron] Uncaught exception:", err);
  cleanupAndQuit();
});
