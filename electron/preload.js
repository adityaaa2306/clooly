// electron/preload.js — Preload Script
// Exposes a safe, typed API to the React renderer via contextBridge.
// React accesses it as: window.electronAPI.onMessage(callback)

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  // Receive messages from the main process (e.g. backend events forwarded via ipcMain)
  onMessage: (callback) => {
    ipcRenderer.on("backend-message", (_event, data) => callback(data));
  },

  // Send messages to the main process
  sendMessage: (channel, data) => {
    ipcRenderer.send(channel, data);
  },

  // Backend WebSocket URL (renderer connects directly for low latency)
  getBackendWSUrl: () => {
    return `ws://${process.env.BACKEND_HOST || "localhost"}:${process.env.BACKEND_PORT || 8000}/ws`;
  },
});