// electron/preload.js - safe renderer bridge
const { contextBridge, ipcRenderer } = require("electron");

function backendHost() {
  return process.env.BACKEND_HOST || "localhost";
}

function backendPort() {
  return process.env.BACKEND_PORT || "8001";
}

contextBridge.exposeInMainWorld("electronAPI", {
  onMessage: (callback) => {
    const listener = (_event, data) => callback(data);
    ipcRenderer.on("backend-message", listener);
    return () => ipcRenderer.removeListener("backend-message", listener);
  },

  sendMessage: (channel, data) => {
    ipcRenderer.send(channel, data);
  },

  getBackendWSUrl: () => {
    return `ws://${backendHost()}:${backendPort()}/ws`;
  },

  getBackendHealthUrl: () => {
    return `http://${backendHost()}:${backendPort()}/health`;
  },

  resizeWindow: (height) => {
    ipcRenderer.send("resize-window", height);
  },
});
