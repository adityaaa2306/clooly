// frontend/index.jsx - React entry point
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

const debugLog = window.cluelyDebugLog || ((message) => console.log(message));

try {
  debugLog("[REACT] index.jsx loaded");

  const rootElement = document.getElementById("root");
  if (!rootElement) {
    throw new Error("React mount element #root was not found");
  }

  debugLog("[REACT] Root element found");

  const root = createRoot(rootElement);
  debugLog("[REACT] ReactDOM root created");

  root.render(<App />);
  document.documentElement.dataset.reactReady = "true";
  debugLog("[REACT] App component rendered");
} catch (error) {
  const message = error && error.stack ? error.stack.split("\n")[0] : String(error);
  debugLog("[REACT-ERROR] " + message, "error");

  const rootElement = document.getElementById("root");
  if (rootElement) {
    rootElement.innerHTML =
      '<div class="boot-splash" style="color:#ff7b7b;">React failed to render</div>';
  }

  throw error;
}
