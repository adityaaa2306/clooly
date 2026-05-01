{
  "name": "cluely-mvp",
  "version": "1.0.0",
  "description": "Real-time AI interview copilot — Electron desktop app",
  "main": "electron/main.js",
  "scripts": {
    "start": "electron .",
    "dev": "concurrently \"npm run build:watch\" \"electron .\"",
    "build": "esbuild frontend/index.jsx --bundle --outfile=frontend/dist/bundle.js --loader:.jsx=jsx",
    "build:watch": "esbuild frontend/index.jsx --bundle --outfile=frontend/dist/bundle.js --loader:.jsx=jsx --watch",
    "lint": "eslint electron/ frontend/"
  },
  "dependencies": {
    "dotenv": "^16.4.5",
    "electron-store": "^10.0.0"
  },
  "devDependencies": {
    "concurrently": "^9.0.1",
    "electron": "^32.2.0",
    "esbuild": "^0.24.0",
    "eslint": "^9.13.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "keywords": [
    "interview",
    "copilot",
    "ai",
    "electron",
    "real-time"
  ],
  "author": "",
  "license": "MIT"
}