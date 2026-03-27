# Project Rules

## Server Port Policy

When starting any development server (uvicorn, flask, fastapi, node, etc.):
- **NEVER** use common ports: 3000, 5000, 8000, 8080
- **ALWAYS** use port 8765 or higher (e.g., 8765, 8766, 9000)
- Before starting, check if the target port is already in use with `lsof -i:<port>` and pick another if occupied
- This prevents conflicts with other running services
