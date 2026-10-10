# Phase 3 — Docker plan

Create separate images for (1) coordinator and (2) bank trainer once the networked FL service exists. Expose an authenticated service port, inject configs via environment variables, and mount each bank dataset in only its client container. Do not fake a distributed deployment by placing all datasets in a shared image.
