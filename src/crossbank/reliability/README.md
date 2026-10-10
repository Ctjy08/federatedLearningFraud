# Phase 4 TODO — coordinator reliability

Implement a *networked* round manager with the following rules:
1. Each update includes `client_id`, `round_id`, `base_model_version`, and a signed/authenticated transport identity.
2. Reject duplicates and stale updates, keeping state changes atomic.
3. Start aggregation only when the minimum distinct-client quorum is satisfied; support deadline policy and minimum quorum explicitly.
4. Checkpoint each completed global model with version/round metadata on durable storage.
5. Handle crash/restart by reloading checkpoint and preventing accidental double aggregation.
6. Separate client liveness (Kubernetes pod restart) from training protocol recovery (application logic).

Test partial failure, concurrent submission, stale updates, restarts, and missed quorum. **No security or reliability guarantees are claimed until implemented and tested.**
