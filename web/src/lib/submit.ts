/// Wraps an async action so calls made while it runs are ignored (a double
/// tap on Confirm sends one move).
export function oneAtATime<T>(fn: () => Promise<T>): () => Promise<T | undefined> {
  let running = false;
  return async () => {
    if (running) return undefined;
    running = true;
    try {
      return await fn();
    } finally {
      running = false;
    }
  };
}
