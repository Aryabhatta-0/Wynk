import { useCallback, useEffect, useRef, useState } from "react";
import { errorMessage } from "@/api";

export type Resource<T> =
  | { state: "loading"; data: null; error: null; reload: () => void }
  | { state: "error"; data: null; error: string; reload: () => void }
  | { state: "ready"; data: T; error: null; reload: () => void };

interface Options<T> {
  /** refetch every `pollMs` while `shouldPoll(data)` is true */
  pollMs?: number;
  shouldPoll?: (data: T) => boolean;
}

/**
 * Loads data through the adapter and tracks loading, error and ready states. A failed poll
 * keeps the last good data on screen rather than blanking it.
 */
export function useResource<T>(load: () => Promise<T>, deps: unknown[], options: Options<T> = {}): Resource<T> {
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>({ data: null, error: null, loading: true });
  const [nonce, setNonce] = useState(0);
  const loadRef = useRef(load);
  loadRef.current = load;
  const opts = useRef(options);
  opts.current = options;

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    setState((s) => ({ data: s.data, error: null, loading: s.data === null }));

    const run = async () => {
      try {
        const data = await loadRef.current();
        if (!alive) return;
        setState({ data, error: null, loading: false });
        const { pollMs, shouldPoll } = opts.current;
        if (pollMs && shouldPoll?.(data)) timer = setTimeout(run, pollMs);
      } catch (err) {
        if (!alive) return;
        setState((s) => (s.data !== null && opts.current.pollMs ? s : { data: null, error: errorMessage(err), loading: false }));
        if (opts.current.pollMs) timer = setTimeout(run, opts.current.pollMs * 2);
      }
    };
    run();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  if (state.loading) return { state: "loading", data: null, error: null, reload };
  if (state.error !== null || state.data === null) return { state: "error", data: null, error: state.error ?? "Nothing was returned.", reload };
  return { state: "ready", data: state.data, error: null, reload };
}
