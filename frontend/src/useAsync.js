import { useCallback, useEffect, useState } from 'react'

// Keyed async state: `loading` is derived from whether the latest result
// belongs to the current deps, so effects never set state synchronously.
// Previous data stays visible while a new request is in flight.
export function useAsync(factory, deps) {
  const [nonce, setNonce] = useState(0)
  const key = JSON.stringify([...deps, nonce])
  const [state, setState] = useState({ key: null, data: undefined, error: null })

  useEffect(() => {
    let alive = true
    factory().then(
      (data) => alive && setState({ key, data, error: null }),
      (error) => alive && setState((s) => ({ key, data: s.data, error })),
    )
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  const reload = useCallback(() => setNonce((n) => n + 1), [])
  return { data: state.data, error: state.error, loading: state.key !== key, reload }
}
