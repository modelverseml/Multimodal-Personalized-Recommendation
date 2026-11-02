import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { api, prefs } from './api'

// The operator browses "as" a shopper: events they generate are written to
// that shopper's history, and recommendations update from it in real time.
const ShopperContext = createContext(null)

export function ShopperProvider({ children }) {
  const [shopper, setShopper] = useState(null)
  const [version, setVersion] = useState(0)

  useEffect(() => {
    const saved = prefs.get('shopper', null)
    if (saved) api.customer(saved).then(setShopper).catch(() => prefs.set('shopper', null))
  }, [])

  const select = useCallback(async (id) => {
    const detail = await api.customer(id)
    prefs.set('shopper', id)
    setShopper(detail)
    setVersion((v) => v + 1)
    return detail
  }, [])

  const createNew = useCallback(async () => {
    const created = await api.createCustomer()
    return select(created.id)
  }, [select])

  // Record a feedback event, then refresh the shopper so every view updates.
  const track = useCallback(async (productId, eventType, requestId) => {
    if (!shopper) return null
    const event = await api.event({
      customer_id: shopper.id, product_id: productId, event_type: eventType, request_id: requestId || null,
    })
    const detail = await api.customer(shopper.id)
    setShopper(detail)
    setVersion((v) => v + 1)
    return event
  }, [shopper])

  return (
    <ShopperContext.Provider value={{ shopper, version, select, createNew, track }}>
      {children}
    </ShopperContext.Provider>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export function useShopper() {
  return useContext(ShopperContext)
}
