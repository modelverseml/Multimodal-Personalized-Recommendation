const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8001/api'
const API_ORIGIN = API_URL.replace(/\/api\/?$/, '')
const TOKEN_KEY = 'recsys_token'

function safeStorage(fn, fallback = null) {
  try {
    return fn()
  } catch {
    return fallback
  }
}

export const tokenStore = {
  get: () => safeStorage(() => localStorage.getItem(TOKEN_KEY)),
  set: (token) => safeStorage(() => localStorage.setItem(TOKEN_KEY, token)),
  clear: () => safeStorage(() => localStorage.removeItem(TOKEN_KEY)),
}

export const prefs = {
  get: (key, fallback) => safeStorage(() => JSON.parse(localStorage.getItem(`recsys_${key}`)) ?? fallback, fallback),
  set: (key, value) => safeStorage(() => localStorage.setItem(`recsys_${key}`, JSON.stringify(value))),
}

export class ApiError extends Error {
  constructor(status, message) {
    super(message)
    this.status = status
  }
}

async function request(path, { method = 'GET', body } = {}) {
  const headers = { 'Content-Type': 'application/json' }
  const token = tokenStore.get()
  if (token) headers.Authorization = `Bearer ${token}`

  const response = await fetch(`${API_URL}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  })

  if (response.status === 401 && token) {
    tokenStore.clear()
    window.dispatchEvent(new Event('auth:expired'))
  }

  const data = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = data?.detail
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg).join(', ')
      : detail || `Request failed (${response.status})`
    throw new ApiError(response.status, message)
  }
  return data
}

const qs = (params) => {
  const entries = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
  return entries.length ? `?${new URLSearchParams(entries)}` : ''
}

export const imageUrl = (product) => `${API_ORIGIN}/images/${product.asin}.jpg`

export const api = {
  login: (email, password) => request('/auth/login', { method: 'POST', body: { email, password } }),
  me: () => request('/auth/me'),
  categories: () => request('/catalog/categories'),
  products: (params) => request(`/catalog/products${qs(params)}`),
  product: (id) => request(`/catalog/products/${id}`),
  search: (q, k = 48) => request(`/catalog/search${qs({ q, k })}`),
  customers: (params = {}) => request(`/customers${qs(params)}`),
  customer: (id) => request(`/customers/${id}`),
  createCustomer: (displayName) => request('/customers', { method: 'POST', body: { display_name: displayName || null } }),
  recommendations: (id, params = {}) => request(`/recommendations/${id}${qs(params)}`),
  event: (body) => request('/events', { method: 'POST', body }),
  model: () => request('/model'),
  evaluation: () => request('/evaluation'),
  monitoring: (days = 7) => request(`/monitoring${qs({ days })}`),
}
