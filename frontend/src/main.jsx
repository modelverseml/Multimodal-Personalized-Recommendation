import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { AuthProvider } from './auth'
import { ShopperProvider } from './shopper'
import App from './App.jsx'
import './styles.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <AuthProvider>
        <ShopperProvider>
          <App />
        </ShopperProvider>
      </AuthProvider>
    </BrowserRouter>
  </StrictMode>,
)
