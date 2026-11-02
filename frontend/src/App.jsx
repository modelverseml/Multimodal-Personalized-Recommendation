import { Navigate, Route, Routes } from 'react-router-dom'
import { ProtectedRoute } from './auth'
import Layout from './components/Layout'
import Evaluation from './pages/Evaluation'
import ForYou from './pages/ForYou'
import Login from './pages/Login'
import Monitoring from './pages/Monitoring'
import Shop from './pages/Shop'

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route element={<ProtectedRoute><Layout /></ProtectedRoute>}>
        <Route index element={<ForYou />} />
        <Route path="shop" element={<Shop />} />
        <Route path="evaluation" element={<Evaluation />} />
        <Route path="monitoring" element={<Monitoring />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
