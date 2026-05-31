import { Header } from './components/layout/Header'
import { Footer } from './components/layout/Footer'
import { DemoPage } from './pages/DemoPage'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1">
        <DemoPage />
      </main>
      <Footer />
    </div>
  )
}
