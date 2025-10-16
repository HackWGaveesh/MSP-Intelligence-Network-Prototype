import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';

// Components
import Navbar from './components/Layout/Navbar';
import Sidebar from './components/Layout/Sidebar';
import LoadingSpinner from './components/UI/LoadingSpinner';
import ErrorBoundary from './components/UI/ErrorBoundary';

// Pages
import Dashboard from './pages/Dashboard';
import ThreatIntelligence from './pages/ThreatIntelligence';
import CollaborationPortal from './pages/CollaborationPortal';
import AnalyticsEngine from './pages/AnalyticsEngine';
import PrivacyControl from './pages/PrivacyControl';
import AgentOrchestrator from './pages/AgentOrchestrator';
import NetworkGraph from './pages/NetworkGraph';
import Settings from './pages/Settings';

// Hooks
import { useWebSocket } from './hooks/useWebSocket';
import { useTheme } from './hooks/useTheme';

// Store
import { useAppStore } from './store/appStore';

// Styles
import './App.css';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: attemptIndex => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

const App: React.FC = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  const { theme, toggleTheme } = useTheme();
  const { 
    isConnected, 
    connectionStatus, 
    lastMessage,
    initializeConnection,
    disconnect 
  } = useWebSocket();
  
  const { 
    systemStatus, 
    updateSystemStatus,
    networkMetrics,
    updateNetworkMetrics 
  } = useAppStore();

  // Initialize application
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Initialize WebSocket connection
        await initializeConnection();
        
        // Initialize system status
        updateSystemStatus({
          status: 'initializing',
          message: 'Starting MSP Intelligence Mesh Network...',
          timestamp: new Date().toISOString()
        });

        // Simulate initialization delay
        await new Promise(resolve => setTimeout(resolve, 2000));

        updateSystemStatus({
          status: 'ready',
          message: 'System ready - All agents operational',
          timestamp: new Date().toISOString()
        });

        setIsLoading(false);
      } catch (error) {
        console.error('Failed to initialize app:', error);
        updateSystemStatus({
          status: 'error',
          message: 'Failed to initialize system',
          timestamp: new Date().toISOString()
        });
        setIsLoading(false);
      }
    };

    initializeApp();

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [initializeConnection, disconnect, updateSystemStatus]);

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      const { type, data } = lastMessage;
      
      switch (type) {
        case 'threat_analysis':
          // Update threat intelligence data
          break;
        case 'collaboration_opportunity':
          // Update collaboration data
          break;
        case 'federated_learning_update':
          // Update federated learning status
          break;
        case 'network_simulation':
          // Update network metrics
          if (data.network_metrics) {
            updateNetworkMetrics(data.network_metrics);
          }
          break;
        default:
          console.log('Unhandled message type:', type);
      }
    }
  }, [lastMessage, updateNetworkMetrics]);

  // Apply theme to document
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-primary-900 via-primary-800 to-secondary-900 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="text-center"
        >
          <LoadingSpinner size="lg" />
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="mt-6 text-3xl font-bold text-white"
          >
            MSP Intelligence Mesh Network
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.5 }}
            className="mt-2 text-lg text-primary-200"
          >
            {systemStatus.message}
          </motion.p>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6, duration: 0.5 }}
            className="mt-4 flex items-center justify-center space-x-2 text-sm text-primary-300"
          >
            <div className={`w-2 h-2 rounded-full ${
              connectionStatus === 'connected' ? 'bg-success-500' : 'bg-warning-500'
            }`} />
            <span>WebSocket: {connectionStatus}</span>
          </motion.div>
        </motion.div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <Router>
          <div className={`min-h-screen transition-colors duration-300 ${
            theme === 'dark' 
              ? 'bg-dark-900 text-white' 
              : 'bg-gray-50 text-gray-900'
          }`}>
            {/* Navigation */}
            <Navbar 
              sidebarOpen={sidebarOpen}
              setSidebarOpen={setSidebarOpen}
              theme={theme}
              toggleTheme={toggleTheme}
              connectionStatus={connectionStatus}
              systemStatus={systemStatus}
            />

            {/* Sidebar */}
            <Sidebar 
              open={sidebarOpen}
              setOpen={setSidebarOpen}
              theme={theme}
            />

            {/* Main Content */}
            <main className={`transition-all duration-300 ${
              sidebarOpen ? 'lg:ml-64' : 'lg:ml-16'
            }`}>
              <AnimatePresence mode="wait">
                <Routes>
                  <Route path="/" element={<Navigate to="/dashboard" replace />} />
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/threat-intelligence" element={<ThreatIntelligence />} />
                  <Route path="/collaboration" element={<CollaborationPortal />} />
                  <Route path="/analytics" element={<AnalyticsEngine />} />
                  <Route path="/privacy" element={<PrivacyControl />} />
                  <Route path="/agents" element={<AgentOrchestrator />} />
                  <Route path="/network" element={<NetworkGraph />} />
                  <Route path="/settings" element={<Settings />} />
                  <Route path="*" element={<Navigate to="/dashboard" replace />} />
                </Routes>
              </AnimatePresence>
            </main>

            {/* Toast Notifications */}
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: theme === 'dark' ? '#1e293b' : '#ffffff',
                  color: theme === 'dark' ? '#ffffff' : '#1e293b',
                  border: `1px solid ${theme === 'dark' ? '#334155' : '#e2e8f0'}`,
                },
                success: {
                  iconTheme: {
                    primary: '#22c55e',
                    secondary: '#ffffff',
                  },
                },
                error: {
                  iconTheme: {
                    primary: '#ef4444',
                    secondary: '#ffffff',
                  },
                },
              }}
            />

            {/* Connection Status Indicator */}
            {connectionStatus !== 'connected' && (
              <motion.div
                initial={{ opacity: 0, y: 100 }}
                animate={{ opacity: 1, y: 0 }}
                className="fixed bottom-4 right-4 bg-warning-500 text-white px-4 py-2 rounded-lg shadow-lg flex items-center space-x-2"
              >
                <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                <span className="text-sm font-medium">
                  Reconnecting to network...
                </span>
              </motion.div>
            )}
          </div>
        </Router>
      </QueryClientProvider>
    </ErrorBoundary>
  );
};

export default App;
