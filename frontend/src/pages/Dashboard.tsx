import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { 
  ShieldCheckIcon, 
  UsersIcon, 
  ChartBarIcon, 
  CpuChipIcon,
  GlobeAltIcon,
  BoltIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';

// Components
import MetricCard from '../components/Dashboard/MetricCard';
import NetworkGraph from '../components/Dashboard/NetworkGraph';
import ThreatHeatmap from '../components/Dashboard/ThreatHeatmap';
import ActivityFeed from '../components/Dashboard/ActivityFeed';
import PerformanceChart from '../components/Dashboard/PerformanceChart';
import AgentStatusGrid from '../components/Dashboard/AgentStatusGrid';
import RealTimeMetrics from '../components/Dashboard/RealTimeMetrics';

// Hooks
import { useWebSocket } from '../hooks/useWebSocket';
import { useAppStore } from '../store/appStore';

// Types
import { NetworkMetrics, SystemStatus } from '../types/dashboard';

const Dashboard: React.FC = () => {
  const [networkMetrics, setNetworkMetrics] = useState<NetworkMetrics>({
    connectedMSPs: 1247,
    threatAlerts: 23,
    revenueGenerated: 890000,
    intelligenceLevel: 94.2,
    networkHealth: 97.8,
    activeCollaborations: 15,
    modelsTrained: 8,
    privacyScore: 98.5
  });

  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    status: 'operational',
    uptime: '99.9%',
    responseTime: '23ms',
    lastUpdate: new Date().toISOString()
  });

  const { lastMessage } = useWebSocket();
  const { networkMetrics: storeMetrics, updateNetworkMetrics } = useAppStore();

  // Update metrics from WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      const { type, data } = lastMessage;
      
      if (type === 'network_simulation' && data.network_metrics) {
        setNetworkMetrics(prev => ({
          ...prev,
          ...data.network_metrics
        }));
        updateNetworkMetrics(data.network_metrics);
      }
      
      if (type === 'threat_analysis') {
        setNetworkMetrics(prev => ({
          ...prev,
          threatAlerts: prev.threatAlerts + 1
        }));
      }
    }
  }, [lastMessage, updateNetworkMetrics]);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setNetworkMetrics(prev => ({
        ...prev,
        connectedMSPs: prev.connectedMSPs + Math.floor(Math.random() * 3) - 1,
        threatAlerts: Math.max(0, prev.threatAlerts + Math.floor(Math.random() * 3) - 1),
        intelligenceLevel: Math.min(99.9, prev.intelligenceLevel + (Math.random() - 0.5) * 0.1),
        networkHealth: Math.min(99.9, prev.networkHealth + (Math.random() - 0.5) * 0.05)
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const metricCards = [
    {
      title: 'Connected MSPs',
      value: networkMetrics.connectedMSPs.toLocaleString(),
      change: '+12.5%',
      changeType: 'positive' as const,
      icon: UsersIcon,
      color: 'primary'
    },
    {
      title: 'Threat Alerts',
      value: networkMetrics.threatAlerts.toString(),
      change: '-8.2%',
      changeType: 'positive' as const,
      icon: ShieldCheckIcon,
      color: 'success'
    },
    {
      title: 'Revenue Generated',
      value: `$${networkMetrics.revenueGenerated.toLocaleString()}`,
      change: '+35.4%',
      changeType: 'positive' as const,
      icon: ChartBarIcon,
      color: 'success'
    },
    {
      title: 'Intelligence Level',
      value: `${networkMetrics.intelligenceLevel.toFixed(1)}%`,
      change: '+2.1%',
      changeType: 'positive' as const,
      icon: CpuChipIcon,
      color: 'primary'
    },
    {
      title: 'Network Health',
      value: `${networkMetrics.networkHealth.toFixed(1)}%`,
      change: '+0.3%',
      changeType: 'positive' as const,
      icon: GlobeAltIcon,
      color: 'success'
    },
    {
      title: 'Active Collaborations',
      value: networkMetrics.activeCollaborations.toString(),
      change: '+18.7%',
      changeType: 'positive' as const,
      icon: UsersIcon,
      color: 'accent'
    },
    {
      title: 'Models Trained',
      value: networkMetrics.modelsTrained.toString(),
      change: '+25.0%',
      changeType: 'positive' as const,
      icon: BoltIcon,
      color: 'warning'
    },
    {
      title: 'Privacy Score',
      value: `${networkMetrics.privacyScore.toFixed(1)}%`,
      change: '+1.2%',
      changeType: 'positive' as const,
      icon: ShieldCheckIcon,
      color: 'success'
    }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5
      }
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Network Intelligence Dashboard
        </h1>
        <p className="mt-2 text-lg text-gray-600 dark:text-gray-300">
          Real-time monitoring of the MSP Intelligence Mesh Network
        </p>
        
        {/* System Status */}
        <div className="mt-4 flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-success-500 rounded-full animate-pulse" />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              System Operational
            </span>
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">
            Uptime: {systemStatus.uptime} | Response: {systemStatus.responseTime}
          </div>
        </div>
      </motion.div>

      {/* Metrics Grid */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        {metricCards.map((metric, index) => (
          <motion.div key={metric.title} variants={itemVariants}>
            <MetricCard {...metric} />
          </motion.div>
        ))}
      </motion.div>

      {/* Main Content Grid */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 lg:grid-cols-3 gap-6"
      >
        {/* Network Graph */}
        <motion.div variants={itemVariants} className="lg:col-span-2">
          <div className="bg-white dark:bg-dark-800 rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                Network Topology
              </h2>
              <div className="flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
                <div className="w-2 h-2 bg-success-500 rounded-full animate-pulse" />
                <span>Live</span>
              </div>
            </div>
            <NetworkGraph />
          </div>
        </motion.div>

        {/* Real-time Metrics */}
        <motion.div variants={itemVariants}>
          <div className="bg-white dark:bg-dark-800 rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              Real-time Metrics
            </h2>
            <RealTimeMetrics />
          </div>
        </motion.div>
      </motion.div>

      {/* Secondary Content Grid */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        {/* Threat Heatmap */}
        <motion.div variants={itemVariants}>
          <div className="bg-white dark:bg-dark-800 rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                Threat Distribution
              </h2>
              <div className="flex items-center space-x-2">
                <ExclamationTriangleIcon className="w-5 h-5 text-warning-500" />
                <span className="text-sm text-warning-600 dark:text-warning-400">
                  {networkMetrics.threatAlerts} Active
                </span>
              </div>
            </div>
            <ThreatHeatmap />
          </div>
        </motion.div>

        {/* Performance Chart */}
        <motion.div variants={itemVariants}>
          <div className="bg-white dark:bg-dark-800 rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              Performance Trends
            </h2>
            <PerformanceChart />
          </div>
        </motion.div>
      </motion.div>

      {/* Bottom Grid */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 lg:grid-cols-3 gap-6"
      >
        {/* Agent Status */}
        <motion.div variants={itemVariants} className="lg:col-span-1">
          <div className="bg-white dark:bg-dark-800 rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                Agent Status
              </h2>
              <div className="flex items-center space-x-2">
                <CheckCircleIcon className="w-5 h-5 text-success-500" />
                <span className="text-sm text-success-600 dark:text-success-400">
                  All Operational
                </span>
              </div>
            </div>
            <AgentStatusGrid />
          </div>
        </motion.div>

        {/* Activity Feed */}
        <motion.div variants={itemVariants} className="lg:col-span-2">
          <div className="bg-white dark:bg-dark-800 rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              Live Activity Feed
            </h2>
            <ActivityFeed />
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default Dashboard;
