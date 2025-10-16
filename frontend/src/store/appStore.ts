import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

// Types
export interface NetworkMetrics {
  connectedMSPs: number;
  threatAlerts: number;
  revenueGenerated: number;
  intelligenceLevel: number;
  networkHealth: number;
  activeCollaborations: number;
  modelsTrained: number;
  privacyScore: number;
}

export interface SystemStatus {
  status: 'initializing' | 'ready' | 'operational' | 'error' | 'maintenance';
  message: string;
  timestamp: string;
  uptime?: string;
  responseTime?: string;
}

export interface AgentStatus {
  agentId: string;
  agentType: string;
  status: 'active' | 'inactive' | 'error' | 'maintenance';
  healthScore: number;
  lastActivity: string;
  modelLoaded: boolean;
  performance: {
    totalRequests: number;
    successfulRequests: number;
    averageResponseTime: number;
  };
}

export interface ThreatData {
  threatId: string;
  threatType: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  title: string;
  description: string;
  detectedAt: string;
  confidence: number;
  affectedSystems: number;
  status: 'active' | 'investigating' | 'contained' | 'resolved';
}

export interface CollaborationOpportunity {
  opportunityId: string;
  type: string;
  title: string;
  description: string;
  estimatedValue: number;
  duration: number;
  requiredSkills: string[];
  status: 'open' | 'in_progress' | 'closed';
  createdAt: string;
  deadline: string;
}

export interface FederatedLearningStatus {
  currentRound: number;
  participants: string[];
  globalAccuracy: number;
  privacyBudgetUsed: number;
  privacyBudgetRemaining: number;
  convergenceStatus: 'converging' | 'converged' | 'diverging';
  lastUpdate: string;
}

// Store State Interface
interface AppState {
  // System Status
  systemStatus: SystemStatus;
  networkMetrics: NetworkMetrics;
  
  // Agent Management
  agents: AgentStatus[];
  selectedAgent: string | null;
  
  // Data Stores
  threats: ThreatData[];
  collaborationOpportunities: CollaborationOpportunity[];
  federatedLearningStatus: FederatedLearningStatus | null;
  
  // UI State
  sidebarOpen: boolean;
  theme: 'light' | 'dark';
  notifications: any[];
  
  // Actions
  updateSystemStatus: (status: Partial<SystemStatus>) => void;
  updateNetworkMetrics: (metrics: Partial<NetworkMetrics>) => void;
  updateAgentStatus: (agentId: string, status: Partial<AgentStatus>) => void;
  addThreat: (threat: ThreatData) => void;
  updateThreat: (threatId: string, updates: Partial<ThreatData>) => void;
  addCollaborationOpportunity: (opportunity: CollaborationOpportunity) => void;
  updateCollaborationOpportunity: (opportunityId: string, updates: Partial<CollaborationOpportunity>) => void;
  updateFederatedLearningStatus: (status: Partial<FederatedLearningStatus>) => void;
  setSelectedAgent: (agentId: string | null) => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  setTheme: (theme: 'light' | 'dark') => void;
  addNotification: (notification: any) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
}

// Initial State
const initialState = {
  systemStatus: {
    status: 'initializing' as const,
    message: 'Initializing MSP Intelligence Mesh Network...',
    timestamp: new Date().toISOString(),
    uptime: '0%',
    responseTime: '0ms'
  },
  
  networkMetrics: {
    connectedMSPs: 0,
    threatAlerts: 0,
    revenueGenerated: 0,
    intelligenceLevel: 0,
    networkHealth: 0,
    activeCollaborations: 0,
    modelsTrained: 0,
    privacyScore: 0
  },
  
  agents: [],
  selectedAgent: null,
  
  threats: [],
  collaborationOpportunities: [],
  federatedLearningStatus: null,
  
  sidebarOpen: false,
  theme: 'dark' as const,
  notifications: []
};

// Store Implementation
export const useAppStore = create<AppState>()(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,
        
        // System Status Actions
        updateSystemStatus: (status) =>
          set((state) => ({
            systemStatus: { ...state.systemStatus, ...status }
          }), false, 'updateSystemStatus'),
        
        updateNetworkMetrics: (metrics) =>
          set((state) => ({
            networkMetrics: { ...state.networkMetrics, ...metrics }
          }), false, 'updateNetworkMetrics'),
        
        // Agent Management Actions
        updateAgentStatus: (agentId, status) =>
          set((state) => ({
            agents: state.agents.map(agent =>
              agent.agentId === agentId
                ? { ...agent, ...status }
                : agent
            )
          }), false, 'updateAgentStatus'),
        
        setSelectedAgent: (agentId) =>
          set({ selectedAgent: agentId }, false, 'setSelectedAgent'),
        
        // Threat Management Actions
        addThreat: (threat) =>
          set((state) => ({
            threats: [threat, ...state.threats].slice(0, 100) // Keep last 100 threats
          }), false, 'addThreat'),
        
        updateThreat: (threatId, updates) =>
          set((state) => ({
            threats: state.threats.map(threat =>
              threat.threatId === threatId
                ? { ...threat, ...updates }
                : threat
            )
          }), false, 'updateThreat'),
        
        // Collaboration Management Actions
        addCollaborationOpportunity: (opportunity) =>
          set((state) => ({
            collaborationOpportunities: [opportunity, ...state.collaborationOpportunities].slice(0, 50)
          }), false, 'addCollaborationOpportunity'),
        
        updateCollaborationOpportunity: (opportunityId, updates) =>
          set((state) => ({
            collaborationOpportunities: state.collaborationOpportunities.map(opportunity =>
              opportunity.opportunityId === opportunityId
                ? { ...opportunity, ...updates }
                : opportunity
            )
          }), false, 'updateCollaborationOpportunity'),
        
        // Federated Learning Actions
        updateFederatedLearningStatus: (status) =>
          set((state) => ({
            federatedLearningStatus: state.federatedLearningStatus
              ? { ...state.federatedLearningStatus, ...status }
              : { ...status } as FederatedLearningStatus
          }), false, 'updateFederatedLearningStatus'),
        
        // UI Actions
        toggleSidebar: () =>
          set((state) => ({ sidebarOpen: !state.sidebarOpen }), false, 'toggleSidebar'),
        
        setSidebarOpen: (open) =>
          set({ sidebarOpen: open }, false, 'setSidebarOpen'),
        
        setTheme: (theme) =>
          set({ theme }, false, 'setTheme'),
        
        // Notification Actions
        addNotification: (notification) =>
          set((state) => ({
            notifications: [
              { ...notification, id: Date.now().toString(), timestamp: new Date().toISOString() },
              ...state.notifications
            ].slice(0, 20) // Keep last 20 notifications
          }), false, 'addNotification'),
        
        removeNotification: (id) =>
          set((state) => ({
            notifications: state.notifications.filter(notification => notification.id !== id)
          }), false, 'removeNotification'),
        
        clearNotifications: () =>
          set({ notifications: [] }, false, 'clearNotifications')
      }),
      {
        name: 'msp-intelligence-mesh-store',
        partialize: (state) => ({
          theme: state.theme,
          sidebarOpen: state.sidebarOpen,
          selectedAgent: state.selectedAgent
        })
      }
    ),
    {
      name: 'msp-intelligence-mesh-store'
    }
  )
);

// Selectors for common use cases
export const useSystemStatus = () => useAppStore((state) => state.systemStatus);
export const useNetworkMetrics = () => useAppStore((state) => state.networkMetrics);
export const useAgents = () => useAppStore((state) => state.agents);
export const useThreats = () => useAppStore((state) => state.threats);
export const useCollaborationOpportunities = () => useAppStore((state) => state.collaborationOpportunities);
export const useFederatedLearningStatus = () => useAppStore((state) => state.federatedLearningStatus);
export const useNotifications = () => useAppStore((state) => state.notifications);
