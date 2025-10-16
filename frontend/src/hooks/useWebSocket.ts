import { useEffect, useRef, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';

interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  lastMessage: WebSocketMessage | null;
  sendMessage: (message: any) => void;
  initializeConnection: () => Promise<void>;
  disconnect: () => void;
}

export const useWebSocket = (): UseWebSocketReturn => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  
  const socketRef = useRef<Socket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const initializeConnection = useCallback(async () => {
    if (socketRef.current?.connected) {
      return;
    }

    try {
      setConnectionStatus('connecting');
      
      // Create WebSocket connection
      const socket = io('ws://localhost:8000/ws', {
        transports: ['websocket'],
        timeout: 10000,
        reconnection: true,
        reconnectionAttempts: maxReconnectAttempts,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
      });

      socketRef.current = socket;

      // Connection event handlers
      socket.on('connect', () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setConnectionStatus('connected');
        reconnectAttempts.current = 0;
        
        // Send initial ping
        socket.emit('ping');
      });

      socket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason);
        setIsConnected(false);
        setConnectionStatus('disconnected');
      });

      socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        setConnectionStatus('error');
        setIsConnected(false);
      });

      // Message handlers
      socket.on('threat_analysis', (data) => {
        setLastMessage({
          type: 'threat_analysis',
          data,
          timestamp: new Date().toISOString()
        });
      });

      socket.on('collaboration_opportunity', (data) => {
        setLastMessage({
          type: 'collaboration_opportunity',
          data,
          timestamp: new Date().toISOString()
        });
      });

      socket.on('federated_learning_update', (data) => {
        setLastMessage({
          type: 'federated_learning_update',
          data,
          timestamp: new Date().toISOString()
        });
      });

      socket.on('network_simulation', (data) => {
        setLastMessage({
          type: 'network_simulation',
          data,
          timestamp: new Date().toISOString()
        });
      });

      socket.on('workflow_completed', (data) => {
        setLastMessage({
          type: 'workflow_completed',
          data,
          timestamp: new Date().toISOString()
        });
      });

      socket.on('threat_detection_simulation', (data) => {
        setLastMessage({
          type: 'threat_detection_simulation',
          data,
          timestamp: new Date().toISOString()
        });
      });

      socket.on('collaboration_opportunity_simulation', (data) => {
        setLastMessage({
          type: 'collaboration_opportunity_simulation',
          data,
          timestamp: new Date().toISOString()
        });
      });

      socket.on('federated_learning_simulation', (data) => {
        setLastMessage({
          type: 'federated_learning_simulation',
          data,
          timestamp: new Date().toISOString()
        });
      });

      socket.on('pong', () => {
        // Handle pong response
        console.log('Received pong from server');
      });

      // Generic message handler for any other message types
      socket.onAny((eventName, data) => {
        if (!['connect', 'disconnect', 'connect_error', 'pong'].includes(eventName)) {
          setLastMessage({
            type: eventName,
            data,
            timestamp: new Date().toISOString()
          });
        }
      });

    } catch (error) {
      console.error('Failed to initialize WebSocket connection:', error);
      setConnectionStatus('error');
      setIsConnected(false);
    }
  }, []);

  const sendMessage = useCallback((message: any) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit('message', message);
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    
    setIsConnected(false);
    setConnectionStatus('disconnected');
  }, []);

  // Auto-reconnect logic
  useEffect(() => {
    if (!isConnected && connectionStatus === 'disconnected' && reconnectAttempts.current < maxReconnectAttempts) {
      reconnectTimeoutRef.current = setTimeout(() => {
        reconnectAttempts.current += 1;
        initializeConnection();
      }, 2000 * reconnectAttempts.current);
    }

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [isConnected, connectionStatus, initializeConnection]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    isConnected,
    connectionStatus,
    lastMessage,
    sendMessage,
    initializeConnection,
    disconnect
  };
};
