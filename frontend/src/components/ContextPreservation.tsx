import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

interface SessionState {
  exists: boolean;
  session_id: number;
  project_id: number;
  current_step: number;
  total_steps: number;
  completion_percent: number;
  state_version: number;
  last_snapshot_at: string | null;
}

interface ConversationMessage {
  id: number;
  role: 'user' | 'assistant' | 'system';
  content: string;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

interface Checkpoint {
  id: number;
  checkpoint_type: string;
  step_number: number | null;
  description: string | null;
  created_at: string;
}

interface SessionStateCardProps {
  sessionId: number;
  projectId: number;
}

interface ConversationPanelProps {
  sessionId: number;
}

interface CheckpointListProps {
  taskId: number;
}

export function SessionStateCard({ sessionId, projectId }: SessionStateCardProps) {
  const [state, setState] = useState<SessionState | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchState = useCallback(async () => {
    try {
      const response = await axios.get(`/api/v1/context/state/${sessionId}`);
      setState(response.data);
    } catch (error) {
      console.error('Failed to fetch state:', error);
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    fetchState();
  }, [fetchState]);

  const handleSnapshot = async () => {
    try {
      await axios.post('/api/v1/context/snapshot', {
        session_id: sessionId,
        project_id: projectId,
        current_step: state?.current_step || 0,
        total_steps: state?.total_steps || 0,
      });
      fetchState(); // Refresh state
    } catch (error) {
      console.error('Failed to save state:', error);
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Session State</h3>
        <div className="text-gray-500">Loading...</div>
      </div>
    );
  }

  if (!state?.exists) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Session State</h3>
        <p className="text-gray-500 mb-4">No state saved yet</p>
        <button
          onClick={handleSnapshot}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Save State
        </button>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">Session State</h3>
        <button
          onClick={handleSnapshot}
          className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600 text-sm"
        >
          Save
        </button>
      </div>

      <div className="space-y-3">
        <div>
          <div className="text-sm text-gray-600 mb-1">Progress</div>
          <div className="flex items-center">
            <div className="flex-1 bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all"
                style={{ width: `${state.completion_percent}%` }}
              />
            </div>
            <span className="ml-2 text-sm font-medium">
              {state.current_step}/{state.total_steps}
            </span>
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {state.completion_percent.toFixed(1)}% complete
          </div>
        </div>

        <div className="text-sm">
          <div className="text-gray-600">Version: {state.state_version}</div>
          <div className="text-gray-600">
            Last saved: {state.last_snapshot_at ? new Date(state.last_snapshot_at).toLocaleString() : 'Never'}
          </div>
        </div>
      </div>
    </div>
  );
}

export function ConversationPanel({ sessionId }: ConversationPanelProps) {
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);

  const fetchMessages = useCallback(async () => {
    try {
      const response = await axios.get(`/api/v1/context/conversation/${sessionId}`, {
        params: { limit: 50 },
      });
      setMessages(response.data.messages);
    } catch (error) {
      console.error('Failed to fetch messages:', error);
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    fetchMessages();
  }, [fetchMessages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    try {
      // Add user message
      await axios.post('/api/v1/context/conversation', {
        session_id: sessionId,
        role: 'user',
        content: input,
      });

      setInput('');
      fetchMessages();
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Conversation History</h3>
        <div className="text-gray-500">Loading...</div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Conversation History</h3>

      <div className="space-y-3 mb-4 max-h-64 overflow-y-auto">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`p-3 rounded-lg ${
              msg.role === 'user'
                ? 'bg-blue-50 ml-8'
                : msg.role === 'assistant'
                ? 'bg-gray-50 mr-8'
                : 'bg-yellow-50'
            }`}
          >
            <div className="text-xs font-semibold uppercase text-gray-500 mb-1">
              {msg.role}
            </div>
            <div className="text-sm text-gray-800">{msg.content}</div>
            <div className="text-xs text-gray-400 mt-1">
              {new Date(msg.created_at).toLocaleTimeString()}
            </div>
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Add message..."
          className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          type="submit"
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
        >
          Send
        </button>
      </form>
    </div>
  );
}

export function CheckpointList({ taskId }: CheckpointListProps) {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchCheckpoints = useCallback(async () => {
    try {
      const response = await axios.get(`/api/v1/context/checkpoints/${taskId}`);
      setCheckpoints(response.data.checkpoints);
    } catch (error) {
      console.error('Failed to fetch checkpoints:', error);
    } finally {
      setLoading(false);
    }
  }, [taskId]);

  useEffect(() => {
    fetchCheckpoints();
  }, [fetchCheckpoints]);

  const handleResume = async (checkpointId: number) => {
    try {
      await axios.post(`/api/v1/context/resume/${taskId}`, {
        checkpoint_id: checkpointId,
      });
      alert(`Resumed from checkpoint ${checkpointId}`);
    } catch (error) {
      console.error('Failed to resume:', error);
      alert('Failed to resume from checkpoint');
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Task Checkpoints</h3>
        <div className="text-gray-500">Loading...</div>
      </div>
    );
  }

  if (checkpoints.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Task Checkpoints</h3>
        <p className="text-gray-500">No checkpoints found</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Task Checkpoints</h3>

      <div className="space-y-3">
        {checkpoints.map((checkpoint) => (
          <div
            key={checkpoint.id}
            className="border border-gray-200 rounded-lg p-4 hover:border-blue-300 transition-colors"
          >
            <div className="flex justify-between items-start mb-2">
              <div>
                <span className="text-xs font-semibold px-2 py-1 bg-blue-100 text-blue-800 rounded">
                  {checkpoint.checkpoint_type}
                </span>
                {checkpoint.step_number && (
                  <span className="ml-2 text-sm text-gray-600">
                    Step {checkpoint.step_number}
                  </span>
                )}
              </div>
              <div className="text-xs text-gray-400">
                {new Date(checkpoint.created_at).toLocaleString()}
              </div>
            </div>

            {checkpoint.description && (
              <div className="text-sm text-gray-700 mb-2">
                {checkpoint.description}
              </div>
            )}

            <button
              onClick={() => handleResume(checkpoint.id)}
              className="px-3 py-1 bg-green-500 text-white rounded text-sm hover:bg-green-600"
            >
              Resume from here
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ContextDashboard({ sessionId, projectId }: { sessionId: number; projectId: number }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <SessionStateCard sessionId={sessionId} projectId={projectId} />
      <ConversationPanel sessionId={sessionId} />
    </div>
  );
}
