import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

interface PermissionRequest {
  id: number;
  project_id: number;
  session_id: number;
  task_id: number;
  operation_type: string;
  target_path: string | null;
  command: string | null;
  description: string;
  status: 'pending' | 'approved' | 'denied' | 'expired';
  created_at: string;
  expires_at: string | null;
}

interface PermissionApprovalProps {
  projectId?: number;
  sessionId?: number;
}

const OPERATION_LABELS: Record<string, string> = {
  'file_write': 'Write File',
  'file_delete': 'Delete File',
  'shell_command': 'Execute Command',
  'external_api': 'External API Call',
  'install_dependencies': 'Install Dependencies',
  'execute_script': 'Execute Script',
  'modify_system': 'Modify System',
  'deploy': 'Deploy Application',
};

export default function PermissionApproval({ projectId, sessionId }: PermissionApprovalProps) {
  const [pendingPermissions, setPendingPermissions] = useState<PermissionRequest[]>([]);
  const [history, setHistory] = useState<PermissionRequest[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchPendingPermissions = useCallback(async () => {
    try {
      const params: Record<string, string | number> = { limit: 50 };
      if (projectId) params.project_id = projectId;
      if (sessionId) params.session_id = sessionId;

      const response = await axios.get('/api/v1/permissions/pending', { params });
      setPendingPermissions(response.data.permissions || []);
    } catch (error) {
      console.error('Failed to fetch permissions:', error);
    } finally {
      setLoading(false);
    }
  }, [projectId, sessionId]);

  useEffect(() => {
    fetchPendingPermissions();
  }, [fetchPendingPermissions]);

  const fetchHistory = async () => {
    if (!projectId) return;
    try {
      const response = await axios.get(`/api/v1/permissions/history/${projectId}`, {
        params: { limit: 100 },
      });
      setHistory(response.data.permissions || []);
    } catch (error) {
      console.error('Failed to fetch history:', error);
    }
  };

  const handleApprove = async (requestId: number, autoApproveSame: boolean) => {
    try {
      await axios.post(`/api/v1/permissions/${requestId}/approve`, {
        auto_approve_same: autoApproveSame,
      });
      await fetchPendingPermissions();
      await fetchHistory();
    } catch (error) {
      console.error('Failed to approve permission:', error);
    }
  };

  const handleDeny = async (requestId: number, reason: string) => {
    try {
      await axios.post(`/api/v1/permissions/${requestId}/deny`, { reason });
      await fetchPendingPermissions();
      await fetchHistory();
    } catch (error) {
      console.error('Failed to deny permission:', error);
    }
  };

  const handleCleanup = async () => {
    try {
      await axios.post('/api/v1/permissions/cleanup');
      await fetchPendingPermissions();
    } catch (error) {
      console.error('Failed to cleanup expired permissions:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'approved':
        return 'text-green-600';
      case 'denied':
        return 'text-red-600';
      case 'expired':
        return 'text-gray-400';
      default:
        return 'text-yellow-600';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold">Permission Requests</h2>
        <button
          onClick={handleCleanup}
          className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded text-sm"
        >
          Clean Expired
        </button>
      </div>

      {loading ? (
        <div className="text-center py-8">Loading...</div>
      ) : (
        <>
          {/* Pending Permissions */}
          {pendingPermissions.length > 0 && (
            <div className="space-y-4">
              <h3 className="font-semibold text-gray-700">Pending Requests ({pendingPermissions.length})</h3>
              <div className="space-y-3">
                {pendingPermissions.map((perm) => (
                  <div
                    key={perm.id}
                    className="border border-yellow-300 bg-yellow-50 rounded-lg p-4"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex items-center gap-2">
                        <span className="px-2 py-1 bg-yellow-200 text-yellow-800 text-xs font-medium rounded">
                          {OPERATION_LABELS[perm.operation_type] || perm.operation_type}
                        </span>
                        <span className={`text-xs font-medium ${getStatusColor(perm.status)}`}>
                          {perm.status.toUpperCase()}
                        </span>
                      </div>
                      <div className="text-xs text-gray-500">
                        {new Date(perm.created_at).toLocaleString()}
                      </div>
                    </div>

                    <p className="text-sm text-gray-800 mb-2">{perm.description}</p>

                    {perm.target_path && (
                      <div className="text-xs bg-white p-2 rounded mb-2">
                        <strong>Path:</strong> {perm.target_path}
                      </div>
                    )}

                    {perm.command && (
                      <div className="text-xs bg-white p-2 rounded mb-2">
                        <strong>Command:</strong> {perm.command}
                      </div>
                    )}

                    <div className="flex gap-2">
                      <button
                        onClick={() => handleApprove(perm.id, false)}
                        className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 text-sm"
                      >
                        Approve
                      </button>
                      <button
                        onClick={() => handleApprove(perm.id, true)}
                        className="px-3 py-1 bg-green-200 text-green-800 rounded hover:bg-green-300 text-sm"
                      >
                        Approve & Auto-same
                      </button>
                      <button
                        onClick={() => handleDeny(perm.id, 'User denied')}
                        className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 text-sm"
                      >
                        Deny
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* History */}
          {history.length > 0 && (
            <div className="mt-8">
              <h3 className="font-semibold text-gray-700 mb-2">Permission History</h3>
              <div className="bg-white border rounded-lg overflow-hidden">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                        Type
                      </th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                        Path/Command
                      </th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                        Status
                      </th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                        Date
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {history.slice(0, 20).map((perm) => (
                      <tr key={perm.id}>
                        <td className="px-4 py-2 text-sm text-gray-900">
                          {OPERATION_LABELS[perm.operation_type] || perm.operation_type}
                        </td>
                        <td className="px-4 py-2 text-sm text-gray-500 truncate max-w-xs">
                          {perm.target_path || perm.command || '-'}
                        </td>
                        <td className={`px-4 py-2 text-sm font-medium ${getStatusColor(perm.status)}`}>
                          {perm.status}
                        </td>
                        <td className="px-4 py-2 text-sm text-gray-500">
                          {new Date(perm.created_at).toLocaleString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {pendingPermissions.length === 0 && !loading && (
        <div className="text-center py-8 text-gray-500">
          No pending permission requests
        </div>
      )}
    </div>
  );
}
