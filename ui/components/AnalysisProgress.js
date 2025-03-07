import { useState, useEffect } from 'react';
import { FaSpinner, FaCheckCircle, FaExclamationTriangle } from 'react-icons/fa';

export default function AnalysisProgress({ jobId, onCompleted }) {
  const [status, setStatus] = useState('running');
  const [progress, setProgress] = useState(0);
  const [messages, setMessages] = useState([]);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!jobId) return;
    
    const fetchStatus = async () => {
      try {
        const res = await fetch(`/api/analysis/status?job_id=${jobId}`);
        if (!res.ok) {
          throw new Error(`Failed to fetch status: ${res.status}`);
        }
        
        const data = await res.json();
        setStatus(data.status);
        
        if (data.messages && Array.isArray(data.messages)) {
          setMessages(prevMessages => {
            // Filter out duplicate messages
            const newMessages = data.messages.filter(
              newMsg => !prevMessages.some(
                existingMsg => existingMsg.timestamp === newMsg.timestamp && 
                existingMsg.content === newMsg.content
              )
            );
            return [...prevMessages, ...newMessages].sort((a, b) => a.timestamp - b.timestamp);
          });
        }
        
        // Calculate progress based on workflow steps
        if (data.status === 'completed') {
          setProgress(100);
          setResults(data.results);
          if (onCompleted) onCompleted(data.results);
        } else if (data.status === 'failed') {
          setProgress(100);
          setError(data.error || 'Analysis failed');
        } else {
          // Estimate progress based on messages
          const totalSteps = 5; // Data loading, exploration, hypothesis, validation, insights
          const completedSteps = calculateCompletedSteps(messages);
          setProgress(Math.min(Math.round((completedSteps / totalSteps) * 100), 95));
        }
      } catch (err) {
        console.error('Error fetching status:', err);
        setError(err.message);
      }
    };
    
    const intervalId = setInterval(fetchStatus, 3000);
    fetchStatus(); // Immediate first fetch
    
    return () => clearInterval(intervalId);
  }, [jobId, onCompleted]);
  
  const calculateCompletedSteps = (messages) => {
    const steps = {
      dataLoading: false,
      exploration: false,
      hypothesis: false,
      validation: false,
      insights: false
    };
    
    // Analyze messages to determine which steps are complete
    for (const msg of messages) {
      const content = msg.content?.toString().toLowerCase() || '';
      if (content.includes('data loaded') || content.includes('retrieved data')) {
        steps.dataLoading = true;
      }
      if (content.includes('exploration complete') || content.includes('data profiling')) {
        steps.exploration = true;
      }
      if (content.includes('hypothesis generated') || content.includes('generating hypothesis')) {
        steps.hypothesis = true;
      }
      if (content.includes('validation complete') || content.includes('validating hypothesis')) {
        steps.validation = true;
      }
      if (content.includes('insights synthesized') || content.includes('generating insights')) {
        steps.insights = true;
      }
    }
    
    return Object.values(steps).filter(Boolean).length;
  };

  const getStatusText = () => {
    switch (status) {
      case 'completed':
        return 'Analysis completed successfully';
      case 'failed':
        return `Analysis failed: ${error || 'Unknown error'}`;
      case 'running':
        return 'Analysis in progress...';
      default:
        return `Status: ${status}`;
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'completed':
        return <FaCheckCircle className="text-green-500 mr-2" />;
      case 'failed':
        return <FaExclamationTriangle className="text-dark-highlight mr-2" />;
      default:
        return <FaSpinner className="animate-spin mr-2 text-dark-highlight" />;
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-dark-secondary p-6 rounded-lg">
        <div className="flex items-center mb-4">
          {getStatusIcon()}
          <h3 className="font-medium">{getStatusText()}</h3>
        </div>
        
        <div className="mb-4">
          <div className="h-2 w-full bg-dark-primary rounded-full overflow-hidden">
            <div 
              className="h-full bg-dark-highlight transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <div className="text-right mt-1 text-sm text-dark-muted">
            {progress}% complete
          </div>
        </div>
      </div>
      
      <div className="bg-dark-secondary p-6 rounded-lg">
        <h3 className="font-medium mb-4">Progress Updates</h3>
        {messages.length === 0 ? (
          <p className="text-dark-muted">Waiting for updates...</p>
        ) : (
          <div className="space-y-3 max-h-60 overflow-y-auto">
            {messages.map((msg, index) => (
              <div key={index} className="py-2 border-b border-dark-primary last:border-0">
                <div className="text-sm text-dark-muted">
                  {new Date(msg.timestamp * 1000).toLocaleTimeString()}
                  {msg.sender && <span> - {msg.sender}</span>}
                </div>
                <p className="mt-1">{msg.content}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
} 