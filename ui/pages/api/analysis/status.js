import { getToken } from 'next-auth/jwt';
import { initialize_system } from '../../../../main';

const secret = process.env.NEXTAUTH_SECRET || 'my-secret';

export default async function handler(req, res) {
  const { 
    query: { job_id },
    method
  } = req;
  
  if (method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  if (!job_id) {
    return res.status(400).json({ error: 'Job ID required' });
  }

  try {
    const token = await getToken({ req, secret });
    
    if (!token) {
      return res.status(401).json({ error: 'Not authenticated' });
    }
    
    // Initialize the system to get access to the memory
    const system = initialize_system();
    const master_planner = system["master_planner"];
    
    // Get the job status from memory
    const job_status = master_planner.memory.retrieve(`job_${job_id}_status`, 'unknown');
    
    let response = {
      job_id: job_id,
      status: job_status
    };
    
    // If the job is completed, include the results
    if (job_status === 'completed') {
      const results = master_planner.memory.retrieve(`job_${job_id}_results`, {});
      response.results = results;
    }
    
    // If the job failed, include the error message
    if (job_status === 'failed') {
      const error = master_planner.memory.retrieve(`job_${job_id}_error`, 'Unknown error');
      response.error = error;
    }
    
    // Get any progress messages if available
    const messages = master_planner.get_messages();
    if (messages && messages.length > 0) {
      response.messages = messages.map(msg => ({
        sender: msg.sender,
        type: msg.message_type,
        content: msg.content,
        timestamp: msg.timestamp
      }));
    }
    
    return res.status(200).json(response);
  } catch (error) {
    console.error('API error:', error);
    return res.status(500).json({ 
      error: 'Failed to fetch job status', 
      message: error.message 
    });
  }
} 