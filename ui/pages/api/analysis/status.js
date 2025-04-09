import { getToken } from 'next-auth/jwt';

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
    
    // TODO: Implement a mechanism to get job status from the backend.
    // The previous direct import of Python code ('../../../../main') is not feasible here.
    // Example placeholder response:
    let job_status = 'unknown'; // Placeholder
    let results = {}; // Placeholder
    let error_msg = null; // Placeholder
    let messages = []; // Placeholder
    
    // Simulating status check (replace with actual backend communication)
    if (job_id === 'completed_example') { 
      job_status = 'completed';
      results = { insights: ['Example insight 1', 'Example insight 2'] };
    } else if (job_id === 'failed_example') {
      job_status = 'failed';
      error_msg = 'Example failure reason';
    } else {
      job_status = 'processing'; // Default simulation
      messages = [{ sender: 'System', type: 'info', content: 'Analysis in progress...', timestamp: new Date().toISOString() }];
    }

    let response = {
      job_id: job_id,
      status: job_status
    };

    if (job_status === 'completed') {
      response.results = results;
    }
    
    if (job_status === 'failed') {
      response.error = error_msg;
    }
    
    if (messages && messages.length > 0) {
      response.messages = messages;
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