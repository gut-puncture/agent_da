import { getToken } from 'next-auth/jwt';
import { initialize_system, run_sample_workflow } from '../../../../main';

const secret = process.env.NEXTAUTH_SECRET || 'my-secret';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { spreadsheetId, sheetName, description } = req.body;

  if (!spreadsheetId || !sheetName) {
    return res.status(400).json({ error: 'Spreadsheet ID and sheet name are required' });
  }

  try {
    const token = await getToken({ req, secret });
    
    if (!token) {
      return res.status(401).json({ error: 'Not authenticated' });
    }
    
    // Initialize the system
    const system = initialize_system();
    const master_planner = system["master_planner"];
    
    // Pass the Google auth token to the Google Sheets connector
    // Note: This would need to be implemented in the connector
    master_planner.memory.store('google_auth_token', token.accessToken);
    
    // Store the description for the data if provided
    if (description) {
      master_planner.memory.store('data_description', description);
    }
    
    // Create the initial data for the workflow
    const initial_data = {
      spreadsheet_id: spreadsheetId,
      sheet_name: sheetName
    };
    
    // Start the analysis asynchronously and return a job ID
    // In a production system, you would use a job queue system like Bull or a serverless approach
    const job_id = `job_${Date.now()}`;
    
    // Store job info in memory
    master_planner.memory.store(`job_${job_id}_status`, 'started');
    master_planner.memory.store(`job_${job_id}_data`, initial_data);
    
    // Start the workflow in a non-blocking way
    setTimeout(() => {
      try {
        const results = run_sample_workflow(master_planner, initial_data);
        master_planner.memory.store(`job_${job_id}_status`, 'completed');
        master_planner.memory.store(`job_${job_id}_results`, results);
      } catch (error) {
        console.error('Analysis error:', error);
        master_planner.memory.store(`job_${job_id}_status`, 'failed');
        master_planner.memory.store(`job_${job_id}_error`, error.message);
      }
    }, 0);
    
    return res.status(200).json({ 
      status: 'started',
      job_id: job_id
    });
  } catch (error) {
    console.error('API error:', error);
    return res.status(500).json({ 
      error: 'Failed to start analysis', 
      message: error.message 
    });
  }
} 