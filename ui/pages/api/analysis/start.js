// Remove import of Python code
import { getToken } from "next-auth/jwt";

const secret = process.env.NEXTAUTH_SECRET;

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

    // Forward request to backend API
    const backendResponse = await fetch(`${process.env.BACKEND_API_URL}/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // Optionally pass auth token in headers if backend requires
        'Authorization': `Bearer ${token.accessToken}`,
      },
      body: JSON.stringify({ spreadsheetId, sheetName, description })
    });

    if (!backendResponse.ok) {
      const error = await backendResponse.text();
      return res.status(backendResponse.status).json({ error });
    }

    const data = await backendResponse.json();

    return res.status(200).json(data);

  } catch (error) {
    console.error('API error:', error);
    return res.status(500).json({ error: 'Failed to start analysis', message: error.message });
  }
}
