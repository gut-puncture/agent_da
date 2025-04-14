import { getToken } from "next-auth/jwt";

const secret = process.env.NEXTAUTH_SECRET;

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  const { job_id } = req.query;

  if (!job_id) {
    return res.status(400).json({ error: 'Job ID required' });
  }

  try {
    const token = await getToken({ req, secret });

    if (!token) {
      return res.status(401).json({ error: 'Not authenticated' });
    }

    // Forward to backend API
    const backendResponse = await fetch(`${process.env.BACKEND_API_URL}/status?job_id=${job_id}`, {
        headers: {
          'Authorization': `Bearer ${token.accessToken}`,
        }
    });

    if (!backendResponse.ok) {
      const error = await backendResponse.text();
      return res.status(backendResponse.status).json({ error });
    }
    
    const data = await backendResponse.json();
    return res.status(200).json(data);

  } catch (error) {
    console.error('API error:', error);
    return res.status(500).json({ error: 'Failed to get analysis status', message: error.message });
  }
}
