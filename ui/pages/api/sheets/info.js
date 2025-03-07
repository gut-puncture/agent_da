import { getToken } from 'next-auth/jwt';
import { google } from 'googleapis';

const secret = process.env.NEXTAUTH_SECRET || 'my-secret';

export default async function handler(req, res) {
  const { 
    query: { id },
    method
  } = req;
  
  if (method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  if (!id) {
    return res.status(400).json({ error: 'Sheet ID required' });
  }

  try {
    const token = await getToken({ req, secret });
    
    if (!token) {
      return res.status(401).json({ error: 'Not authenticated' });
    }

    const auth = new google.auth.OAuth2();
    auth.setCredentials({
      access_token: token.accessToken,
    });

    const sheets = google.sheets({ version: 'v4', auth });
    
    // Get the spreadsheet metadata
    const spreadsheet = await sheets.spreadsheets.get({
      spreadsheetId: id,
      fields: 'properties.title,sheets.properties'
    });
    
    // Extract sheet names
    const sheetNames = spreadsheet.data.sheets.map(sheet => ({
      id: sheet.properties.sheetId,
      name: sheet.properties.title,
    }));
    
    return res.status(200).json({ 
      title: spreadsheet.data.properties.title,
      sheets: sheetNames
    });
  } catch (error) {
    console.error('API error:', error);
    return res.status(500).json({ 
      error: 'Failed to fetch sheet info', 
      message: error.message 
    });
  }
} 