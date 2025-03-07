import { getToken } from 'next-auth/jwt';
import { google } from 'googleapis';

const secret = process.env.NEXTAUTH_SECRET || 'my-secret';

export default async function handler(req, res) {
  try {
    const token = await getToken({ req, secret });
    
    if (!token) {
      return res.status(401).json({ error: 'Not authenticated' });
    }

    const auth = new google.auth.OAuth2();
    auth.setCredentials({
      access_token: token.accessToken,
    });

    const drive = google.drive({ version: 'v3', auth });
    
    // Search for Google Sheets files
    const response = await drive.files.list({
      q: "mimeType='application/vnd.google-apps.spreadsheet' and trashed=false",
      fields: 'files(id, name, createdTime, modifiedTime, webViewLink)',
      orderBy: 'modifiedTime desc',
      pageSize: 25,
    });

    return res.status(200).json({ files: response.data.files || [] });
  } catch (error) {
    console.error('API error:', error);
    return res.status(500).json({ 
      error: 'Failed to fetch sheets', 
      message: error.message 
    });
  }
} 