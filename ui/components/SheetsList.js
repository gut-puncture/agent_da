import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { FaSpinner, FaFile, FaExclamationTriangle } from 'react-icons/fa';
import { formatDistance } from 'date-fns';

export default function SheetsList() {
  const [sheets, setSheets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const router = useRouter();

  useEffect(() => {
    async function fetchSheets() {
      try {
        setLoading(true);
        const res = await fetch('/api/sheets/list');
        
        if (!res.ok) {
          throw new Error(`Failed to fetch sheets: ${res.status}`);
        }
        
        const data = await res.json();
        setSheets(data.files || []);
      } catch (err) {
        console.error('Error fetching sheets:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    
    fetchSheets();
  }, []);

  const handleSheetClick = (id) => {
    router.push(`/sheets/${id}`);
  };

  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      return formatDistance(date, new Date(), { addSuffix: true });
    } catch (e) {
      return 'Unknown date';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <FaSpinner className="animate-spin h-8 w-8 text-dark-highlight" />
        <p className="ml-4">Loading your spreadsheets...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-dark-secondary p-6 rounded-lg flex items-start">
        <FaExclamationTriangle className="text-dark-highlight mt-1 mr-3 flex-shrink-0" />
        <div>
          <h3 className="font-medium mb-2">Error loading sheets</h3>
          <p className="text-dark-muted mb-4">{error}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="btn btn-primary"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (sheets.length === 0) {
    return (
      <div className="bg-dark-secondary p-6 rounded-lg text-center">
        <h3 className="font-medium mb-4">No Google Sheets Found</h3>
        <p className="text-dark-muted mb-6">
          We couldn't find any Google Sheets in your account. 
          Please make sure you have created some spreadsheets in Google Sheets.
        </p>
        <a 
          href="https://docs.google.com/spreadsheets" 
          target="_blank" 
          rel="noopener noreferrer"
          className="btn btn-primary"
        >
          Create a New Sheet
        </a>
      </div>
    );
  }

  return (
    <div className="grid gap-4">
      <h2 className="text-xl font-semibold mb-4">Your Google Sheets</h2>
      <div className="grid gap-4">
        {sheets.map((sheet) => (
          <div 
            key={sheet.id}
            className="bg-dark-secondary rounded-lg p-4 hover:bg-dark-accent transition-colors cursor-pointer"
            onClick={() => handleSheetClick(sheet.id)}
          >
            <div className="flex items-start">
              <FaFile className="text-dark-highlight mt-1 mr-3 flex-shrink-0" />
              <div className="flex-grow min-w-0">
                <h3 className="font-medium mb-1 truncate">{sheet.name}</h3>
                <p className="text-dark-muted text-sm">
                  Last modified: {formatDate(sheet.modifiedTime)}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
} 