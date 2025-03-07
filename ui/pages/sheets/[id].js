import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useSession } from 'next-auth/react';
import Head from 'next/head';
import Layout from '../../components/Layout';
import { FaSpinner, FaTable, FaChartBar, FaExclamationTriangle } from 'react-icons/fa';

export default function SheetDetails() {
  const router = useRouter();
  const { id } = router.query;
  const { data: session, status } = useSession();
  const [sheetInfo, setSheetInfo] = useState(null);
  const [selectedSheet, setSelectedSheet] = useState('');
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Redirect if not authenticated
    if (status === 'unauthenticated') {
      router.push('/');
    }
  }, [status, router]);

  useEffect(() => {
    if (!id || status !== 'authenticated') return;

    async function fetchSheetInfo() {
      try {
        setLoading(true);
        const res = await fetch(`/api/sheets/info?id=${id}`);
        
        if (!res.ok) {
          throw new Error(`Failed to fetch sheet info: ${res.status}`);
        }
        
        const data = await res.json();
        setSheetInfo(data);
        
        // Select the first sheet by default
        if (data.sheets && data.sheets.length > 0) {
          setSelectedSheet(data.sheets[0].name);
        }
      } catch (err) {
        console.error('Error fetching sheet info:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    
    fetchSheetInfo();
  }, [id, status]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!id || !selectedSheet) {
      return;
    }
    
    try {
      setSubmitting(true);
      
      const res = await fetch('/api/analysis/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          spreadsheetId: id,
          sheetName: selectedSheet,
          description: description.trim() || undefined,
        }),
      });
      
      if (!res.ok) {
        throw new Error(`Failed to start analysis: ${res.status}`);
      }
      
      const data = await res.json();
      
      // Redirect to the analysis page with the job ID
      router.push(`/analysis?job_id=${data.job_id}`);
    } catch (err) {
      console.error('Error starting analysis:', err);
      setError(err.message);
      setSubmitting(false);
    }
  };

  if (status === 'loading' || !id) {
    return (
      <Layout>
        <div className="flex items-center justify-center p-8">
          <FaSpinner className="animate-spin h-8 w-8 text-dark-highlight" />
          <p className="ml-4">Loading...</p>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <Head>
        <title>{sheetInfo?.title || 'Sheet Details'} | Data Analysis Agent</title>
      </Head>

      <div className="w-full max-w-6xl mx-auto">
        {loading ? (
          <div className="flex items-center justify-center p-8">
            <FaSpinner className="animate-spin h-8 w-8 text-dark-highlight" />
            <p className="ml-4">Loading sheet information...</p>
          </div>
        ) : error ? (
          <div className="bg-dark-secondary p-6 rounded-lg flex items-start">
            <FaExclamationTriangle className="text-dark-highlight mt-1 mr-3 flex-shrink-0" />
            <div>
              <h3 className="font-medium mb-2">Error loading sheet information</h3>
              <p className="text-dark-muted mb-4">{error}</p>
              <button 
                onClick={() => router.push('/')} 
                className="btn btn-primary"
              >
                Back to Sheets List
              </button>
            </div>
          </div>
        ) : (
          <>
            <h1 className="text-2xl font-bold mb-6">{sheetInfo?.title}</h1>
            
            <div className="bg-dark-secondary p-6 rounded-lg mb-8">
              <h2 className="text-xl font-medium mb-4">Configure Analysis</h2>
              
              <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                  <label htmlFor="sheet" className="block mb-2 font-medium">
                    Select Sheet to Analyze
                  </label>
                  <div className="relative">
                    <select
                      id="sheet"
                      value={selectedSheet}
                      onChange={(e) => setSelectedSheet(e.target.value)}
                      className="w-full p-3 bg-dark-primary rounded-md border border-dark-accent focus:outline-none focus:ring-2 focus:ring-dark-highlight appearance-none"
                      required
                    >
                      {sheetInfo?.sheets.map((sheet) => (
                        <option key={sheet.id} value={sheet.name}>
                          {sheet.name}
                        </option>
                      ))}
                    </select>
                    <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
                      <FaTable className="text-dark-muted" />
                    </div>
                  </div>
                </div>
                
                <div>
                  <label htmlFor="description" className="block mb-2 font-medium">
                    Data Description (Optional)
                  </label>
                  <textarea
                    id="description"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Describe what this data represents to help the AI understand the context..."
                    className="w-full p-3 min-h-[100px] bg-dark-primary rounded-md border border-dark-accent focus:outline-none focus:ring-2 focus:ring-dark-highlight text-dark-text"
                  />
                </div>
                
                <div className="flex items-center justify-between pt-4">
                  <button
                    type="button"
                    onClick={() => router.push('/')}
                    className="btn btn-secondary"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="btn btn-primary flex items-center"
                    disabled={submitting}
                  >
                    {submitting ? (
                      <>
                        <FaSpinner className="animate-spin mr-2" />
                        Starting Analysis...
                      </>
                    ) : (
                      <>
                        <FaChartBar className="mr-2" />
                        Start Analysis
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>
          </>
        )}
      </div>
    </Layout>
  );
} 