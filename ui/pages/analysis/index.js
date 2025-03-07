import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useSession } from 'next-auth/react';
import Head from 'next/head';
import Layout from '../../components/Layout';
import AnalysisProgress from '../../components/AnalysisProgress';
import InsightsList from '../../components/InsightsList';
import { FaChevronLeft, FaFileDownload, FaExclamationTriangle } from 'react-icons/fa';

export default function Analysis() {
  const router = useRouter();
  const { job_id } = router.query;
  const { data: session, status } = useSession();
  const [analysisResults, setAnalysisResults] = useState(null);
  const [showResults, setShowResults] = useState(false);

  useEffect(() => {
    // Redirect if not authenticated
    if (status === 'unauthenticated') {
      router.push('/');
    }
    
    // Redirect if no job ID
    if (status === 'authenticated' && !job_id && !router.query.new) {
      router.push('/');
    }
  }, [status, router, job_id]);

  const handleAnalysisCompleted = (results) => {
    setAnalysisResults(results);
    setShowResults(true);
  };

  const handleExportResults = () => {
    if (!analysisResults) return;
    
    // Create a JSON blob and download it
    const dataStr = JSON.stringify(analysisResults, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis-results-${job_id}.json`;
    document.body.appendChild(a);
    a.click();
    
    // Clean up
    URL.revokeObjectURL(url);
    document.body.removeChild(a);
  };

  if (status === 'loading') {
    return (
      <Layout>
        <div className="flex items-center justify-center p-8">
          <p>Loading...</p>
        </div>
      </Layout>
    );
  }

  // Determine if we have results to show
  const hasResults = analysisResults && showResults;
  
  // Extract insights from the results if available
  const insights = hasResults && analysisResults.results && 
    analysisResults.results.insight_synthesis ? 
    analysisResults.results.insight_synthesis : [];

  return (
    <Layout>
      <Head>
        <title>Analysis {job_id ? `#${job_id}` : ''} | Data Analysis Agent</title>
      </Head>

      <div className="w-full max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold">
            {hasResults ? 'Analysis Results' : 'Analysis in Progress'}
          </h1>
          
          <div className="flex space-x-3">
            <button
              onClick={() => router.push('/')}
              className="btn btn-secondary flex items-center"
            >
              <FaChevronLeft className="mr-2" />
              Back to Sheets
            </button>
            
            {hasResults && (
              <button
                onClick={handleExportResults}
                className="btn btn-primary flex items-center"
              >
                <FaFileDownload className="mr-2" />
                Export Results
              </button>
            )}
          </div>
        </div>
        
        {!job_id ? (
          <div className="bg-dark-secondary p-6 rounded-lg flex items-start">
            <FaExclamationTriangle className="text-dark-highlight mt-1 mr-3 flex-shrink-0" />
            <div>
              <h3 className="font-medium mb-2">No Analysis Job Selected</h3>
              <p className="text-dark-muted mb-4">
                Please select a Google Sheet to analyze first.
              </p>
              <button 
                onClick={() => router.push('/')} 
                className="btn btn-primary"
              >
                Select a Sheet
              </button>
            </div>
          </div>
        ) : !hasResults ? (
          <AnalysisProgress 
            jobId={job_id} 
            onCompleted={handleAnalysisCompleted} 
          />
        ) : (
          <div className="space-y-8">
            <InsightsList insights={insights} />
            
            {/* Optional: Raw results in a collapsible section */}
            <div className="bg-dark-secondary p-6 rounded-lg">
              <h3 className="font-medium mb-4">Raw Analysis Data</h3>
              <div className="bg-dark-primary p-4 rounded overflow-auto max-h-96">
                <pre className="text-dark-text text-sm">
                  {JSON.stringify(analysisResults, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
} 