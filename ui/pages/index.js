import { useSession, signIn } from 'next-auth/react';
import Head from 'next/head';
import Layout from '../components/Layout';
import SheetsList from '../components/SheetsList';
import { FaGoogle } from 'react-icons/fa';

export default function Home() {
  const { data: session, status } = useSession();
  const loading = status === 'loading';

  return (
    <Layout>
      <Head>
        <title>Data Analysis Agent</title>
        <meta name="description" content="AI-powered data analysis for Google Sheets" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="w-full max-w-6xl mx-auto">
        {!session && !loading ? (
          <div className="bg-dark-secondary p-8 rounded-lg text-center">
            <h1 className="text-3xl font-bold mb-4">Intelligent Data Analysis</h1>
            <p className="mb-8 text-dark-muted max-w-2xl mx-auto">
              Connect your Google Sheets to our intelligent agent to discover deep insights
              and non-obvious patterns in your data. Our AI-powered analysis helps you make
              better decisions with your data.
            </p>
            <button
              onClick={() => signIn('google')}
              className="flex items-center justify-center mx-auto px-6 py-3 bg-dark-highlight rounded-lg text-white font-medium hover:bg-dark-highlight/90 transition-colors"
            >
              <FaGoogle className="mr-2" />
              Sign in with Google
            </button>
          </div>
        ) : (
          <>
            <h1 className="text-2xl font-bold mb-6">Select a Google Sheet to Analyze</h1>
            {loading ? (
              <p>Loading...</p>
            ) : (
              <SheetsList />
            )}
          </>
        )}
      </div>
    </Layout>
  );
} 