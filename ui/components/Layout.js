import { useSession, signIn, signOut } from 'next-auth/react';
import { useRouter } from 'next/router';
import { FaFileAlt, FaChartBar, FaCog, FaSignOutAlt, FaGoogle } from 'react-icons/fa';
import Link from 'next/link';

export default function Layout({ children }) {
  const { data: session, status } = useSession();
  const router = useRouter();
  const isLoading = status === 'loading';
  
  // Determine which section is active based on the current path
  const isActive = (path) => {
    return router.pathname === path ? 'bg-dark-accent' : '';
  };

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <div className="bg-dark-secondary w-64 text-dark-text flex flex-col">
        <div className="p-4 border-b border-dark-accent">
          <h1 className="text-2xl font-bold text-dark-highlight">Data Analysis Agent</h1>
          <p className="text-dark-muted text-sm">Intelligent Insights</p>
        </div>
        
        <div className="flex-1 py-4">
          {session ? (
            <nav>
              <Link href="/">
                <div className={`flex items-center px-4 py-3 hover:bg-dark-accent cursor-pointer ${isActive('/')}`}>
                  <FaFileAlt className="mr-3" />
                  <span>Select File</span>
                </div>
              </Link>
              <Link href="/analysis">
                <div className={`flex items-center px-4 py-3 hover:bg-dark-accent cursor-pointer ${isActive('/analysis')}`}>
                  <FaChartBar className="mr-3" />
                  <span>Analysis</span>
                </div>
              </Link>
              <Link href="/settings">
                <div className={`flex items-center px-4 py-3 hover:bg-dark-accent cursor-pointer ${isActive('/settings')}`}>
                  <FaCog className="mr-3" />
                  <span>Settings</span>
                </div>
              </Link>
            </nav>
          ) : (
            <div className="p-4 text-center">
              <p className="mb-4">Sign in to access your Google Sheets files</p>
            </div>
          )}
        </div>
        
        <div className="p-4 border-t border-dark-accent">
          {session ? (
            <div>
              <div className="flex items-center mb-4">
                {session.user.image ? (
                  <img 
                    src={session.user.image} 
                    alt={session.user.name} 
                    className="w-8 h-8 rounded-full mr-2" 
                  />
                ) : (
                  <div className="w-8 h-8 bg-dark-highlight rounded-full mr-2 flex items-center justify-center">
                    {session.user.name?.charAt(0).toUpperCase() || 'U'}
                  </div>
                )}
                <div>
                  <p className="text-sm font-medium">{session.user.name || 'User'}</p>
                  <p className="text-xs text-dark-muted">{session.user.email}</p>
                </div>
              </div>
              <button
                onClick={() => signOut()}
                className="w-full flex items-center justify-center px-4 py-2 bg-dark-accent rounded hover:bg-dark-accent/80"
              >
                <FaSignOutAlt className="mr-2" />
                Sign Out
              </button>
            </div>
          ) : (
            <button
              onClick={() => signIn('google')}
              className="w-full flex items-center justify-center px-4 py-2 bg-dark-highlight rounded hover:bg-dark-highlight/80"
              disabled={isLoading}
            >
              <FaGoogle className="mr-2" />
              {isLoading ? 'Loading...' : 'Sign in with Google'}
            </button>
          )}
        </div>
      </div>
      
      {/* Main content */}
      <div className="flex-1 flex flex-col">
        <main className="flex-1 p-6">
          {children}
        </main>
        <footer className="p-4 text-center text-dark-muted text-sm">
          &copy; {new Date().getFullYear()} Data Analysis Agent
        </footer>
      </div>
    </div>
  );
} 