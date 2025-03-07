import { useState } from 'react';
import { FaLightbulb, FaChevronDown, FaChevronUp } from 'react-icons/fa';

export default function InsightsList({ insights = [] }) {
  const [expandedInsight, setExpandedInsight] = useState(null);

  const toggleInsight = (id) => {
    if (expandedInsight === id) {
      setExpandedInsight(null);
    } else {
      setExpandedInsight(id);
    }
  };

  if (!insights || insights.length === 0) {
    return (
      <div className="bg-dark-secondary p-6 rounded-lg text-center">
        <h3 className="font-medium mb-4">No Insights Available</h3>
        <p className="text-dark-muted">
          The analysis hasn't generated any insights yet.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold">Generated Insights</h2>
      <div className="space-y-4">
        {insights.map((insight) => (
          <div 
            key={insight.id} 
            className="bg-dark-secondary rounded-lg overflow-hidden transition-shadow duration-300"
          >
            <div 
              className="p-4 flex items-start cursor-pointer"
              onClick={() => toggleInsight(insight.id)}
            >
              <FaLightbulb className="text-dark-highlight mt-1 mr-3 flex-shrink-0" />
              <div className="flex-grow">
                <div className="flex justify-between items-center">
                  <h3 className="font-medium">{insight.title}</h3>
                  {expandedInsight === insight.id ? (
                    <FaChevronUp className="text-dark-muted" />
                  ) : (
                    <FaChevronDown className="text-dark-muted" />
                  )}
                </div>
                <div className="flex items-center mt-1">
                  <div 
                    className="h-1 w-16 bg-dark-primary rounded-full overflow-hidden mr-2"
                  >
                    <div 
                      className="h-full bg-dark-highlight"
                      style={{ width: `${insight.importance * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm text-dark-muted">
                    {Math.round(insight.importance * 100)}% importance
                  </span>
                </div>
              </div>
            </div>
            
            {expandedInsight === insight.id && (
              <div className="p-4 pt-0 border-t border-dark-primary mt-4">
                <p className="mb-4">{insight.description}</p>
                
                {insight.action_items?.length > 0 && (
                  <div className="mb-4">
                    <h4 className="font-medium mb-2">Actions to Consider:</h4>
                    <ul className="list-disc list-inside pl-4 space-y-1">
                      {insight.action_items.map((action, index) => (
                        <li key={index}>{action}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {insight.source_hypotheses?.length > 0 && (
                  <div className="mb-4">
                    <h4 className="font-medium mb-2">Based on:</h4>
                    <div className="text-sm text-dark-muted">
                      {insight.source_hypotheses.join(', ')}
                    </div>
                  </div>
                )}
                
                {insight.tags?.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-4">
                    {insight.tags.map((tag, index) => (
                      <span 
                        key={index}
                        className="px-2 py-1 bg-dark-accent text-xs rounded-full"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
} 