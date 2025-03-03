import logging
import numpy as np
import pandas as pd
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import scipy.stats as stats
from scipy import signal
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import lilliefors
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import warnings

from agent_framework_core import (
    BaseAgent, 
    AgentMemory, 
    Message, 
    AgentState,
    DatasetInfo
)

from hypothesis_generator import Hypothesis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HypothesisValidator")

@dataclass
class ValidationResult:
    """Detailed results of a statistical validation test."""
    test_name: str
    statistic: float
    p_value: float
    confidence_level: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "confidence_level": self.confidence_level,
            "additional_metrics": self.additional_metrics,
            "warnings": self.warnings,
            "error": self.error
        }

class StatisticalTest:
    """Base class for statistical tests."""
    
    def __init__(self, name: str, min_sample_size: int = 30):
        self.name = name
        self.min_sample_size = min_sample_size
    
    def validate(self, data: Dict[str, pd.Series], **kwargs) -> ValidationResult:
        """
        Run the statistical test.
        
        Args:
            data: Dictionary mapping column names to pandas Series
            kwargs: Additional test-specific parameters
            
        Returns:
            ValidationResult object with test results
        """
        raise NotImplementedError("Subclasses must implement validate method")
    
    def check_assumptions(self, data: Dict[str, pd.Series]) -> List[str]:
        """
        Check if data meets test assumptions.
        
        Args:
            data: Dictionary mapping column names to pandas Series
            
        Returns:
            List of warning messages if assumptions are violated
        """
        warnings = []
        
        # Check sample size
        for col_name, series in data.items():
            if len(series.dropna()) < self.min_sample_size:
                warnings.append(f"Small sample size for {col_name}: {len(series.dropna())} < {self.min_sample_size}")
        
        return warnings

class CorrelationTest(StatisticalTest):
    """Test for correlation between variables."""
    
    def __init__(self, method: str = "pearson"):
        super().__init__(f"{method}_correlation", min_sample_size=30)
        self.method = method
    
    def validate(self, data: Dict[str, pd.Series], **kwargs) -> ValidationResult:
        if len(data) != 2:
            raise ValueError("Correlation test requires exactly two variables")
        
        col1, col2 = list(data.values())
        
        # Remove rows where either column has NaN
        mask = ~(col1.isna() | col2.isna())
        col1, col2 = col1[mask], col2[mask]
        
        warnings = self.check_assumptions({"col1": col1, "col2": col2})
        
        try:
            if self.method == "pearson":
                statistic, p_value = stats.pearsonr(col1, col2)
            elif self.method == "spearman":
                statistic, p_value = stats.spearmanr(col1, col2)
            elif self.method == "kendall":
                statistic, p_value = stats.kendalltau(col1, col2)
            else:
                raise ValueError(f"Unknown correlation method: {self.method}")
            
            confidence = 1 - p_value if p_value < 0.05 else 0.0
            
            return ValidationResult(
                test_name=self.name,
                statistic=statistic,
                p_value=p_value,
                confidence_level=confidence,
                additional_metrics={
                    "correlation_coefficient": statistic,
                    "sample_size": len(col1)
                },
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                test_name=self.name,
                statistic=0.0,
                p_value=1.0,
                confidence_level=0.0,
                warnings=warnings,
                error=str(e)
            )
    
    def check_assumptions(self, data: Dict[str, pd.Series]) -> List[str]:
        warnings = super().check_assumptions(data)
        
        if self.method == "pearson":
            # Check normality
            for name, series in data.items():
                _, p_value = stats.normaltest(series)
                if p_value < 0.05:
                    warnings.append(f"Data in {name} may not be normally distributed (p={p_value:.4f})")
            
            # Check linearity
            col1, col2 = list(data.values())
            residuals = np.polyfit(col1, col2, 1)
            if np.std(residuals) > 0.5:  # Arbitrary threshold
                warnings.append("Relationship may not be linear")
        
        return warnings

class OutlierTest(StatisticalTest):
    """Test for the presence and significance of outliers."""
    
    def __init__(self, method: str = "zscore"):
        super().__init__(f"{method}_outlier_test")
        self.method = method
    
    def validate(self, data: Dict[str, pd.Series], **kwargs) -> ValidationResult:
        if len(data) != 1:
            raise ValueError("Outlier test requires exactly one variable")
        
        series = list(data.values())[0]
        series = series.dropna()
        
        warnings = self.check_assumptions({"data": series})
        
        try:
            if self.method == "zscore":
                z_scores = np.abs(stats.zscore(series))
                outliers = z_scores > 3
                outlier_count = np.sum(outliers)
                confidence = min(outlier_count / len(series) * 2, 1.0)
                
                return ValidationResult(
                    test_name=self.name,
                    statistic=float(np.max(z_scores)),
                    p_value=float(np.mean(outliers)),
                    confidence_level=confidence,
                    additional_metrics={
                        "outlier_count": int(outlier_count),
                        "outlier_percentage": float(outlier_count / len(series))
                    },
                    warnings=warnings
                )
                
            elif self.method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                outliers = (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))
                outlier_count = np.sum(outliers)
                confidence = min(outlier_count / len(series) * 2, 1.0)
                
                return ValidationResult(
                    test_name=self.name,
                    statistic=float(IQR),
                    p_value=float(np.mean(outliers)),
                    confidence_level=confidence,
                    additional_metrics={
                        "outlier_count": int(outlier_count),
                        "outlier_percentage": float(outlier_count / len(series)),
                        "Q1": float(Q1),
                        "Q3": float(Q3),
                        "IQR": float(IQR)
                    },
                    warnings=warnings
                )
                
            elif self.method == "lof":
                # Local Outlier Factor
                lof = LocalOutlierFactor(contamination="auto")
                outliers = lof.fit_predict(series.values.reshape(-1, 1))
                outlier_count = np.sum(outliers == -1)
                confidence = min(outlier_count / len(series) * 2, 1.0)
                
                return ValidationResult(
                    test_name=self.name,
                    statistic=float(np.mean(lof.negative_outlier_factor_)),
                    p_value=float(outlier_count / len(series)),
                    confidence_level=confidence,
                    additional_metrics={
                        "outlier_count": int(outlier_count),
                        "outlier_percentage": float(outlier_count / len(series))
                    },
                    warnings=warnings
                )
            
            else:
                raise ValueError(f"Unknown outlier detection method: {self.method}")
                
        except Exception as e:
            return ValidationResult(
                test_name=self.name,
                statistic=0.0,
                p_value=1.0,
                confidence_level=0.0,
                warnings=warnings,
                error=str(e)
            )

class DistributionTest(StatisticalTest):
    """Test for distribution characteristics."""
    
    def __init__(self, method: str = "shapiro"):
        super().__init__(f"{method}_distribution_test")
        self.method = method
    
    def validate(self, data: Dict[str, pd.Series], **kwargs) -> ValidationResult:
        if len(data) != 1:
            raise ValueError("Distribution test requires exactly one variable")
        
        series = list(data.values())[0]
        series = series.dropna()
        
        warnings = self.check_assumptions({"data": series})
        
        try:
            if self.method == "shapiro":
                # Test for normality
                statistic, p_value = stats.shapiro(series)
                is_normal = p_value > 0.05
                confidence = 1 - p_value if not is_normal else p_value
                
                # Calculate additional metrics
                skewness = float(stats.skew(series))
                kurtosis = float(stats.kurtosis(series))
                
                return ValidationResult(
                    test_name=self.name,
                    statistic=statistic,
                    p_value=p_value,
                    confidence_level=confidence,
                    additional_metrics={
                        "is_normal": is_normal,
                        "skewness": skewness,
                        "kurtosis": kurtosis
                    },
                    warnings=warnings
                )
                
            elif self.method == "anderson":
                # Anderson-Darling test
                result = stats.anderson(series)
                statistic = result.statistic
                critical_values = result.critical_values
                significance_levels = [15., 10., 5., 2.5, 1.]
                
                # Find the highest significance level where we reject normality
                for sig_level, critical_value in zip(significance_levels, critical_values):
                    if statistic > critical_value:
                        p_value = sig_level / 100
                        break
                else:
                    p_value = 1.0
                
                confidence = 1 - p_value
                
                return ValidationResult(
                    test_name=self.name,
                    statistic=statistic,
                    p_value=p_value,
                    confidence_level=confidence,
                    additional_metrics={
                        "critical_values": critical_values.tolist(),
                        "significance_levels": significance_levels
                    },
                    warnings=warnings
                )
                
            else:
                raise ValueError(f"Unknown distribution test method: {self.method}")
                
        except Exception as e:
            return ValidationResult(
                test_name=self.name,
                statistic=0.0,
                p_value=1.0,
                confidence_level=0.0,
                warnings=warnings,
                error=str(e)
            )

class TimeSeriesTest(StatisticalTest):
    """Test for time series characteristics."""
    
    def __init__(self, method: str = "adf"):
        super().__init__(f"{method}_timeseries_test", min_sample_size=50)
        self.method = method
    
    def validate(self, data: Dict[str, pd.Series], **kwargs) -> ValidationResult:
        if len(data) != 1:
            raise ValueError("Time series test requires exactly one variable")
        
        series = list(data.values())[0]
        series = series.dropna()
        
        warnings = self.check_assumptions({"data": series})
        
        try:
            if self.method == "adf":
                # Augmented Dickey-Fuller test for stationarity
                result = adfuller(series)
                statistic = result[0]
                p_value = result[1]
                confidence = 1 - p_value if p_value < 0.05 else 0.0
                
                return ValidationResult(
                    test_name=self.name,
                    statistic=statistic,
                    p_value=p_value,
                    confidence_level=confidence,
                    additional_metrics={
                        "critical_values": result[4],
                        "is_stationary": p_value < 0.05
                    },
                    warnings=warnings
                )
                
            elif self.method == "kpss":
                # KPSS test for trend stationarity
                result = kpss(series)
                statistic = result[0]
                p_value = result[1]
                confidence = 1 - p_value if p_value < 0.05 else 0.0
                
                return ValidationResult(
                    test_name=self.name,
                    statistic=statistic,
                    p_value=p_value,
                    confidence_level=confidence,
                    additional_metrics={
                        "critical_values": result[3],
                        "is_trend_stationary": p_value > 0.05
                    },
                    warnings=warnings
                )
                
            else:
                raise ValueError(f"Unknown time series test method: {self.method}")
                
        except Exception as e:
            return ValidationResult(
                test_name=self.name,
                statistic=0.0,
                p_value=1.0,
                confidence_level=0.0,
                warnings=warnings,
                error=str(e)
            )

class HypothesisValidator(BaseAgent):
    """
    Agent for validating hypotheses using statistical tests.
    
    This agent applies rigorous statistical testing to validate or reject
    hypotheses generated by the HypothesisGenerator. It uses a variety of
    statistical methods appropriate for different types of relationships
    and data characteristics.
    """
    
    def __init__(self, agent_id: str, memory: AgentMemory):
        """
        Initialize the Hypothesis Validator agent.
        
        Args:
            agent_id: Unique identifier for this agent
            memory: Shared memory system for agent communication
        """
        super().__init__(agent_id, memory)
        self.logger = logging.getLogger(f"HypothesisValidator_{agent_id}")
        
        # Initialize test registry
        self.test_registry = {
            "correlation": {
                "pearson": CorrelationTest("pearson"),
                "spearman": CorrelationTest("spearman"),
                "kendall": CorrelationTest("kendall")
            },
            "outlier": {
                "zscore": OutlierTest("zscore"),
                "iqr": OutlierTest("iqr"),
                "lof": OutlierTest("lof")
            },
            "distribution": {
                "shapiro": DistributionTest("shapiro"),
                "anderson": DistributionTest("anderson")
            },
            "timeseries": {
                "adf": TimeSeriesTest("adf"),
                "kpss": TimeSeriesTest("kpss")
            }
        }
    
    def _execute(self, action: str = "validate_hypotheses", **kwargs) -> Any:
        """
        Execute the specified action.
        
        Args:
            action: The action to execute
            kwargs: Additional arguments specific to the action
            
        Returns:
            The result of the action
        """
        if action == "validate_hypotheses":
            return self.validate_hypotheses(**kwargs)
        elif action == "validate_single_hypothesis":
            return self.validate_single_hypothesis(**kwargs)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def validate_hypotheses(self,
                          dataset_name: str,
                          hypotheses_key: Optional[str] = None,
                          data_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Validate multiple hypotheses for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            hypotheses_key: Key for retrieving hypotheses from memory
            data_key: Key for retrieving dataset from memory
            
        Returns:
            List of validated hypothesis dictionaries
        """
        self.logger.info(f"Validating hypotheses for {dataset_name}")
        
        # Retrieve hypotheses
        h_key = hypotheses_key or f"hypotheses:{dataset_name}"
        hypotheses = self.memory.retrieve(h_key, [])
        
        if not hypotheses:
            error_msg = f"No hypotheses found for {dataset_name}"
            self.logger.error(error_msg)
            return {"error": error_msg}
        
        # Retrieve data
        d_key = data_key or f"data:{dataset_name}"
        data = self.memory.retrieve(d_key)
        
        if data is None:
            error_msg = f"Dataset not found: {dataset_name}"
            self.logger.error(error_msg)
            return {"error": error_msg}
        
        # Convert data to DataFrame if it's a dictionary
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        # Validate each hypothesis
        validated_hypotheses = []
        for hypothesis in hypotheses:
            try:
                result = self.validate_single_hypothesis(
                    hypothesis=hypothesis,
                    data=data
                )
                validated_hypotheses.append(result)
            except Exception as e:
                self.logger.error(f"Error validating hypothesis {hypothesis.id}: {str(e)}")
                # Add error status to hypothesis
                hypothesis.validate(
                    status="error",
                    method="none",
                    result={"error": str(e)}
                )
                validated_hypotheses.append(hypothesis.to_dict())
        
        # Store validated hypotheses
        self.memory.store(h_key, validated_hypotheses)
        
        # Send completion message
        self.send_message(
            recipient="master_planner",
            message_type="hypotheses_validated",
            content={
                "dataset_name": dataset_name,
                "count": len(validated_hypotheses),
                "memory_key": h_key
            }
        )
        
        return validated_hypotheses
    
    def validate_single_hypothesis(self,
                                 hypothesis: Union[Hypothesis, Dict[str, Any]],
                                 data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate a single hypothesis.
        
        Args:
            hypothesis: Hypothesis object or dictionary
            data: DataFrame containing the data
            
        Returns:
            Validated hypothesis dictionary
        """
        # Convert dictionary to Hypothesis object if needed
        if isinstance(hypothesis, dict):
            hypothesis = Hypothesis.from_dict(hypothesis)
        
        # Get relevant data columns
        try:
            column_data = {
                col: data[col] for col in hypothesis.columns_involved
                if col in data.columns
            }
        except KeyError as e:
            self.logger.error(f"Column not found: {str(e)}")
            hypothesis.validate(
                status="error",
                method="none",
                result={"error": f"Column not found: {str(e)}"}
            )
            return hypothesis.to_dict()
        
        # Select appropriate tests based on relationship type
        test_results = self._run_appropriate_tests(
            hypothesis.relationship_type,
            column_data
        )
        
        # Determine overall validation status
        if not test_results:
            status = "error"
            method = "none"
            result = {"error": "No applicable tests found"}
        else:
            # Use the most confident test result
            best_result = max(test_results, key=lambda x: x.confidence_level)
            
            status = "confirmed" if best_result.confidence_level >= 0.95 else \
                    "rejected" if best_result.confidence_level <= 0.05 else \
                    "indeterminate"
            
            method = best_result.test_name
            result = {
                "primary_test": best_result.to_dict(),
                "additional_tests": [r.to_dict() for r in test_results if r != best_result]
            }
        
        # Update hypothesis with validation results
        hypothesis.validate(status, method, result)
        
        return hypothesis.to_dict()
    
    def _run_appropriate_tests(self,
                             relationship_type: str,
                             data: Dict[str, pd.Series]) -> List[ValidationResult]:
        """
        Run appropriate statistical tests based on relationship type.
        
        Args:
            relationship_type: Type of relationship to test
            data: Dictionary mapping column names to pandas Series
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        if relationship_type == "correlation":
            # Run correlation tests
            for test in self.test_registry["correlation"].values():
                results.append(test.validate(data))
        
        elif relationship_type == "outlier":
            # Run outlier tests
            for test in self.test_registry["outlier"].values():
                results.append(test.validate(data))
        
        elif relationship_type == "distribution":
            # Run distribution tests
            for test in self.test_registry["distribution"].values():
                results.append(test.validate(data))
        
        elif relationship_type == "timeseries":
            # Run time series tests
            for test in self.test_registry["timeseries"].values():
                results.append(test.validate(data))
        
        # Filter out failed tests
        results = [r for r in results if r.error is None]
        
        return results 